"""
train
"""
import os
from functools import partial

import torch
from tqdm import tqdm

import dn_dataloader
import dataloader
import model
import transformer
import util
from decoding import Decode, get_decode_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train import Trainer, Data, Arch
import train
import trainer
import dn_transformer

tqdm.monitor_interval = 0
tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")

from vocabulary import Vocabulary


class DNTrainer(Trainer):
    """docstring for Trainer."""

    def set_args(self):
        """
        get_args
        """
        # fmt: off
        super().set_args()
        parser = self.parser
        parser.add_argument('--share_embed', default=False, action='store_true', help='share src and trg embeddings.')
        # fmt: on
        parser.add_argument('--mask_prob', default=0.15, type=float, help='dropout prob')
        parser.add_argument('--mask_mask_prob', default=0.8, type=float, help='dropout prob')
        parser.add_argument('--mask_random_prob', default=0.15, type=float, help='dropout prob')
        parser.add_argument('--eval_steps', default=None, type=int, help='dropout prob')
        parser.add_argument('--finetuning', default=False, action='store_true',
                            help='tie decoder input & output embeddings')
        parser.add_argument('--vocab_file', default=None, type=str, help='saved vocabulary json')


    def load_data(self, dataset, train, dev, test):
        assert self.data is None
        params = self.params
        if dataset in {Data.news15, Data.net, Data.dakshina}:
            self.data = dataloader.Transliteration(train, dev, test, params.shuffle,
                                                   vocab_file=params.vocab_file)
        elif dataset in {Data.denoise}:
            self.data = dn_dataloader.DeNoising(train_file=train,
                                     dev_file=dev,
                                     test_file=test,
                                     shuffle=params.shuffle,
                                     mask_prob=params.mask_prob,
                                     mask_mask_prob=params.mask_mask_prob,
                                     mask_random_prob=params.mask_random_prob,
                                     vocab_file=params.vocab_file)
        else:
            super(DNTrainer, self).load_data(dataset, train, dev, test)
        self.logger.info("src vocab size %d", self.data.source_vocab_size)
        self.logger.info("trg vocab size %d", self.data.target_vocab_size)
        self.logger.info("src vocab %r", self.data.source[:500])
        self.logger.info("trg vocab %r", self.data.target[:500])

    def get_model_class(self, params):
        model_class, kwargs = super(DNTrainer, self).get_model_class(params)
        #if params.finetuning:
        #    pass
        return model_class, kwargs

    def build_model(self):
        assert self.model is None
        params = self.params
        if params.arch == Arch.hardmono:
            params.indtag, params.mono = True, True
        model_class, kwargs = self.get_model_class(params)
        kwargs["share_embeddings"] = params.share_embed
        if model_class is None:
            model_class = {
                Arch.transformer: transformer.Transformer,
                Arch.dntransformer: dn_transformer.DNTransformer,
                Arch.fdntransformer: dn_transformer.FinetunedDNTransformer
            }.get(params.arch)

        #print("build model:", model_class)
        #input(">>>")

        self.model = model_class(**kwargs)
        if params.indtag:
            self.logger.info("number of attribute %d", self.model.nb_attr)
            self.logger.info("dec 1st rnn %r", self.model.dec_rnn.layers[0])
        if params.arch in {Arch.softinputfeed, Arch.approxihardinputfeed, Arch.largesoftinputfeed}:
            self.logger.info("merge_input with %r", self.model.merge_input)
        self.logger.info("model: %r", self.model)
        self.logger.info("number of parameter %d", self.model.count_nb_params())
        self.model = self.model.to(self.device)



    def setup_evalutator(self):
        arch, dataset = self.params.arch, self.params.dataset
        if dataset in {Data.denoise}:
            self.evaluator = util.DNTransformerEvaluator(self.data.source)
        elif dataset in {Data.dakshina, Data.net}:
            self.evaluator = util.TranslitEvaluator()
            #self.evaluator = util.BasicEvaluator()
        else:
            super().setup_evalutator()
        #print("evaluator:", self.evaluator)
        #input(">>>")

    def evaluate(self, mode, batch_size, epoch_idx, decode_fn, mean_acc=None):
        self.model.eval()
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        results = self.evaluator.evaluate_all(
            sampler, batch_size, nb_batch, self.model, decode_fn, mean_acc=mean_acc
        )
        for result in results:
            self.logger.info(
                f"{mode} {result.long_desc} is {result.res} at epoch {epoch_idx}"
            )
        return results


    def predict_and_decode(self, batch_data, decode_fn):
        if len(batch_data) == 4:
            src, src_mask, trg, trg_mask = batch_data
        else:
            src, src_mask, trg, trg_mask, loss_mask = batch_data
        pred, _ = decode_fn(self.model, src, src_mask)
        self.evaluator.add(src, pred, trg)
        # data = (src, src_mask, trg, trg_mask)
        losses, acc = self.model.get_loss(batch_data, reduction=False)
        losses = losses.cpu()
        return losses, pred, trg


    def _score_update_best(self, best_res, m):
        if type(self.evaluator) == util.DNTransformerEvaluator:
            er = m.evaluation_result
            ber = best_res.evaluation_result
            if len(er) == 1 and er[0].res >= ber.res:
                best_res = m
            elif len(er) >= 2 and er[0].res >= ber.res and er[1].res >= ber.res:
                best_res = m
        else:
            best_res = super()._score_update_best(best_res, m)

        return best_res

    def _evaluate(self, decode_fn, epoch_idx, bs, save=True):
        with torch.no_grad():
            devloss = self.calc_loss(trainer.DEV, bs, epoch_idx, return_acc=True)
            acc = None
            if len(devloss) == 2:
                devloss, acc = devloss
            eval_res = self.evaluate(trainer.DEV, bs, epoch_idx, decode_fn, acc)

        if save:
            self.save_model(None, devloss, eval_res, self.params.model, acc)

        return devloss, eval_res, acc

    def train(self, epoch_idx, batch_size, max_norm, eval_steps=None, decode_fn=None):
        logger, model = self.logger, self.model
        logger.info("At %d-th epoch with lr %f.", epoch_idx, self.get_lr())
        model.train()
        sampler, nb_batch = self.iterate_batch(trainer.TRAIN, batch_size)
        losses, cnt = 0, 0
        train_acc = 0
        is_first = True
        for batch in tqdm(sampler(batch_size), total=nb_batch):
            is_first = self.maybe_print(batch, is_first)
            loss = model.get_loss(batch)
            cnt, losses, train_acc = self.update(cnt, loss, losses, max_norm, model, train_acc)
            if isinstance(self.params.eval_steps, int) and self.global_steps % self.params.eval_steps == 0:
                _, _, _ = self._evaluate(decode_fn, epoch_idx, batch_size, save=True)
                if isinstance(train_acc, float):
                    self.logger.info(f"* Running mean TRAIN MP accuracy: {round(100*train_acc / cnt, 6)}")
                self.logger.info(f"* Running mean TRAIN MP loss: {round(100 * losses / cnt, 6)}")
            if self.global_steps > self.params.max_steps: break
        loss = losses / cnt
        self.logger.info(f"Epoch {epoch_idx}: Running AVG train loss {loss} @steps  ({self.global_steps})")

        return loss

    def maybe_print(self, batch, is_first):
        if is_first and not self.params.finetuning:
            src, src_mask, trg, trg_mask, loss_mask = batch
            self.evaluator.print_examples(self.evaluator.i2c, 5, loss_mask, None, src, trg,
                                          logger=self.logger)
        return False

    def update(self, cnt, loss, losses, max_norm, model, train_acc):
        if isinstance(loss, tuple) and len(loss) == 2:
            loss, acc = loss
            if train_acc is not None and acc is not None:
                train_acc += acc.item()
            else:
                train_acc = None
        else:
            train_acc = None
        self.optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        self.logger.debug("loss %f with total grad norm %f",loss, util.grad_norm(model.parameters()),)
        self.optimizer.step()
        if not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step()
        self.global_steps += 1
        losses += loss.item()
        cnt += 1
        return cnt, losses, train_acc


def main():
    """
    main
    """
    trainer = DNTrainer()
    params = trainer.params
    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    trainer.setup_evalutator()

    if params.load and params.load != "0":
        if params.load == "smart":
            start_epoch = trainer.smart_load_model(params.model) + 1
        else:
            start_epoch = trainer.load_model(params.load) + 1
        trainer.logger.info("continue training from epoch %d", start_epoch)
        trainer.setup_training()
        trainer.load_training(params.model)
    elif params.finetuning:
        assert os.path.isfile(params.init)
        trainer.logger.info(f"Initializing with provided model {params.init}")
        trainer.load_model(params.init)
        trainer.setup_training()
        start_epoch = 0

    else:  # start from scratch
        start_epoch = 0
        trainer.build_model()
        if params.init:
            if os.path.isfile(params.init):
                trainer.load_state_dict(params.init)
            else:
                trainer.dump_state_dict(params.init)
        trainer.setup_training()

    #with torch.no_grad():
    #    trainer.reload_and_test(params.model, params.load, params.bs, decode_fn)
    #trainer.cleanup(params.saveall, save_fps, params.model)

    trainer.run(start_epoch, decode_fn=decode_fn)




if __name__ == "__main__":
    main()
