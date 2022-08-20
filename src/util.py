import logging
import os
import random
import string
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial
from typing import List

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data_conf import BOS_IDX, EOS_IDX, STEP_IDX, PAD_IDX

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")


class NamedEnum(Enum):
    def __str__(self):
        return self.value


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().norm(norm_type)
            total_norm += param_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


class WarmupInverseSquareRootSchedule(LambdaLR):
    """Linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_factor = warmup_steps**0.5
        super(WarmupInverseSquareRootSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5


def maybe_mkdir(filename):
    """
    maybe mkdir
    """
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def get_logger(log_file, log_level="info"):
    """
    create logger and output to file and stdout
    """
    assert log_level in ["info", "debug"]
    log_formatter = LogFormatter()
    logger = logging.getLogger()
    log_level = {"info": logging.INFO, "debug": logging.DEBUG}[log_level]
    logger.setLevel(log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(log_formatter)
    logger.addHandler(stream)
    filep = logging.FileHandler(log_file, mode="a")
    filep.setFormatter(log_formatter)
    logger.addHandler(filep)
    return logger


def get_temp_log_filename(prefix="exp", dir="scratch/explog"):
    id = id_generator()
    fp = f"{dir}/{prefix}-{id}"
    maybe_mkdir(fp)
    return fp


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


###################33333

class Unpacker(object):
    @classmethod
    def truncate_sequences(cls, batch, lengths=None):
        if lengths is None:
            lengths = [len(x) for x in batch]
        return [seq[:l] for seq,l in zip(batch, lengths)]

    @classmethod
    def compute_ids_lengths(cls, seqs_list, eos_id=EOS_IDX, include_eos=True):
        return [seq.index(eos_id) + (1 if include_eos else 0) for seq in seqs_list]

    @classmethod
    def batch_tensor_to_list(cls, batch, transpose=True):
        if transpose:
            batch = batch.transpose(0, 1)
        batch = batch.cpu().numpy()
        bs, seq_len = batch.shape
        output = []
        for i in range(bs):
            seq = []
            for j in range(seq_len):
                elem = batch[i, j]
                seq.append(elem)
            output.append(seq)
        return output

    @classmethod
    def unpack_batch(cls, batch, lengths=None, transpose=True):
        if not isinstance(batch, list):
            batch = cls.batch_tensor_to_list(batch, transpose=transpose)
        if lengths is None:
            lengths = cls.compute_ids_lengths(batch)
        seqs = cls.truncate_sequences(batch, lengths=lengths)
        return seqs, lengths

    @classmethod
    def unpack_trg_preds(cls, trg, preds, transpose_preds=True):
        trg_seqs, trg_lens = cls.unpack_batch(trg)
        pred_seqs, pred_lens = cls.unpack_batch(preds, lengths=trg_lens, transpose=transpose_preds)
        return trg_seqs, pred_seqs

#########################333


def unpack_batch(batch, lengths=None):
    if isinstance(batch, list) and isinstance(batch[0], list):
        if lengths is None:
            r = [[char for char in seq[:l]] for seq,l in zip(batch, lengths)]
        else:
            r = [[char for char in seq] for seq in batch]
        return r
        #return [
        #    [char for char in seq if char != BOS_IDX and char != EOS_IDX]
        #    for seq in batch
        #]
    batch = batch.transpose(0, 1).cpu().numpy()
    bs, seq_len = batch.shape
    output = []
    if lengths is None:
        lengths = [None]*bs

    for i,l in zip(range(bs), lengths):
        seq = []
        for j in range(seq_len):
            elem = batch[i, j]
            seq.append(elem)
        if l:
            seq = seq[:l]
        output.append(seq)
    return output


@dataclass
class Eval:
    desc: str
    long_desc: str
    res: float


class Evaluator(object):
    def __init__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def evaluate(self, predict, ground_truth):
        raise NotImplementedError

    def add(self, source, predict, target):
        raise NotImplementedError

    def compute(self, reset=True) -> List[Eval]:
        raise NotImplementedError

    def evaluate_all(
        self, data_iter, batch_size, nb_data, model, decode_fn, mean_acc=None
    ) -> List[Eval]:
        for src, src_mask, trg, trg_mask in tqdm(data_iter(batch_size), total=nb_data):
            pred, _ = decode_fn(model, src, src_mask)
            self.add(src, pred, trg)
        return self.compute(reset=True)


class BasicEvaluator(Evaluator):
    """docstring for BasicEvaluator"""

    def __init__(self):
        self.correct = 0
        self.distance = 0
        self.nb_sample = 0

    def reset(self):
        self.correct = 0
        self.distance = 0
        self.nb_sample = 0

    def evaluate(self, predict, ground_truth):
        """
        evaluate single instance
        """
        correct = 1
        if len(predict) == len(ground_truth):
            for elem1, elem2 in zip(predict, ground_truth):
                if elem1 != elem2:
                    correct = 0
                    break
        else:
            correct = 0
        dist = edit_distance(predict, ground_truth)
        return correct, dist

    def add(self, source, predict, target):
        #print()
        #print(source[:, 0])
        #print()
        #print(target[:, 0])
        #print()
        #print(predict[:, 0])

        #predict = unpack_batch(predict)
        #target = unpack_batch(target)
        target, predict = Unpacker.unpack_trg_preds(target, predict)

        for p, t in zip(predict, target):

            correct, distance = self.evaluate(p, t)
            #print("correct:", correct)
            #print("dist:", distance)
            #input(">>>")

            self.correct += correct
            self.distance += distance
            self.nb_sample += 1

    def compute(self, reset=True):
        accuracy = round(self.correct / self.nb_sample * 100, 4)
        distance = round(self.distance / self.nb_sample, 4)
        if reset:
            self.reset()
        return [
            Eval("acc", "accuracy", accuracy),
            Eval("dist", "average edit distance", distance),
        ]


class HistnormEvaluator(BasicEvaluator):
    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)


class G2PEvaluator(BasicEvaluator):
    def __init__(self):
        self.src_dict = defaultdict(list)

    def reset(self):
        self.src_dict = defaultdict(list)

    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)

    def add(self, source, predict, target):
        source = unpack_batch(source)
        predict = unpack_batch(predict)
        target = unpack_batch(target)
        for s, p, t in zip(source, predict, target):
            correct, distance = self.evaluate(p, t)
            self.src_dict[str(s)].append((correct, distance))

    def compute(self, reset=True):

        correct, distance, nb_sample = 0, 0, 0
        for evals in self.src_dict.values():
            corr, dist = evals[0]
            for c, d in evals:
                if c > corr:
                    corr = c
                if d < dist:
                    dist = d
            correct += corr
            distance += dist
            nb_sample += 1
        acc = round(correct / nb_sample * 100, 4)
        distance = round(distance / nb_sample, 4)
        if reset:
            self.reset()
        return [
            Eval("acc", "accuracy", acc),
            Eval("per", "phenome error rate", distance),
        ]


class P2GEvaluator(G2PEvaluator):
    def compute(self, reset=True):
        results = super().compute(reset=reset)
        return [results[0], Eval("ger", "grapheme error rate", results[1].res)]


class PairBasicEvaluator(BasicEvaluator):
    """docstring for PairBasicEvaluator"""

    def evaluate(self, predict, ground_truth):
        """
        evaluate single instance
        """
        predict = [x for x in predict if x != STEP_IDX]
        ground_truth = [x for x in ground_truth if x != STEP_IDX]
        return super().evaluate(predict, ground_truth)


class PairG2PEvaluator(PairBasicEvaluator, G2PEvaluator):
    pass



class TranslitEvaluator(BasicEvaluator):
    """docstring for TranslitEvaluator"""

    def __init__(self):
        self.src_dict = defaultdict(list)

    def reset(self):
        self.src_dict = defaultdict(list)

    def add(self, source, predict, target):
        source = unpack_batch(source)
        predict = unpack_batch(predict)
        target = unpack_batch(target)

        #print("target:")
        #print(target)
        target, predict = Unpacker.unpack_trg_preds(target, predict)
        source = Unpacker.unpack_batch(source)
        for s, p, t in zip(source, predict, target):
            #eow = t.index(EOS_IDX) + 1
            #p, t = p[:eow], t[:eow]
            correct, distance = self.evaluate(p, t)
            self.src_dict[str(s)].append((correct, distance, len(p), len(t)))
            #print()
            #print("t:", t)
            #print("p:", p)
            #print("correct:", correct)
            #print("dist:", distance)
            #input(">>>>")



    def compute(self, reset=True):
        correct, fscore, nb_sample = 0, 0, 0
        for evals in self.src_dict.values():
            corr, dist, pred_len, trg_len = evals[0]
            for c, d, pl, tl in evals:
                if c > corr:
                    corr = c
                if d < dist:
                    dist = d
                    pred_len = pl
                    trg_len = tl
            lcs = (trg_len + pred_len - dist) / 2
            r = lcs / trg_len
            try:
                p = lcs / pred_len
            except ZeroDivisionError:
                p = 0
            f = 2 * r * p / (r + p)

            correct += corr
            fscore += f
            nb_sample += 1

        acc = round(correct / nb_sample * 100, 4)
        mean_fscore = round(fscore / nb_sample, 4)
        if reset:
            self.reset()

        #print("translit evaluator compute()")
        #input(">>>")

        return [
            Eval("acc", "accuracy", acc),
            Eval("meanfs", "mean F-score", mean_fscore),
        ]

class TranslitEvaluator2(BasicEvaluator):
    """docstring for TranslitEvaluator"""

    def __init__(self):
        self.src_dict = defaultdict(list)

    def reset(self):
        self.src_dict = defaultdict(list)

    def add(self, source, predict, target):
        source = unpack_batch(source)
        predict = unpack_batch(predict)
        target = unpack_batch(target)
        for s, p, t in zip(source, predict, target):
            eow = t.index(EOS_IDX) + 1
            p, t = p[:eow], t[:eow]
            print()
            print("t:", t)
            print("p:", p)
            correct, distance = self.evaluate(p, t)
            self.src_dict[str(s)].append((correct, distance))
            print("correct:", correct)
            print("dist:", distance)
            input(">>>>")


    def compute(self, reset=True):
        correct, fscore, nb_sample = 0, 0, 0
        for evals in self.src_dict.values():
            corr, dist, pred_len, trg_len = evals[0]
            for c, d, pl, tl in evals:
                if c > corr:
                    corr = c
                if d < dist:
                    dist = d
                    pred_len = pl
                    trg_len = tl
            lcs = (trg_len + pred_len - dist) / 2
            r = lcs / trg_len
            try:
                p = lcs / pred_len
            except ZeroDivisionError:
                p = 0
            f = 2 * r * p / (r + p)

            correct += corr
            fscore += f
            nb_sample += 1

        acc = round(correct / nb_sample * 100, 4)
        mean_fscore = round(fscore / nb_sample, 4)
        if reset:
            self.reset()

        print("translit evaluator 2 compute()")
        input(">>>")


        return [
            Eval("acc", "accuracy", acc),
            Eval("meanfs", "mean F-score", mean_fscore),
        ]


class DNTransformerEvaluator(TranslitEvaluator):
    def __init__(self, i2c):
        self.src_dict = defaultdict(list)
        self.i2c = i2c

        # vocabulary error counts
        self.vec = defaultdict(lambda : defaultdict(int))

    def reset(self):
        self.src_dict = defaultdict(list)
        self.vec = defaultdict(lambda: defaultdict(int))

    def evaluate(self, predict, ground_truth, pred_mask):
        """
        evaluate single instance
        """
        correct = 1
        voc = defaultdict(lambda: defaultdict(int))
        confusions = defaultdict(lambda: defaultdict(int))
        #print("\npred_mask:", pred_mask)
        #print("predict:", predict)
        #print("ground_truth:", ground_truth)

        non_masked = [(p_elem, t_elem) for p_elem, t_elem, m in zip(predict, ground_truth, pred_mask) if m ]
        num_corr = 0
        for p_elem, t_elem  in non_masked:
            if p_elem != t_elem:
                correct = 0
                voc[t_elem]["fn"] += 1
                voc[p_elem]["fp"] += 1
                confusions[t_elem][p_elem] += 1
            else:
                voc[t_elem]["tp"] += 1
                num_corr += 1
        #dist = edit_distance(predict, ground_truth)
        return correct, voc, confusions, len(non_masked), num_corr


    def add(self, source, predict, target, char_mask):
        target = Unpacker.batch_tensor_to_list(target)
        #print("add() predict:", predict)
        predict = Unpacker.batch_tensor_to_list(predict)
        #print("add() predict:", predict)

        source = Unpacker.batch_tensor_to_list(source)
        char_mask = Unpacker.batch_tensor_to_list(char_mask)

        for s, p, t, m in zip(source, predict, target, char_mask):
            correct, voc, confusions, num_char_preds, num_corr = self.evaluate(p, t, m)
            self.src_dict[str(s)].append((correct, voc, confusions, num_char_preds, num_corr))

    def _compute_accuracy(self):
        correct, nb_sample = 0, 0
        for evals in self.src_dict.values():
            corr, dist, pred_len, trg_len = evals[0]
            for c, d, pl, tl in evals:
                if c > corr:
                    corr = c
            correct += corr
            nb_sample += 1

        acc = round(correct / nb_sample * 100, 4)
        return acc

    @staticmethod
    def _try_divide(a, b, default=0):
        try:
            r = a / b
        except ZeroDivisionError:
            r = default
        return r

    @staticmethod
    def compute_p_r_f_s(err):
        #p = err["tp"] / (err["tp"] + err["fp"])
        #r = err["tp"] / (err["tp"] + err["fn"])
        p = DNTransformerEvaluator._try_divide(err["tp"], (err["tp"] + err["fp"]))
        r = DNTransformerEvaluator._try_divide(err["tp"], (err["tp"] + err["fn"]))
        f = DNTransformerEvaluator._try_divide((2 * p * r), (p + r))
        tn = DNTransformerEvaluator._try_divide(err["tn"], (err["tn"] + err["fp"]))
        return p, r, f, tn

    def compute(self, reset=True, mean_acc=None):
        correct = 0
        nb_sample = 0
        corr_chars = 0
        num_chars = 0

        confusions = defaultdict(Counter)
        errors = defaultdict(Counter)
        all_err = Counter()

        for evals in self.src_dict.values():
            corr, voc, conf, num_char_preds, num_charm_corr = evals[0]
            correct += corr
            nb_sample += 1
            num_chars += num_char_preds
            corr_chars += num_charm_corr
            for t_elem, p_elem in conf.items():
                confusions[t_elem].update(p_elem)
            for term, error_types in voc.items():
                errors[term].update(error_types)
                all_err.update(error_types)

            confusions.update(conf)

        p, r, f, tn = self.compute_p_r_f_s(all_err)

        term_results = []
        for term, error_types in errors.items():
            p, r, f, tn = self.compute_p_r_f_s(error_types)
            term_results.append({
                "symbol": term,
                "p": p,
                "r": r,
                "f": f,
                "tn": tn
            })

        word_acc = round(correct / nb_sample * 100, 4)
        evals = [
            Eval("acc", "character accuracy", corr_chars / num_chars),
            Eval("P", "true negative rate", tn),
            Eval("R", "character recall", p),
            Eval("P", "character precision", p),
            Eval("F", "character F-1", f),
            Eval("wacc", "word accuracy", word_acc),
        ]

        if reset:
            self.reset()

        return evals, term_results


    def evaluate_all(self, data_iter, batch_size, nb_data, model, decode_fn, mean_acc=None) -> List[Eval]:

        N = 5
        if mean_acc is None:
            # print a random batch outputs
            print_idx = random.randint(1, nb_data)
            print_idx = None
            for j, bd in enumerate(tqdm(data_iter(batch_size), total=nb_data)):
                if len(bd) == 4:
                    src, src_mask, trg, trg_mask = bd
                    loss_mask = None
                else:
                    (src, src_mask, trg, trg_mask, loss_mask) = bd
                #if is_first:
                #    self.print_examples(self.i2c, N, loss_mask, None, src, trg)
                pred, _ = decode_fn(model, src, src_mask)
                #print("pred:")
                #print(pred)
                self.add(src, pred, trg, loss_mask)
                #print("loss_mask:", loss_mask.size())
                if print_idx == j:
                    self.print_examples(self.i2c, N, loss_mask, pred, src, trg)
                #input(">>>")

        return self.compute(reset=True)





    @staticmethod
    def print_examples(i2c, N, loss_mask, pred, src, trg, logger=None):
        if pred is None:
            u_pred = [[str(None)]] * N
        else:
            u_pred = unpack_batch(pred)
        u_trg = unpack_batch(trg)
        u_src = unpack_batch(src)

        if loss_mask is None:
            u_lm = [[str(None)]] * N
        else:
            u_lm = unpack_batch(loss_mask)

        if logger:
            logger.info("\nSome Examples")
            for ss, p, t, lm in zip(u_src[:N], u_pred[:N], u_trg[:N], u_lm[:N]):
                length = t.index(PAD_IDX) if PAD_IDX in t else len(t)
                logger.info("-" * 10)
                logger.info(f"input: {' '.join([i2c[j] for j in ss[:length]])}")
                logger.info(f"targ: {' '.join([i2c[int(j)] for j in t[:length]])}")
                logger.info(f"pred: {None if pred is None else ' '.join([i2c[j] for j in p[:length]])}" )
                logger.info(f"LM: {lm[:length]}")
                #if pred is not None:
                #    print()
                #    print("t")
                #    print(t)
                #    print(len(t))
                #    print("p")
                #    print(p)
                #    print(len(p))
                #    print("length:", length)
                #    input(">>>")


        else:
            print("Some Examples")
            for ss, p, t, lm in zip(u_src[:N], u_pred[:N], u_trg[:N], u_lm[:N]):
                print("-" * 10)
                print("input:", [i2c[j] for j in ss])
                print("targ:", [i2c[int(j)] for j in t])
                print("pred:", None if pred is None else " ".join([i2c[j] for j in p]) )
                print("LM:", lm)








class PairTranslitEvaluator(PairBasicEvaluator, TranslitEvaluator):
    pass


def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
    return int(table[len(str2)][len(str1)])



def main():
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    MASK = "?"

    while True:
        e_input = [MASK if random.random() < 0.25 else c for c in alph]
        e_input = "".join(e_input)
        print("e_input:", e_input)



        input(">>>")












if __name__ == "__main__":
    main()



