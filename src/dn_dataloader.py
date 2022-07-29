"""
Created by davan 
7/28/22
"""

from data_conf import (BOS, EOS, PAD, UNK , ALIGN, STEP, MASK,
                       PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX, STEP_IDX, MASK_IDX)

import dataloader


from collections import defaultdict, Counter
import numpy as np
import torch
from tqdm import tqdm
import copy
from typing import Dict, List, Optional
import random
import json



class DeNoising(dataloader.Seq2SeqDataLoader):

    def __init__(self, mask_prob=0.15, mask_mask_prob=0.8, mask_random_prob=0.15, **kwargs):
        super().__init__(**kwargs)
        self.mask_prob = mask_prob
        self.mask_mask_prob = mask_mask_prob
        self.mask_random_prob = mask_random_prob
        #self._lang_vocabs = None
        self.eval_randoms = {}





    def read_file(self, file):
        with open(file, "r") as source_file:
            for line in source_file:
                if line:
                    assert len(line) > 1
                    yield [t for t in line.strip().split() if t]

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.target[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.target[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.target[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK
        assert self.target[UNK_IDX] == UNK
        assert self.target[MASK_IDX] == MASK
        assert self.source[MASK_IDX] == MASK

    def build_vocab(self, special_tokens=None):
        src_set = defaultdict(set)
        self.nb_train = 0
        for src in self.read_file(self.train_file):
            self.nb_train += 1
            lang_id = src[0]
            src_set[lang_id].update(src)

        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is not None:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])

        all_set = []
        for lang, voc_set in src_set.items():
            all_set.extend(list(voc_set))
            src_set[lang] = list(voc_set)
        source = [PAD, BOS, EOS, UNK, MASK] + sorted(all_set)
        self._lang_vocabs = src_set

        return source, list(source)

    def _iter_helper(self, file):
        """
        Reads examples from file and converts them from string tokens to token ids.
        """
        for source in self.read_file(file):
            src = [self.source_c2i[BOS]]
            for s in source:
                src.append(self.source_c2i.get(s, UNK_IDX))
            src.append(self.source_c2i[EOS])
            #print("src:", src)
            #input(">>>")
            yield src, source[0]

    def _batch_sample(self, batch_size, file, shuffle, max_seq_len=None):
        key = self._file_identifier(file)
        if key not in self.batch_data:
            lst = list()
            src_lang = []
            for src, lang_id in tqdm(self._iter_helper(file), desc="read file"):
                lst.append(src)
                src_lang.append(lang_id)
            #src_data, src_mask = self.list_to_tensor([src for src, _ in lst])
            #src_data = [src for src, _ in lst]
            #trg_data, trg_mask = self.list_to_tensor([trg for _, trg in lst])
            self.batch_data[key] = (lst, src_lang)

        src_data, src_lang = self.batch_data[key]
        #src_data, src_mask = self.list_to_tensor([self.random_char(src) for src in src_data])
        nb_example = len(src_data)


        if shuffle:
            idx = np.random.permutation(nb_example)
        else:
            idx = np.arange(nb_example)

        is_first = True
        for start in range(0, nb_example, batch_size):
            idx_ = idx[start : start + batch_size]
            examples = [copy.deepcopy(src_data[dx]) for dx in idx_]
            lang_ids = [copy.deepcopy(src_lang[dx]) for dx in idx_]

            trg_data_b, trg_mask_b = self.list_to_tensor(examples)
            #examples = [self.random_char(e, l) for e, l in zip(examples, lang_ids)]
            _expls, loss_mask = [], []
            for e, l in zip(examples, lang_ids):
                output_mask = True
                x, target_mask = self.random_chars(e, l, output_mask)
                _expls.append(x)
                loss_mask.append(target_mask)

            loss_mask, _ = self.list_to_tensor(loss_mask)
            src_data_b, src_mask_b = self.list_to_tensor(_expls)
            #src_len = int(src_mask_b.sum(dim=0).max().item())
            #trg_len = src_len
            #src_mask_b = src_mask_b[:src_len].to(self.device)
            #trg_mask_b = trg_mask_b[:trg_len].to(self.device)


            #if is_first and not shuffle:
            #    tutil.print_examples(self.source, 5, loss_mask, None, src_data_b, trg_data_b)
            #    is_first = False
            yield (src_data_b.to(self.device), src_mask_b.to(self.device),
                   trg_data_b.to(self.device), trg_mask_b.to(self.device),
                   loss_mask.to(self.device))


    def list_to_tensor(self, lst: List[List[int]], max_seq_len=None):
        max_len = max([len(x) for x in lst])
        if max_seq_len is not None:
            max_len = min(max_len, max_seq_len)
        data = torch.zeros((max_len, len(lst)), dtype=torch.long)
        #print("data shape:", data.size())
        #print("max_len:", max_len)
        #print("len lst:", len(lst))
        for i, seq in enumerate(lst):
            #print("seq:", seq)
            data[: len(seq), i] = torch.tensor(seq)

        #print("data:")
        #print(data)

        mask = (data > 0).float()
        return data, mask

    def random_chars(self, token_ids, lang_id, return_output_label=False, error_all_outputs=False):
        """
        from:
        https://github.com/huggingface/transformers/blob/f9cde97b313c3218e1b29ea73a42414
        dfefadb40/examples/lm_finetuning/simple_lm_finetuning.py#L276-L301

        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []
        j = random.randint(1, len(token_ids) - 2)
        jtok = token_ids[j]
        self.add_noise(j, lang_id, token_ids)

        for i, token in enumerate(token_ids):
            prob = random.random()
            #print(f"i={i} prob={prob} token={token}")
            # BERT: mask token with 15% probability
            #if token not in {BOS_IDX, EOS_IDX} and prob < self.mask_prob:
            if prob < self.mask_prob:
                self.add_noise(i, lang_id, token_ids)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(token if error_all_outputs else PAD_IDX)

        output_label[j] = jtok
        return (token_ids, output_label) if return_output_label else token_ids

    def add_noise(self, i, lang_id, token_ids):
        prob = random.random()
        # BERT: 80% randomly change token to mask token
        if prob < self.mask_mask_prob:
            token_ids[i] = MASK_IDX

        # BERT: 10% randomly change token to random token
        elif prob < self.mask_random_prob:
            # vocab entries should have a higher token_id than MASK_IDX
            # replace token with char from same language vocabulary
            chosen = random.choice(self._lang_vocabs[lang_id])
            token_ids[i] = self.source_c2i[chosen]
        else:
            # -> BERT: rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            pass

