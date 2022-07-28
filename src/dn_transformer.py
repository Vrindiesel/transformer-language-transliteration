"""
Created by davan 
7/26/22
"""

from transformer import Transformer


import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import PAD_IDX


class DNTransformer(Transformer):
    def loss(self, predict, target, reduction=True):
        """
        compute loss
        """
        predict = predict.view(-1, self.trg_vocab_size)

        # nll_loss = F.nll_loss(predict, target.view(-1), ignore_index=PAD_IDX)
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD_IDX)

        # accuracy
        _, pred_ids = torch.topk(predict, 1)
        accuracy = (pred_ids[non_pad_mask] == target[non_pad_mask]).float().mean()

        if reduction:
            nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
            smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
            smooth_loss = smooth_loss / self.trg_vocab_size
            loss = (1.0 - self.label_smooth) * nll_loss + self.label_smooth * smooth_loss
        else:
            loss = F.nll_loss(
                predict, target.view(-1), ignore_index=PAD_IDX, reduction="none"
            )
            loss = loss.view(target.shape)
            loss = loss.sum(dim=0) / (target != PAD_IDX).sum(dim=0)

        return loss, accuracy

    def get_loss(self, data, reduction=True, ret_preds=False):
        src, src_mask, trg, trg_mask, loss_mask = data
        out = self.forward(src, src_mask, trg, trg_mask)
        loss, accuracy = self.loss(out[:-1], loss_mask[1:], reduction=reduction)
        if ret_preds:
            return loss, accuracy, out
        else:
            return loss, accuracy




