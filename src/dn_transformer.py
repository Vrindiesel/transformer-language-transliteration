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

torch.autograd.set_detect_anomaly(True)

class DNTransformer(Transformer):
    def loss(self, predict, target, reduction=True):
        """
        compute loss
        """
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD_IDX)
        predict = predict.view(-1, self.trg_vocab_size)
        _, pred_ids = torch.topk(predict, 1)
        accuracy = (pred_ids[non_pad_mask] == target[non_pad_mask]).float().mean()
        if not reduction:
            loss = F.nll_loss(
                predict, target.view(-1), ignore_index=PAD_IDX, reduction="none"
            )
            loss = loss.view(target.shape)
            loss = loss.sum(dim=0) / (target != PAD_IDX).sum(dim=0)
        else:
            nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
            smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
            smooth_loss = smooth_loss / self.trg_vocab_size
            loss = (1.0 - self.label_smooth) * nll_loss + self.label_smooth * smooth_loss
        return loss, accuracy


    def get_loss(self, data, reduction=True, ret_preds=False):
        if len(data) == 5:
            src, src_mask, trg, trg_mask, loss_mask = data
            out = self.forward(src, src_mask, trg, trg_mask)
            target = loss_mask[1:]
            predict = out[:-1]

            """
            target = target.view(-1, 1)
            non_pad_mask = target.ne(PAD_IDX)
            predict = predict.view(-1, self.trg_vocab_size)
            #print("predict", predict.size())
            #print("predict:", predict.gather(dim=-1, index=target).size())
            #print("smooth:", predict.sum(dim=-1, keepdim=True).size())

            if not reduction:
                loss = F.nll_loss(
                    predict, target.view(-1), ignore_index=PAD_IDX, reduction="none"
                )
                loss = loss.view(target.shape)
                loss = loss.sum(dim=0) / (target != PAD_IDX).sum(dim=0)
            else:
                nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
                smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
                smooth_loss = smooth_loss / self.trg_vocab_size
                loss = (1.0 - self.label_smooth) * nll_loss + self.label_smooth * smooth_loss
            #print("loss")
            #print(loss)
            #out[:-1], trg[1:]
            #p,l = out[:-1, ..., :], loss_mask[1:, ...]
            #print("\n", p.size(), l.size())
            """

            loss, accuracy = self.loss(predict, target, reduction=reduction)
            retval = [loss, accuracy]
            if ret_preds:
                retval.append(out)
            retval = tuple(retval)
        else:
            retval = super().get_loss(data, reduction=reduction)
        #print(retval)
        return retval

class FinetunedDNTransformer(DNTransformer):
    pass


