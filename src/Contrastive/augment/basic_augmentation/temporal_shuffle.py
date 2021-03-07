#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-21 10:37
     # @Author  : Awiny
     # @Site    :
     # @Project : amax-pytorch-i3d
     # @File    : TemporalShuffle.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""

import torch
import torch.nn as nn
import os
import difflib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


class TemporalShuffle(nn.Module):
    """
    for this module, random shuffle temporal dim, we want to find if the temporal information is important
    """
    def __init__(self, s=1):
        super(TemporalShuffle, self).__init__()
        self.s = s

    def forward(self, x):
        """
        random shuffle temporal dim
        :param x: b x c x t x h x w
        :return: out: b x c x t' x h x w
        """
        t = x.size(2)
        origin_idx = list(range(t))
        idxs = []
        K = 4
        similarity = 1
        # ==================================method1========================
        while similarity >= 1:
            if self.s == 1:
                idxs = torch.randperm(t)
            elif self.s == 2:
                idx = torch.randperm(K)
                for i in range(K):
                    for j in range(t // K):
                        idxs.append(idx[i].item() * t // K + j)
            else:
                for i in range(K):
                    idx = torch.randperm(t//K)
                    for j in range(len(idx)):
                        idxs.append(t//K*i + idx[j].item())
            similarity = difflib.SequenceMatcher(None, idxs, origin_idx).ratio()
            # print(idxs)
            # print(similarity)
        out = x[:, :, idxs, :, :]
        return out