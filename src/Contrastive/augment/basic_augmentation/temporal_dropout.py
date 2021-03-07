#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-13 19:34
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : TemporalDropoutBlock.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)


class TemporalDropoutBlock(nn.Module):
    """
    method1, for 3d feature map BxCxTxHxW reshape as Bx[CxT]xHxW
    """
    def __init__(self, dropout_radio):
        super(TemporalDropoutBlock, self).__init__()
        self.dropout = nn.Dropout(dropout_radio)

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.view(b, c*t, h, w)
        x = self.dropout(x)
        x = x.view(b, c, t, h, w)
        return x


class TemporalDropoutBlock3D(nn.Module):
    r"""
    method2, for 3d feature map BxCxTxHxW, random dropout in T
    """

    def __init__(self, drop_prob):
        super(TemporalDropoutBlock3D, self).__init__()
        self.dropout = nn.Dropout3d(drop_prob)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class TemporalBranchDropout(nn.Module):
    """
    Branch dropout
    """
    def __init__(self, drop_prob):
        super(TemporalBranchDropout, self).__init__()
        self.dropout = nn.Dropout(1)
        self.drop_prob = drop_prob

    def forward(self, x):
        prob = random.random()
        if prob < self.drop_prob:
            x = self.dropout(x)
            #print(x)
        else:
            x = x
        return x

