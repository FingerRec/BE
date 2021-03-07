#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-12 21:54
     # @Author  : Awiny
     # @Site    :
     # @Project : amax-pytorch-i3d
     # @File    : TemporalSubBlock.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch.nn as nn
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

class TemporalSubBlock(nn.Module):
    def __init__(self):
        super(TemporalSubBlock, self).__init__()
    def forward(self, x):
        b, c, t, h, w = x.size()
        y = torch.zeros((b, c, t-1, h, w)).cuda()
        for i in range(t-1):
            y[:,:,i,:,:] = x[:,:,i+1,:,:] - x[:,:,i,:,:]
        return y

class TemporalSubMeanBlock(nn.Module):
    def __init__(self):
        super(TemporalSubMeanBlock, self).__init__()

    def forward(self, x):
        b, c, t, h, w = x.size()
        mean = x.mean(2)
        for i in range(t):
            x[:,:,i,:,:] -= mean
        return x
