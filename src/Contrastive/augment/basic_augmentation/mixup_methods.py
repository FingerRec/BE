import csv
import numpy as np
import torch
import random
import time
# import cv2
import math
from numpy import random


def add_gaussian_noise(x):
    std = 1
    mean = 0
    return x + torch.randn(x.size()).cuda() * std + mean


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def simply_aug(x):
    """

    :param x: 3D input
    :return:
    """
    prob = random.random()
    if prob < 0.25:
        x = x.transpose(1, 2)
    elif prob < 0.5:
        x = flip(x, 1)
    elif prob < 0.75:
        x = flip(x, 2)
    else:
        x = x
    return add_gaussian_noise(x)


class MixUp(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def mixup_data(self, x, y, use_cuda=True):
        """
        return mixed inputs. pairs of targets
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        # print(lam)
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TemporalMixup(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def mixup_data(self, x):
        """
        return mixed inputs. pairs of targets
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        b, c, t, h, w = x.size()
        from numpy import random
        # skip = random.randint()
        skip = 4
        mixed_x = x
        for i in range(b):
            for j in range(t):
                mixed_x[i, :, j, :, :] = lam * x[i, :, j, :, :] + (1 - lam) * x[i, :, (j+skip) % t, :, :]
        return mixed_x

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SpatialMixup(object):
    def __init__(self, alpha, trace=True, version=2):
        self.alpha = alpha
        self.trace = trace
        self.version = version

    def mixup_data(self, x):
        """
        return mixed inputs. pairs of targets
        """
        import random
        if self.version == 1:
        # # ================version 1: random select sample and fusion with stable frame (all video)===================
            b, c, t, h, w = x.size()
            loss_prob = random.random() * self.alpha
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x)
            for i in range(b):
                tmp = (i+1) % b
                img_index = random.randint(t)
                for j in range(t):
                    mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[tmp, :, img_index, :, :]
                    # cv2.imshow("", mixed_x[i,:,j,:,:])
            return mixed_x
        # ================version 2: random select one same video sample and fusion with stable frame=================
        elif self.version == 2:
            b, c, t, h, w = x.size()
            from numpy import random
            loss_prob = random.random() * self.alpha
            # mixed_x = torch.zeros_like(x).cuda()
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x).cuda()
            for i in range(b):
                img_index = random.randint(t)
                # static = simply_aug(x[i, :, img_index, :, :])
                static = x[i, :, img_index, :, :]
                for j in range(t):
                    mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * static
                    # mixed_x[i, :, j, :, :] = (1+loss_prob)*x[i, :, j, :, :] - loss_prob * static
            return mixed_x
        # #================================ version 3: x and y all change================================
        else:
            b, c, t, h, w = x.size()
            from numpy import random
            loss_prob = random.random() * 0.3
            gama = 3  # control the importance of spatial information
            mixed_x = x
            index = torch.randperm(b)
            for i in range(b):
                img_index = random.randint(t)
                for j in range(t):
                    mixed_x[i, :, j, :, :] = (1-loss_prob) * x[i, :, j, :, :] + loss_prob * x[index[i], :, img_index, :, :]
            # return mixed_x, y, y[index], loss_prob/gama
            return mixed_x

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return (1-lam) * criterion(pred, y_a) + lam * criterion(pred, y_b)




class Cut(object):
    def __init__(self, beta, cut_prob):
        self.beta = beta
        self.cut_prob = cut_prob

    def rand_bbox(self, size, lam):
        T = size[2]
        W = size[3]
        H = size[4]
        cut_rat = np.sqrt(lam)
        # cut_t = np.int(T * cut_rat)
        cut_t = np.int(T)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbt1, bbt2, bbx1, bby1, bbx2, bby2

    def cut_data(self, input):
        lam = np.random.beta(self.beta, self.beta) * self.cut_prob
        bbt1, bbt2, bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        input[:, :, :, bbx1:bbx2, bby1:bby2] = 0
        return input


class PartMix(object):
    """
    crop part of a and part of b, contact them
    """
    def __init__(self, beta):
        self.beta = beta

    def rand_bbox(self, size, lam):
        T = size[2]
        W = size[3]
        H = size[4]

        bbt1 = 0
        bbt2 = T
        bbx1 = 0
        bbx2 = W
        bby1 = 0
        bby2 = math.ceil(lam*H)

        return bbt1, bbt2, bbx1, bby1, bbx2, bby2

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbt1, bbt2, bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1) / (input.size()[-1] * input.size()[-2] * input.size()[-3]))
        return input, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class CombineMix(object):
    """
    crop part of a and part of b, contact them
    """
    def __init__(self, beta):
        self.beta = beta

    def rand_bbox(self, size, lam):
        T = size[2]
        bbt1 = 0
        bbt2 = math.ceil(lam*T)

        return bbt1, bbt2

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbt1, bbt2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbt1:bbt2, :, :] = input[rand_index, :, bbt1:bbt2, :, :]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - (bbt2 - bbt1)/input.size()[-3]
        return input, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class CutMix(object):
    def __init__(self, beta, cutmix_prob):
        self.beta = beta
        self.cutmix_prob = cutmix_prob

    def rand_bbox(self, size, lam):
        T = size[2]
        W = size[3]
        H = size[4]
        cut_rat = np.sqrt(1. - lam)
        cut_t = np.int(T * cut_rat)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbt1, bbt2, bbx1, bby1, bbx2, bby2

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbt1, bbt2, bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1) / (input.size()[-1] * input.size()[-2] * input.size()[-3]))
        return input, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


class GridMix(object):
    def __init__(self, beta):
        self.beta = beta

    def rand_bbox(self, size, lam):
        W = size[3]
        H = size[4]
        # cut_rat = np.sqrt(1. - lam)
        cut_rat = lam
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        bbx1 = cut_w
        bby1 = cut_h
        bbx2 = W
        bby2 = H
        return bbx1, bby1, bbx2, bby2

    def imgs_resize(self, input, f_size):
        """
        resize input (spatial) into fixed size
        :param input:
        :param size:
        :return:
        """
        b, c, t, h, w = input.size()
        resize_imgs = torch.nn.functional.interpolate(input, size=(t, f_size[0], f_size[1]), scale_factor=None,
                                                      mode='trilinear', align_corners=True)
        return resize_imgs

    def mixup_data(self, input, target):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        output = torch.zeros_like(input)
        if bbx1 > 1 and bby1 > 1:
            output[:, :, :, :bbx1, :bby1] = self.imgs_resize(input[:, :, :, :, :], (bbx1, bby1))
            output[:, :, :, bbx1:bbx2, bby1:bby2] = self.imgs_resize(input[rand_index, :, :, bbx1:bbx2, bby1:bby2],
                                                                    (bbx2 - bbx1, bby2 - bby1))
        else:
            output = input
        return output, target_a, target_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)
