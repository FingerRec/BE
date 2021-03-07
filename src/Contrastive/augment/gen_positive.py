from torch import nn
from augment.basic_augmentation.mixup_methods import *
from augment.video_transformations.videotransforms import ColorJitter


class GenPositive(nn.Module):
    def __init__(self, prob=0.3):
        super(GenPositive, self).__init__()
        self.iv_mixup = SpatialMixup(0.3, trace=False, version=2)
        self.im_mixup = SpatialMixup(0.3, trace=False, version=3)
        self.cut = Cut(1, 0.05)
        self.prob = prob

    def intra_video_mixup(self, x):
        return self.iv_mixup.mixup_data(x)

    def inter_video_mixup(self, x):
        return self.im_mixup.mixup_data(x)

    def video_cut(self, x):
        return self.cut.cut_data(x)

    def forward(self, x):
        # x = self.inter_video_mixup(x)
        x = self.intra_video_mixup(x)
        return x
