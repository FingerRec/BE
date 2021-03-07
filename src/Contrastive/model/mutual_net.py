import torch.nn as nn


class MutualNet(nn.Module):
    def __init__(self, embeddingnet):
        super(MutualNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        feature_x = self.embeddingnet(x)
        feature_y = self.embeddingnet(y)
        return feature_x, feature_y