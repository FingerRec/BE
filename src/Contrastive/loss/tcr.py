import torch.nn as nn
import torch


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def tcr(feats_o, feats_r):
    loss = nn.MSELoss()
    feats_r = flip(feats_r, 2)
    b, c, t, h, w = feats_o.size()
    o_t = nn.AdaptiveAvgPool3d((t, 1, 1))(feats_o)
    o_r = nn.AdaptiveAvgPool3d((t, 1, 1))(feats_r)
    output = loss(o_t, o_r)
    return output