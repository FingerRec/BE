from torch import nn
import torch


class TC(nn.Module):
    def __init__(self, args):
        super(TC, self).__init__()
        self.args = args

    def forward(self, input):
        output = input.cuda()
        # output = self.mixup.mixup_data(output)
        output = torch.autograd.Variable(output)
        return output
