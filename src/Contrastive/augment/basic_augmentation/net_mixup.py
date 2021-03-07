import numpy as np


class NETMIXUP(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def gen_prob(self):
        lam = np.random.beta(self.alpha, self.alpha)
        return lam

    def construct(self, a, b, mixup_radio=0.5):
        c = mixup_radio * a + (1 - mixup_radio) * b
        return c
