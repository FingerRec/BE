import torch
import difflib
import random
from augment.basic_augmentation.mixup_methods import SpatialMixup


def swap_one_time(seq):
    """
    swap a seq random place one time
    :param seq:
    :return:
    """
    length = len(seq)
    index = random.randint(0, length-1)
    index2 = random.randint(0, length-1)
    new_seq = seq.clone()
    new_seq[index] = seq[index2]
    new_seq[index2] = seq[index]
    return new_seq


def gen_sim_seq(K, radio=0.95, segments=4):
    """
    generate shuffled video sequences as negative, while random shuffle is always zero, control the segments
    (eg. divide into 4 segments and shuffle these segments)
    :param K:
    :param radio:
    :return:
    """
    similarity = 1
    idx = torch.arange(K)
    assert K % segments == 0
    seg_len = K // segments
    origin_idx = torch.arange(K).long()
    # revise_idx = torch.tensor(list(range(K - 1, -1, -1)))
    # print(revise_idx)
    while similarity > radio:
        # seg_idx = torch.randperm(segments)
        # for i in range(segments):
        #     idx[i*seg_len:(i+1)*seg_len] = origin_idx[seg_idx[i]*seg_len:(seg_idx[i]+1)*seg_len]
        idx = swap_one_time(idx)
        similarity = difflib.SequenceMatcher(None, idx, origin_idx).ratio()
    # print(idx)
    # print(origin_idx)
    # print(similarity)
    return idx


def batch_lst(lst, k):
    return lst[k:] + lst[:k]


class TRIPLET(object):
    def __init__(self, t_radio=0.95, s_radio=0.7):
        self.t_radio = t_radio
        self.s_radio = s_radio
        self.spaital_mixup = SpatialMixup(0.8)

    def construct(self, input):
        b, c, t, h, w = input.size()
        #spatial_noise = generate_noise((c, h, w), b)
        # print(sum(sum(sum(sum(spatial_noise)))))
        # print(sum(sum(sum(sum(sum(input))))))
        # postive = torch.zeros_like(input)
        # drop_radio = random.random() * self.s_radio
        postive = self.spaital_mixup.mixup_data(input, trace=False)
        # for i in range(t):
        #     postive[:, :, i, :, :] = (1 - drop_radio) * input[:, :, i, :, :] + drop_radio * spatial_noise[:, :, :, :]
        # if the match low than radio, as it negative
        # negative_seq = gen_sim_seq(t, self.radio)
        # negative = torch.zeros_like(input)
        # negative = negative[:, :, negative_seq, :, :]
        index = random.randint(1, b-1)
        indexs = batch_lst(list(range(b)), index)
        negative = input[indexs]
        # print(negative - input)
        return input, postive, negative