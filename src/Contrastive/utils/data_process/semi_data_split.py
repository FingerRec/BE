import os
import numpy as np


def split(file, new_file, radio=0.1, cls_num=51):
    """
    ucf101/hmdb51
    :param file:
    :param new_file:
    :param radio:
    :param cls_num:
    :return:
    """
    cls_nums = np.zeros(cls_num)
    f = open(file, 'r')
    for data in f.readlines():
        name, frame, cls = data.split(" ")
        cls = int(cls)
        cls_nums[cls] += 1
    new_lists = list()
    new_cls_nums = np.zeros(cls_num)
    f = open(file, 'r')
    for data in f.readlines():
        name, frame, cls = data.split(" ")
        cls = int(cls)
        if new_cls_nums[cls] < radio * cls_nums[cls]:
            new_lists.append(data)
            new_cls_nums[cls] += 1
    with open(new_file, 'w') as f:
        f.writelines(new_lists)
    return


def split_kinetics(file, new_file, radio=0.1, cls_num=400):
    cls_nums = np.zeros(cls_num)
    f = open(file, 'r')
    for data in f.readlines():
        name, cls = data.split(" ")
        cls = int(cls)
        cls_nums[cls] += 1
    new_lists = list()
    new_cls_nums = np.zeros(cls_num)
    f = open(file, 'r')
    for data in f.readlines():
        name, cls = data.split(" ")
        cls = int(cls)
        if new_cls_nums[cls] < radio * cls_nums[cls]:
            new_lists.append(data)
            new_cls_nums[cls] += 1
    with open(new_file, 'w') as f:
        f.writelines(new_lists)
    return


if __name__ == '__main__':
    file = '../datasets/lists/kinetics-400/ssd_kinetics_video_trainlist.txt'
    radio = 0.5
    new_file = '../datasets/lists/kinetics-400/{}_ssd_kinetics_video_trainlist.txt'.format(radio)
    split_kinetics(file, new_file, radio=radio, cls_num=400)