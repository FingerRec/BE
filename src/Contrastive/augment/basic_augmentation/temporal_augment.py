import random
import torch


def temporal_augment(l_new_data, augment_type, trace=False):
    """
    :param l_new_data:
    :param rotation_type
    :return:
    """
    B, C, T, H, W = l_new_data.size()
    if not trace:
        rotated_data = torch.zeros_like(l_new_data).cuda()
    else:
        rotated_data = l_new_data
    flip_type = augment_type // 4
    rot_type = augment_type % 4
    # flip at first
    for i in range(B):
        if flip_type[i] == 0:
            rotated_data[i] = l_new_data[i]
        elif flip_type[i] == 1:  # left-right flip
            rotated_data[i] = l_new_data[i].flip(3)
        elif flip_type[i] == 2:  # temporal flip
            rotated_data[i] = l_new_data[i].flip(1)
        else:  # left-right + temporal flip
            rotated_data[i] = l_new_data[i].flip(3).flip(1)
    # then rotation
    for i in range(B):
        if rot_type[i] == 0:
            rotated_data[i] = l_new_data[i]
        elif rot_type[i] == 1:  # 90 degree
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(2)
        elif rot_type[i] == 2:  # 180 degree
            rotated_data[i] = l_new_data[i].flip(2).flip(3)
        else:  # 270 degree
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(3)

    return rotated_data


def inverse_temporal_augment(l_new_data, augment_type, trace=False):
    """
    :param l_new_data:
    :param rotation_type
    :return:
    """
    B, C, T, H, W = l_new_data.size()
    if not trace:
        rotated_data = torch.zeros_like(l_new_data).cuda()
    else:
        rotated_data = l_new_data
    flip_type = augment_type // 4
    rot_type = augment_type % 4
    # flip at first
    for i in range(B):
        if flip_type[i] == 0:
            rotated_data[i] = l_new_data[i]
        elif flip_type[i] == 1:  # left-right flip
            rotated_data[i] = l_new_data[i].flip(3)
        elif flip_type[i] == 2:  # temporal flip
            rotated_data[i] = l_new_data[i].flip(1)
        else:  # left-right + temporal flip
            rotated_data[i] = l_new_data[i].flip(3).flip(1)
    # then rotation
    for i in range(B):
        if rot_type[i] == 0:
            rotated_data[i] = l_new_data[i]
        elif rot_type[i] == 1:  # -90 degree
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(3)
        elif rot_type[i] == 2:  # -180 degree
            rotated_data[i] = l_new_data[i].flip(3).flip(2)
        else:  # -270 degree
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(2)

    return rotated_data