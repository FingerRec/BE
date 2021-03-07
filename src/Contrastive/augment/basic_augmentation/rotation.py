import random
import torch


def batch_rotation(l_new_data):
    B, C, T, H, W = l_new_data.size()
    rotation_type = random.randint(0, 2)
    # print(rotation_type)
    if rotation_type == 0:
        index = list(range(T - 1, -1, -1))
        rotation_data = l_new_data[:, :, index, :, :]
    elif rotation_type == 1:
        index = list(range(H - 1, -1, -1))
        rotation_data = l_new_data[:, :, :, index, :]
    else:
        index = list(range(W - 1, -1, -1))
        rotation_data = l_new_data[:, :, :, :, index]
    return rotation_data, rotation_type


def sample_rotation(l_new_data, rotation_type, trace=False):
    """
    her/vec flip (0, 1)
    rotation 90/180/270 degree(2, 3, 4)
    her flip + rotate90 / ver flip + rotate 90 (5, 6)
    :param l_new_data:
    :param rotation_type
    :return:
    """
    B, C, T, H, W = l_new_data.size()
    if not trace:
        rotated_data = torch.zeros_like(l_new_data).cuda()
    else:
        rotated_data = l_new_data
    for i in range(B):
        if rotation_type[i] == 0:
            rotated_data[i] = l_new_data[i].flip(2)
        elif rotation_type[i] == 1:
            rotated_data[i] = l_new_data[i].flip(3)
        elif rotation_type[i] == 2:
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(2)
        elif rotation_type[i] == 3:
            rotated_data[i] = l_new_data[i].flip(2).flip(3)
        elif rotation_type[i] == 4:
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(3)
        elif rotation_type[i] == 5:
            rotated_data[i] = l_new_data[i].flip(2).transpose(2, 3).flip(2)
        elif rotation_type[i] == 6:
            rotated_data[i] = l_new_data[i].flip(3).transpose(2, 3).flip(2)
        else:
            rotated_data[i] = l_new_data[i]
    return rotated_data


def sample_rotation_cls(l_new_data, rotation_type):
    """
    her/vec flip (0, 1)
    rotation 90/180/270 degree(2, 3, 4)
    her flip + rotate90 / ver flip + rotate 90 (5, 6)
    :param l_new_data:
    :param rotation_type:
    :return:
    """
    B, C, T, H, W = l_new_data.size()
    rotated_data = torch.zeros_like(l_new_data).cuda()
    for i in range(B):
        if rotation_type[i] == 0:
            rotated_data[i] = l_new_data[i].flip(2)
        elif rotation_type[i] == 1:
            rotated_data[i] = l_new_data[i].flip(3)
        elif rotation_type[i] == 2:
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(2)
        elif rotation_type[i] == 3:
            rotated_data[i] = l_new_data[i].flip(2).flip(3)
        elif rotation_type[i] == 4:
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(3)
        elif rotation_type[i] == 5:
            rotated_data[i] = l_new_data[i].flip(2).transpose(2, 3).flip(2)
        elif rotation_type[i] == 6:
            rotated_data[i] = l_new_data[i].flip(3).transpose(2, 3).flip(2)
        else:
            rotated_data[i] = l_new_data[i]
    return rotated_data


def four_rotation_cls(l_new_data, rotation_type):
    """
    rotation 0/90/180/270 degree(0,1,2,3)
    :return:
    """
    B, C, T, H, W = l_new_data.size()
    rotated_data = torch.zeros_like(l_new_data).cuda()
    for i in range(B):
        if rotation_type[i] == 1:
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(2)
        elif rotation_type[i] == 2:
            rotated_data[i] = l_new_data[i].flip(2).flip(3)
        elif rotation_type[i] == 3:
            rotated_data[i] = l_new_data[i].transpose(2, 3).flip(3)
        else:
            rotated_data[i] = l_new_data[i]
    return rotated_data


def all_flips(l_new_data, flip_type, trace=False):
    """
    her/vec flip (0, 1)
    rotation 90/180/270 degree(2, 3, 4)
    her flip + rotate90 / ver flip + rotate 90 (5, 6)
    :param l_new_data:
    :param rotation_type
    :return:
    """
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
    flip_type = flip_type // 4
    rot_type = flip_type % 4
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


if __name__ == '__main__':
    a = torch.tensor([[1,2],[3,4]]).view(1, 1, 1, 2, 2)
    print(a.size())
    for i in range(8):
        print(torch.tensor(i))
        print(sample_rotation_cls(a, torch.tensor([i])))