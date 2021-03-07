#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from utils.utils import Timer
import torch
import torch.backends.cudnn as cudnn
from data.config import data_config, augmentation_config
from data.dataloader import data_loader_init
from model.config import model_config
from loss.config import optim_init
from augment.gen_positive import GenPositive
from bk.option_old import args
import numpy as np
import cv2
import skvideo.io


def save_one_video(video, idx=1, title='origin'):
    video = video.squeeze(0)
    video_tensor = torch.tensor(video.detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16 x 31 x 31 x 3
    path = "../experiments/gen_videos/{}_{}.jpg".format(title, idx)
    img = np.zeros((video_tensor.size(1), video_tensor.size(2), video_tensor.size(3)), np.uint8)
    img.fill(90)
    cv2.putText(img, title, (10, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    output = img
    for i in range(video_tensor.shape[0]):
        if i % 3 == 0:
            output = np.concatenate((output, np.uint8(video_tensor[i] * 255)), axis=1)
    cv2.imwrite(path, output)
    print("index: {} finished".format(idx))
    return output


def rgb_flow(prvs, next):
    hsv = np.zeros_like(prvs)
    hsv[..., 1] = 255
    prvs = cv2.cvtColor(prvs, cv2.COLOR_RGB2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow_rgb


def write_video(path, output):
    fps = 30
    writer = skvideo.io.FFmpegWriter(path,
                                     outputdict={'-b': '300000000', '-r': str(fps)})
    # print(len(output))
    for frame in output:
        frame = np.array(frame)
        writer.writeFrame(frame)
    writer.close()


def save_as_video(a, p, n, tn, idx=1, title='triples'):
    a = a.squeeze(0)
    p = p.squeeze(0)
    n = n.squeeze(0)
    tn = tn.squeeze(0)
    path = "../experiments/gen_videos/{}_{}.mp4".format(title, idx)
    a_p_path = "../experiments/gen_videos/{}_{}.mp4".format('a&p_', idx)
    a_n_path = "../experiments/gen_videos/{}_{}.mp4".format('a&n_', idx)
    flows_path = "../experiments/gen_videos/{}_{}.mp4".format('flows_', idx)
    a_tensor = torch.tensor(a.detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16 x 31 x 31 x 3
    p_tensor = torch.tensor(p.detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16 x 31 x 31 x 3
    n_tensor = torch.tensor(n.detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16 x 31 x 31 x 3
    tn_tensor = torch.tensor(tn.detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16 x 31 x 31 x 3
    output = []
    a_p_output = []
    a_n_output = []
    flows_output = []
    # print(a_tensor.size(0))
    for i in range(a_tensor.size(0) - 1):
        a_img = np.uint8(a_tensor[i] * 255)
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        p_img = np.uint8(p_tensor[i] * 255)
        n_img = np.uint8(n_tensor[i] * 255)
        tn_img = cv2.cvtColor(np.uint8(tn_tensor[i] * 255), cv2.COLOR_BGR2RGB)
        a_img_next = np.uint8(a_tensor[i+1] * 255)
        p_img_next = np.uint8(p_tensor[i+1] * 255)
        n_img_next = np.uint8(n_tensor[i + 1] * 255)
        tn_img_next = cv2.cvtColor(np.uint8(tn_tensor[i + 1] * 255), cv2.COLOR_BGR2RGB)
        flow_a = rgb_flow(a_img, a_img_next)
        flow_p = rgb_flow(p_img, p_img_next)
        flow_n = rgb_flow(n_img, n_img_next)
        flow_tn = rgb_flow(tn_img, tn_img_next)

        rgb_cat = np.concatenate((a_img, p_img, n_img, tn_img), 1)
        flow_cat = np.concatenate((flow_a, flow_p, flow_n, flow_tn), 1)
        img = np.concatenate((rgb_cat, flow_cat), 0)
        output.append(img)
        # a_p_output.append(np.concatenate((a_img, p_img), axis=1))
        # a_n_output.append(np.concatenate((a_img, n_img), axis=1))
        # flows_output.append(np.concatenate((flow_a, flow_p, flow_n), axis=1))
    # write_video(a_p_path, a_p_output)
    # write_video(a_n_path, a_n_output)
    # write_video(flows_path, flows_output)
    write_video(path, output)
    print("video: {} finished".format(idx))
    return


def validate(tc, val_loader, model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (inputs, target, index) in enumerate(val_loader):
            if i > 100:
                break
            # target = target.cuda(async=True)
            target = target.cuda()
            for j in range(len(inputs)):
                inputs[j] = inputs[j].float()
                inputs[j] = inputs[j].cuda()
            # ===================forward=====================
            anchor, positive, negative, t_wrap, s_wrap = inputs
            positive = tc(positive)
            save_as_video(anchor, positive, negative, t_wrap, idx=index.cpu().data)
            # anchor_cat = save_one_video(anchor, idx=index, title='anchor')
            # positive_cat = save_one_video(positive, idx=index, title='positive')
            # negative_cat = save_one_video(negative, idx=index, title='negative')
            # imgs = np.concatenate((anchor_cat, positive_cat, negative_cat))
            # path = "../experiments/gen_videos/{}_{}.jpg".format('triplet', index)
            # cv2.imwrite(path, imgs)
    return None


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.batch_size = 1
    args.data_length = 128
    args.stride = 1
    args.spatial_size = 224
    args.mode = 'rgb'
    args.eval_indict = 'loss'
    args.pt_loss = 'flow'
    args.workers = 1
    args.print_freq = 100
    args.dataset = 'ucf101'
    args.train_list = '../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt'
    args.val_list = '../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt'
    # args.root = ""
    args.root = "/data1/awinywang/Data/ft_local/ucf101/jpegs_256/" # 144
    pos_aug = GenPositive()
    torch.manual_seed(1)
    cudnn.benchmark = True
    timer = Timer()
    # == dataset config==
    num_class, data_length, image_tmpl = data_config(args)
    train_transforms, test_transforms, eval_transforms = augmentation_config(args)
    train_data_loader, val_data_loader, eval_data_loader, _, _, _ = data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms)
    # == model config==
    model = model_config(args, num_class)
    # == optim config==
    train_criterion, val_criterion, optimizer = optim_init(args, model)
    # == data augmentation(self-supervised) config==
    validate(pos_aug, val_data_loader, model)