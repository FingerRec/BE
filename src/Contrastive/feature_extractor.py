import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from data.config import data_config, augmentation_pretext_config
from data.dataloader import data_loader_init
from model.config import pretext_model_config
from augment.config import TC
from bk.option_old import args
import torch.nn as nn


def single_extract(tc, val_loader, model):
    model.eval()
    features = {'data':[], 'target':[]}
    with torch.no_grad():
        for i, (input, target, index) in enumerate(val_loader):
            inputs = input
            # inputs = tc(input)
            output = model(inputs)
            # print(output.size())
            output = nn.AdaptiveAvgPool3d(1)(output).view(output.size(0), output.size(1))
            # print(output.size())
            # print(target)
            for j in range(output.size(0)):
                features['data'].append(output[j].cpu().numpy())
                features['target'].append(target[j].cpu().numpy())
            if i % 10 == 0:
                print("{}/{} finished".format(i, len(val_loader)))
    return features


def feature_extract(tc, data_loader, model):
    features = single_extract(tc, data_loader, model)
    return features


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cudnn.benchmark = True
    # == dataset config==
    num_class, data_length, image_tmpl = data_config(args)
    train_transforms, test_transforms, _ = augmentation_pretext_config(args)
    train_data_loader, val_data_loader, _, _, _, _ = data_loader_init(args, data_length, image_tmpl, train_transforms,
                                                                      test_transforms, _)
    # == model config==
    model = pretext_model_config(args, num_class)
    tc = TC(args)
    # front = "contrastive_kinetics_warpping_{}".format(args.dataset)
    # front = "triplet_ucf101_warpping_{}".format(args.dataset)
    front = "{}_{}_{}".format(args.arch, args.front, args.dataset)
    dir = '../experiments/features/{}'.format(front)
    if not os.path.exists(dir):
        os.makedirs(dir)
    features = feature_extract(tc, val_data_loader, model)
    np.save('{}/val_features.npy'.format(dir), features)
    features = feature_extract(tc, train_data_loader, model)
    np.save('{}/train_features.npy'.format(dir), features)


if __name__ == '__main__':
    main()
