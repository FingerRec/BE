import os
import torch
from torch.nn.init import xavier_uniform_, constant_, zeros_, normal_, kaiming_uniform_, kaiming_normal_
import torch.nn as nn
import math


def weight_transform(model_dict, pretrain_dict):
    '''

    :return:
    '''
    count = 0
    # for k, v in pretrain_dict.items():
    #     count += 1
    #     print(k)
    #     if count > 100:
    #         break
    # for k, v in model_dict.items():
    #     print(k)
    #     count += 1
    #     if count > 20:
    #         break

    weight_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict and 'custom' not in k} # and 'custom' not in k
    for k, v in weight_dict.items():
        print("load: {}".format(k))
    # print(weight_dict)
    model_dict.update(weight_dict)
    return model_dict


def weights_init(model):
    """ Initializes the weights of the CNN model using the Xavier
    initialization.
    """
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
    #         xavier_uniform_(m.weight, gain=math.sqrt(2.0))
    #         if m.bias:
    #             constant_(m.bias, 0.1)
    #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
    #         normal_(m.weight, 1.0, 0.02)
    #         zeros_(m.bias)
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Conv3d) or isinstance(model, nn.Conv1d):
        xavier_uniform_(model.weight.data, gain=math.sqrt(2.0))
        # if model.bias:
        #     constant_(model.bias.data, 0.1)
    elif isinstance(model, nn.BatchNorm2d) or isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm3d):
        normal_(model.weight.data, 1.0, 0.02)


def pt_load_weight(args, model, model_ema, optimizer, contrast):
    # random initialization
    model.apply(weights_init)
    model_ema.apply(weights_init)
    if args.pt_resume:
        if os.path.isfile(args.pt_resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            model_ema.load_state_dict(checkpoint['model_ema'])
            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model, model_ema


def ft_load_weight(args, model):
    if args.ft_weights == "":
        print("no pretrained model available. Train from Scratch.....................")
        model.apply(weights_init)
        # weights_init(model)
    else:
        # weights_init(model)
        model.apply(weights_init)
        checkpoint = torch.load(args.ft_weights)
        try:
            print("model epoch {} lowese val: {}".format(checkpoint['epoch'], checkpoint['lowest_val']))
        except KeyError as e:
            try:
                print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
            except KeyError:
                print("not train from this code!")
        try:
            # print("????")
            print(args.ft_weights.split('/')[-1][:9])
            if args.ft_weights.split('/')[-1][:11] == 'mutual_loss':
                pretrain_dict = {('.'.join(k.split('.')[2:]))[2:]: v for k, v in list(checkpoint['state_dict'].items())}
            elif args.ft_weights.split('/')[-1][:9] == 'byol_ckpt':
                # print("???")
                pretrain_dict = {k[26:]: v for k, v in list(checkpoint['model'].items())}
                # pretrain_dict = {k[7:]: v for k, v in list(checkpoint['model_ema'].items())}
            elif args.ft_weights.split('/')[-1][:4] in ['ckpt', 'curr', 'ucf1']:
                # print("???")
                pretrain_dict = {k[7:]: v for k, v in list(checkpoint['model'].items())}
                # pretrain_dict = {k: v for k, v in list(checkpoint['model'].items())}
            elif args.ft_weights.split('/')[-1][:3] == 'i3d':
                pretrain_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            else:
                pretrain_dict = {'.'.join(k.split('.')[2:]): v for k, v in list(checkpoint['state_dict'].items())}
        except KeyError:
            # pretrain_dict = checkpoint['model']
            pretrain_dict = checkpoint
        # pretrain_dict = {k[26:]: v for k, v in list(checkpoint['state_dict'].items())}
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict)
        model.load_state_dict(model_dict)
    if args.ft_resume:
        if os.path.isfile(args.ft_resume):
            print(("=> loading checkpoints '{}'".format(args.ft_resume)))
            checkpoint = torch.load(args.ft_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoints '{}' (epoch {}) best_prec1 {}"
                   .format(args.evaluate, checkpoint['epoch'], best_prec1)))
        else:
            print(("=> no checkpoints found at '{}'".format(args.ft_resume)))
    return model