from model.i3d import I3D
from model.r2p1d import R2Plus1DNet
from model.r3d import resnet18, resnet34, resnet50
from model.c3d import C3D
from model.s3d_g import S3D_G
from model.s3d import S3DG
import torch.nn as nn
from model.model import TCN
import torch
from utils.load_weights import ft_load_weight


def pt_model_config(args, num_class):
    if args.arch == 'i3d':
        model = I3D(num_classes=101, modality=args.pt_mode, with_classifier=False)
        model_ema = I3D(num_classes=101, modality=args.pt_mode,  with_classifier=False)
    elif args.arch == 'r2p1d':
        model = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)
        model_ema = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)
    elif args.arch == 'r3d18':
        model = resnet18(num_classes=num_class, with_classifier=False)
        model_ema = resnet18(num_classes=num_class, with_classifier=False)
    elif args.arch == 'r3d34':
        model = resnet34(num_classes=num_class, with_classifier=False)
        model_ema = resnet34(num_classes=num_class, with_classifier=False)
    elif args.arch == 'r3d50':
        model = resnet50(num_classes=num_class, with_classifier=False)
        model_ema = resnet50(num_classes=num_class, with_classifier=False)
    elif args.arch == 'c3d':
        model = C3D(with_classifier=False, num_classes=num_class)
        model_ema = C3D(with_classifier=False, num_classes=num_class)
    elif args.arch == 's3d':
        model = S3D_G(num_class=num_class, in_channel=3, gate=True, with_classifier=False)
        model_ema = S3D_G(num_class=num_class, in_channel=3, gate=True, with_classifier=False)
    else:
        Exception("Not implemene error!")
    model = torch.nn.DataParallel(model)
    model_ema = torch.nn.DataParallel(model_ema)
    return model, model_ema


def ft_model_config(args, num_class):
    with_classifier = True
    if args.arch == 'i3d':
        base_model = I3D(num_classes=num_class, modality=args.ft_mode, dropout_prob=args.ft_dropout, with_classifier=with_classifier)
        # args.logits_channel = 1024
        if args.ft_spatial_size == '112':
            out_size = (int(args.ft_data_length) // 8, 4, 4)
        else:
            out_size = (int(args.ft_data_length) // 8, 7, 7)
    elif args.arch == 'r2p1d':
        base_model = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'c3d':
        base_model = C3D(num_classes=num_class, with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d18':
        base_model = resnet18(num_classes=num_class, sample_size=int(args.ft_spatial_size), with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d34':
        base_model = resnet34(num_classes=num_class, sample_size=int(args.ft_spatial_size), with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d50':
        base_model = resnet50(num_classes=num_class, sample_size=int(args.ft_spatial_size), with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 's3d':
        # base_model = S3D_G(num_class=num_class, drop_prob=args.dropout, in_channel=3)
        base_model = S3DG(num_classes=num_class, dropout_keep_prob=args.ft_dropout, input_channel=3, spatial_squeeze=True, with_classifier=True)
        # args.logits_channel = 1024
        out_size = (2, 7, 7)
    else:
        Exception("unsuporrted arch!")
    base_model = ft_load_weight(args, base_model)
    model = TCN(base_model,  out_size, args)
    model = nn.DataParallel(model).cuda()
    # cudnn.benchmark = True
    return model
