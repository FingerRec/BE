import augment.video_transformations.videotransforms as videotransforms
import augment.video_transformations.video_transform_PIL_or_np as video_transform
from augment.video_transformations.volume_transforms import ClipToTensor

from torchvision import transforms


def pt_data_config(args):
    if args.pt_dataset == 'ucf101':
        num_class = 101
        image_tmpl = "frame{:06d}.jpg"
    elif args.pt_dataset == 'hmdb51':
        num_class = 51
        # image_tmpl = "frame{:06d}.jpg"
        image_tmpl = "img_{:05d}.jpg"
        # image_tmpl = "image_{:05d}.jpg"
    elif args.pt_dataset == 'kinetics':
        num_class = 400
        image_tmpl = "img_{:05d}.jpg"
        # args.root = "/data1/DataSet/Kinetics/compress/"
    elif args.pt_dataset == 'sth_v1':
        num_class = 174
        image_tmpl = "{:05d}.jpg"
    elif args.pt_dataset == 'diving48':
        num_class = 48
        image_tmpl = "image_{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    return num_class, int(args.pt_data_length), image_tmpl


def ft_data_config(args):
    if args.ft_dataset == 'ucf101':
        num_class = 101
        image_tmpl = "frame{:06d}.jpg"
    elif args.ft_dataset == 'hmdb51':
        num_class = 51
        # image_tmpl = "frame{:06d}.jpg"
        image_tmpl = "img_{:05d}.jpg"
        # image_tmpl = "image_{:05d}.jpg"
    elif args.ft_dataset == 'kinetics':
        num_class = 400
        image_tmpl = "img_{:05d}.jpg"
        # args.root = "/data1/DataSet/Kinetics/compress/"
    elif args.ft_dataset == 'sth_v1':
        num_class = 174
        image_tmpl = "{:05d}.jpg"
    elif args.ft_dataset == 'diving48':
        num_class = 48
        image_tmpl = "image_{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    return num_class, int(args.ft_data_length), image_tmpl


def pt_augmentation_config(args):
    if int(args.pt_spatial_size) == 112:
        # print("??????????????????????????")
        resize_size = 128
    else:
        resize_size = 256
    if args.pt_mode == 'rgb':
        normalize = video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = video_transform.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224])
    train_transforms = transforms.Compose([
        # videotransforms.RandomCrop(int(args.spatial_size)),
        video_transform.RandomRotation(10),
        # video_transform.ColorDistortion(1),
        # video_transform.STA_RandomRotation(10),
        # video_transform.Each_RandomRotation(10),
        video_transform.Resize(resize_size),
        video_transform.RandomCrop(int(args.pt_spatial_size)),
        video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),
        ClipToTensor(channel_nb=3 if args.pt_mode == 'rgb' else 2),
        normalize
        # videotransforms.ColorJitter(),
        # videotransforms.RandomHorizontalFlip()
    ])
    test_transforms = transforms.Compose([
                                        video_transform.Resize(resize_size),
                                        video_transform.CenterCrop(int(args.pt_spatial_size)),
                                         ClipToTensor(channel_nb=3 if args.pt_mode == 'rgb' else 2),
                                         normalize
    ]
                                         )
    eval_transfroms = transforms.Compose([
                                            video_transform.Resize(resize_size),
                                            video_transform.CenterCrop(int(args.pt_spatial_size)),
                                         ClipToTensor(channel_nb=3 if args.pt_mode == 'rgb' else 2),
                                         normalize
    ]
                                         )
    return train_transforms, test_transforms, eval_transfroms


def ft_augmentation_config(args):
    if int(args.ft_spatial_size) == 112:
        resize_size = 128
    else:
        resize_size = 256
    if args.ft_mode == 'rgb':
        normalize = video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = video_transform.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224])
    train_transforms = transforms.Compose([
        video_transform.RandomRotation(10),
        video_transform.Resize(resize_size),
        video_transform.RandomCrop(int(args.ft_spatial_size)),
        video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),
        ClipToTensor(channel_nb=3 if args.ft_mode == 'rgb' else 2),
        normalize
    ])
    test_transforms = transforms.Compose([
                                        video_transform.Resize(resize_size),
                                        video_transform.CenterCrop(int(args.ft_spatial_size)),
                                         ClipToTensor(channel_nb=3 if args.ft_mode == 'rgb' else 2),
                                         normalize
    ]
                                         )
    eval_transfroms = transforms.Compose([
                                            video_transform.Resize(resize_size),
                                            video_transform.CenterCrop(int(args.ft_spatial_size)),
                                         ClipToTensor(channel_nb=3 if args.ft_mode == 'rgb' else 2),
                                         normalize
    ]
                                         )
    return train_transforms, test_transforms, eval_transfroms
