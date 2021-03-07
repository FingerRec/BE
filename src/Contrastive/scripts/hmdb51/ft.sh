#!/usr/bin/env bash
python main.py --gpus 0 --method ft --arch i3d --pt_method be \
--ft_train_list ../datasets/lists/hmdb51/hmdb51_rgb_train_split_1.txt \
--ft_val_list ../datasets/lists/hmdb51/hmdb51_rgb_val_split_1.txt \
--ft_dataset hmdb51 --ft_mode rgb \
--ft_lr 0.001 --ft_lr_steps 10 20 25 30 35 40 --ft_epochs 45 --ft_batch_size 4 \
--ft_data_length 64 --ft_spatial_size 224 --ft_workers 4 --ft_stride 1 --ft_dropout 0.5 \
--ft_print-freq 100 --ft_fixed 0 \
--ft_weights ../experiments/ucf101_contrastive.pth