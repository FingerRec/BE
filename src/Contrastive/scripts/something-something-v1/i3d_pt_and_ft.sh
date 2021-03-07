#!/usr/bin/env bash
python main.py --gpus 3 --method pt_and_ft --pt_method be \
--pt_batch_size 48 --pt_workers 8  --arch i3d --pt_spatial_size 224 --pt_stride 4 --pt_data_length 16 \
--pt_nce_k 3569 --pt_softmax \
--pt_moco --pt_epochs 10 --pt_save_freq 4 --pt_print_freq 100 --pt_dataset sth_v1 \
--pt_train_list ../datasets/lists/something_something_v1/train_videofolder.txt \
--pt_val_list ../datasets/lists/something_something_v1/val_videofolder.txt \
--pt_root /data1/DataSet/something-something/20bn-something-something-v1/ \
--ft_train_list ../datasets/lists/something_something_v1/train_videofolder.txt \
--ft_val_list ../datasets/lists/something_something_v1/val_videofolder.txt \
--ft_root /data1/DataSet/something-something/20bn-something-something-v1/ \
--ft_dataset sth_v1 --ft_mode rgb \
--ft_lr 0.001 --ft_lr_steps 10 20 25 30 35 40 --ft_epochs 45 --ft_batch_size 48 \
--ft_data_length 16 --ft_spatial_size 224 --ft_workers 8 --ft_stride 1 --ft_dropout 0.5 \
--ft_print-freq 100 --ft_fixed 0