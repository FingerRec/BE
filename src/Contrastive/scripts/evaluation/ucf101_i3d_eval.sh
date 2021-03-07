#!/usr/bin/env bash
python test.py \
--method ft \
--train_list ../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt \
--val_list ../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--arch i3d \
--mode rgb \
--batch_size 1 \
--stride 1 \
--data_length 64 \
--clip_size 64 \
--spatial_size 224 \
--workers 1 \
--dropout 0.5 \
--gpus 2 \
--weights ../experiments/fine_tune_rgb_model_latest.pth.tar #Scratch: ?
#--weights ../experiments/logs/ucf101_i3d_ft/ft_04-02-1128/fine_tune_rgb_model_latest.pth.tar #SSL: 78.83
#--weights ../experiments/logs/hmdb51_i3d_ft/ft_02-18-1134/fine_tune_rgb_model_latest.pth.tar #25.94
#--weights ../experiments/logs/hmdb51_i3d_ft/ft_03-23-2206/fine_tune_rgb_model_latest.pth.tar #49.86