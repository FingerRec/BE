#!/usr/bin/env bash
python feature_extractor.py \
--eval_indict feature_extract \
--method ft \
--train_list ../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt \
--val_list ../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--arch c3d \
--mode rgb \
--lr 0.001 \
--lr_steps 10 20 25 30 35 40 \
--epochs 45 \
--batch_size 1 \
--data_length 64 \
--spatial_size 224 \
--workers 8 \
--dropout 0.5 \
--gpus 2 \
--logs_path ../experiments/logs/hmdb51_c3d_ft \
--print-freq 100  \
--front c3d_contrastive_ucf101_warpping_ucf101 \
--weights ../experiments/MoCo/ucf101/models/08-19-1644_aug_CJ/ckpt_epoch_20.pth #ucf101_triplet
# --weights ../experiments/MoCo/ucf101/models/08-18-1956_aug_CJ/ckpt_epoch_40.pth # ucf101_contrastive_wrapping
# --weights ../experiments/triplet/ucf101/models/08-04-2112_aug_CJ/ckpt_epoch_10.pth # ucf101_triplet
# --weights ../experiments/MoCo/ucf101/models/08-14-1615_aug_CJ/ckpt_epoch_150.pth # ucf101_contrastive
# --weights ../experiments/MoCo/ucf101/models/08-12-1150_aug_CJ/ckpt_epoch_42.pth # kinetics