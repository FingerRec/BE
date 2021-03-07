#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from utils.utils import Timer
import torch
import torch.backends.cudnn as cudnn
from utils.utils import AverageMeter
from data.config import ft_data_config, ft_augmentation_config
from data.dataloader import ft_data_loader_init
from model.config import ft_model_config
from loss.config import ft_optim_init
from utils.learning_rate_adjust import ft_adjust_learning_rate
from augment.config import TC
from utils.utils import accuracy
import random
from datetime import datetime

lowest_val_loss = float('inf')
best_prec1 = 0
torch.manual_seed(1)


def fine_tune_train_and_val(args, recorder):
    # =
    global lowest_val_loss, best_prec1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning
    torch.manual_seed(1)
    cudnn.benchmark = True
    timer = Timer()
    # == dataset config==
    num_class, data_length, image_tmpl = ft_data_config(args)
    train_transforms, test_transforms, eval_transforms = ft_augmentation_config(args)
    train_data_loader, val_data_loader, _, _, _, _ = ft_data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms)
    # == model config==
    model = ft_model_config(args, num_class)
    recorder.record_message('a', '='*100)
    recorder.record_message('a', '-'*40+'finetune'+'-'*40)
    recorder.record_message('a', '='*100)
    # == optim config==
    train_criterion, val_criterion, optimizer = ft_optim_init(args, model)
    # == data augmentation(self-supervised) config==
    tc = TC(args)
    # == train and eval==
    print('*'*70+'Step2: fine tune'+'*'*50)
    for epoch in range(args.ft_start_epoch, args.ft_epochs):
        timer.tic()
        ft_adjust_learning_rate(optimizer, args.ft_lr, epoch, args.ft_lr_steps)
        train_prec1, train_loss = train(args, tc, train_data_loader, model, train_criterion, optimizer, epoch, recorder)
        # train_prec1, train_loss = random.random() * 100, random.random()
        recorder.record_ft_train(train_loss / 5.0, train_prec1 / 100.0)
        if (epoch + 1) % args.ft_eval_freq == 0:
            val_prec1, val_loss = validate(args, tc, val_data_loader, model, val_criterion, recorder)
            # val_prec1, val_loss = random.random() * 100, random.random()
            recorder.record_ft_val(val_loss / 5.0, val_prec1 / 100.0)
            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            checkpoint = {'epoch': epoch + 1, 'arch': "i3d", 'state_dict': model.state_dict(),
                          'best_prec1': best_prec1}
        recorder.save_ft_model(checkpoint,  is_best)
        timer.toc()
        left_time = timer.average_time * (args.ft_epochs - epoch)
        message = "Step2: fine tune best_prec1 is: {} left time is : {} now is : {}".format(best_prec1, timer.format(left_time), datetime.now())
        print(message)
        recorder.record_message('a', message)
    return recorder.filename


def train(args, tc, train_loader, model, criterion, optimizer, epoch, recorder, MoCo_init=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    if MoCo_init:
        model.eval()
    else:
        model.train()
    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda()
        index = index.cuda()
        inputs = tc(input)
        target = torch.autograd.Variable(target)
        output = model(inputs)
        loss = criterion(output, target) # + mse_loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        # # gradient check
        # plot_grad_flow(model.module.base_model.named_parameters())
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.ft_print_freq == 0:
            message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr']))
            print(message)
            recorder.record_message('a', message)
            message = "Finetune Training: Top1:{} Top3:{}".format(top1.avg, top3.avg)
            print(message)
            recorder.record_message('a', message)
    return top1.avg, losses.avg


def validate(args, tc, val_loader, model, criterion, recorder, MoCo_init=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target, index) in enumerate(val_loader):
            target = target.cuda()
            inputs = tc(input)
            target = torch.autograd.Variable(target)
            output = model(inputs)
            loss = criterion(output, target)
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.ft_print_freq == 0:
                message = ('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses
                    ))
                print(message)
                recorder.record_message('a', message)
                message = "Finetune Eval: Top1:{} Top3:{}".format(top1.avg, top3.avg)
                print(message)
                recorder.record_message('a', message)
    return top1.avg, losses.avg

