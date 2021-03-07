import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import datetime
import shutil


class Record:
    def __init__(self, args):
        super(Record, self).__init__()
        if args.method == 'pt_and_ft':
            self.origin_path = '../experiments/' + args.pt_dataset + '_to_' + args.ft_dataset
        elif args.method == 'pt':
            self.origin_path = '../experiments/pt_' + args.pt_dataset
        else:
            self.origin_path = '../experiments/ft_' + args.ft_dataset
        if not os.path.exists(self.origin_path):
            os.mkdir(self.origin_path)
        self.path = self.origin_path + '/' + args.date
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if args.method in ['pt', 'pt_and_ft']:
            self.pt_path = self.path + '/pt'
            self.pt_model_path = '{}/models'.format(self.pt_path)
            if not os.path.exists(self.pt_path):
                os.mkdir(self.pt_path)
            if not os.path.exists(self.pt_model_path):
                os.mkdir(self.pt_model_path)
        if args.method in ['ft', 'pt_and_ft']:
            self.ft_path = self.path + '/ft'
            self.ft_model_path = '{}/models'.format(self.ft_path)
            if not os.path.exists(self.ft_path):
                os.mkdir(self.ft_path)
            if not os.path.exists(self.ft_model_path):
                os.mkdir(self.ft_model_path)
        self.args = args
        # pretrain init
        self.pt_init()
        self.pt_train_loss_list = list()
        self.pt_checkpoint = ''
        # finetune init
        self.ft_init()
        self.ft_train_acc_list = list()
        self.ft_val_acc_list = list()
        self.ft_train_loss_list = list()
        self.ft_val_loss_list = list()
        self.front = self.args.method
        print(self.args)
        self.record_txt = os.path.join(self.path, self.front + '_logs.txt')
        self.record_init(args, 'w')
        self.src_init()
        self.filename = ''
        self.best_name = ''

    def pt_init(self):
        return

    def ft_init(self):
        return

    def src_init(self):
        if not os.path.exists(self.path + '/src_record'):
            shutil.copytree('../src', self.path + '/src_record')

    def record_init(self, args, open_type):
        with open(self.record_txt, open_type) as f:
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def record_message(self, open_type, message):
        with open(self.record_txt, open_type) as f:
            f.write(message + '\n\n')

    def record_ft_train(self, loss, acc=0):
        self.ft_train_acc_list.append(acc)
        self.ft_train_loss_list.append(loss)

    def record_ft_val(self, loss, acc=0):
        self.ft_val_acc_list.append(acc)
        self.ft_val_loss_list.append(loss)

    def record_pt_train(self, loss):
        self.pt_train_loss_list.append(loss)

    def plot_figure(self, plot_list, name='_performance'):
        epoch = len(plot_list[0][0])
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(self.args.arch + '_' + self.args.status + name)
        for i in range(len(plot_list)):
            plt.plot(axis, plot_list[i][0], label=plot_list[i][1])
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('%')
        plt.grid(True)
        plt.savefig(os.path.join(self.path, '{}.pdf'.format(self.args.status + name)))
        plt.close(fig)

    def save_ft_model(self, model, is_best=False):
        self.save_ft_checkpoint(self.args, model, is_best)
        plot_list = list()
        plot_list.append([self.ft_train_acc_list, 'train_acc'])
        plot_list.append([self.ft_val_acc_list, 'val_acc'])
        plot_list.append([self.ft_train_loss_list, 'train_loss'])
        plot_list.append([self.ft_val_loss_list, 'val_loss'])
        self.plot_figure(plot_list)

    def save_pt_model(self, args, state, epoch):
        self.save_pt_checkpoint(args, state, epoch)
        plot_list = list()
        plot_list.append([self.pt_train_loss_list, 'train_loss'])
        self.plot_figure(plot_list)
        print('==> Saving...')

    def save_pt_checkpoint(self, args, state, epoch):
        save_file = os.path.join(self.pt_model_path, 'current.pth')
        self.pt_checkpoint = save_file
        torch.save(state, save_file)
        if epoch % args.pt_save_freq == 0:
            save_file = os.path.join(self.pt_model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()

    def save_ft_checkpoint(self, args, state, is_best):
        self.filename = self.ft_path + '/' + args.ft_mode + '_model_latest.pth.tar'
        torch.save(state, self.filename)
        if is_best:
            self.best_name = self.ft_path + '/' + args.ft_mode + '_model_best.pth.tar'
            shutil.copyfile(self.filename, self.best_name)
