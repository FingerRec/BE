import numpy as np


def pt_adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    # if epoch < 2:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-7
    #     return 0
    # print(epoch)
    # print(np.asarray(opt.pt_lr_decay_epochs))
    steps = np.sum(epoch > np.asarray(opt.pt_lr_decay_epochs))
    if steps > 0:
        new_lr = opt.pt_learning_rate * (opt.pt_lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def ft_adjust_learning_rate(optimizer, intial_lr, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.3 ** (sum(epoch >= np.array(lr_steps)))
    lr = intial_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

