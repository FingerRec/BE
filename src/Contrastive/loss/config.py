import torch
import torch.optim as optim

from loss.NCE.NCEAverage import MemoryMoCo
from loss.NCE.NCECriterion import NCECriterion
from loss.NCE.NCECriterion import NCESoftmaxLoss


def get_fine_tuning_parameters(model, ft_begin_module='custom'):
    if not ft_begin_module:
        return model.parameters()
    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        parameters.append({'params': v})
        # if ft_begin_module in k:
        #     add_flag = True
        #     print(k)
        #     parameters.append({'params': v})
        # if add_flag:
        #     parameters.append({'params': v})
    return parameters


def pt_optim_init(args, model, n_data):
    contrast = MemoryMoCo(128, n_data, args.pt_nce_k, args.pt_nce_t, args.pt_softmax).cuda()
    criterion = NCESoftmaxLoss() if args.pt_softmax else NCECriterion(n_data)
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.pt_learning_rate,
                                momentum=args.pt_momentum,
                                weight_decay=args.pt_weight_decay)
    return contrast, criterion, optimizer


def ft_optim_init(args, model):
    train_criterion = torch.nn.NLLLoss().cuda()
    val_criterion = torch.nn.NLLLoss().cuda()

    if args.ft_fixed == 1:
        parameters = get_fine_tuning_parameters(model, ft_begin_module='custom')
    else:
        parameters = model.parameters()
    if args.ft_optim == 'sgd':
        optimizer = optim.SGD(parameters,
                              lr=args.ft_lr,
                              momentum=args.ft_momentum,
                              weight_decay=args.ft_weight_decay)
    elif args.ft_optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr)
    else:
        Exception("not supported optim")
    if args.ft_fixed == 1:
        count = 0
        for param_group in optimizer.param_groups:
            count += 1
        print("param group is: {}".format(count))
        count2 = 0
        for param_group in optimizer.param_groups:
            count2 += 1
            param_group['lr'] = param_group['lr'] * count2 / count
    return train_criterion, val_criterion, optimizer

