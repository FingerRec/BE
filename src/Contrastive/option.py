import argparse
parser = argparse.ArgumentParser()

# ============================================== global setting =====================================
parser.add_argument('--method', type=str, default='pt_and_ft', choices=['pt', 'ft', 'pt_and_ft'], help='pretrain or fine_tune or together')
parser.add_argument('--status', type=str, default='pt', choices=['pt', 'ft'], help='pretrain or fine_tune')
parser.add_argument('--arch', default='i3d', type=str, choices=['i3d', 'r3d18', 'r3d34', 'r3d50', 'r2p1d', 'c3d', 's3d'])
parser.add_argument('--gpus', type=str, default="0", help="define gpu id")

# ============================================== pretrain =====================================
parser.add_argument('--pt_mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--pt_method', type=str, default='be', choices=['moco', 'be', 'be_triplet'])
parser.add_argument('--pt_print_freq', type=int, default=10, help='print frequency')
parser.add_argument('--pt_tb_freq', type=int, default=500, help='tb frequency')
parser.add_argument('--pt_save_freq', type=int, default=5, help='save frequency')
parser.add_argument('--pt_batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--pt_workers', type=int, default=16, help='num of workers to use')
parser.add_argument('--pt_epochs', type=int, default=240, help='number of training epochs')
parser.add_argument('--pt_stride', default=1, type=int, help='stride of temporal image')
parser.add_argument('--pt_weights', default="", type=str, help='checkpoints')
parser.add_argument('--pt_spatial_size', default='224', choices=['112', '224'], help='the network input size')
parser.add_argument('--pt_data_length', default='64', help='input clip length')
parser.add_argument('--pt_clips', default='4', help='global local clips num')
parser.add_argument('--pt_flow_prefix', default="", type=str)

# optimization
parser.add_argument('--pt_learning_rate', type=float, default=0.003, help='learning rate')
parser.add_argument('--pt_lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
parser.add_argument('--pt_lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--pt_beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--pt_beta2', type=float, default=0.999, help='beta2 for Adam')
parser.add_argument('--pt_weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--pt_momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--pt_start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# crop
parser.add_argument('--pt_crop', type=float, default=0.2, help='minimum crop')

# dataset
parser.add_argument('--pt_dataset', type=str, default='hmdb51', choices=['hmdb51', 'ucf101', 'kinetics', 'diving48', 'sth_v1'])

# resume
parser.add_argument('--pt_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# augmentation setting
parser.add_argument('--pt_aug', type=str, default='CJ', choices=['NULL', 'CJ'])

# warm up
parser.add_argument('--pt_warm', action='store_true', help='add warm-up setting')
parser.add_argument('--pt_amp', action='store_true', help='using mixed precision')
parser.add_argument('--pt_opt_level', type=str, default='O2', choices=['O1', 'O2'])

# model definition
parser.add_argument('--pt_model', type=str, default='resnet50', choices=['resnet50', 'resnet50x2', 'resnet50x4'])
# loss function
parser.add_argument('--pt_softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
parser.add_argument('--pt_nce_k', type=int, default=16384)
parser.add_argument('--pt_nce_t', type=float, default=0.07)
parser.add_argument('--pt_nce_m', type=float, default=0.5)

# memory setting
parser.add_argument('--pt_moco', action='store_true', help='using MoCo (otherwise Instance Discrimination)')
parser.add_argument('--pt_alpha', type=float, default=0.999, help='exponential moving average weight')

# dataset setting
parser.add_argument('--pt_train_list', type=str, default='../datasets/lists/ucf101/ucf101_rgb_train_split_1.txt')
parser.add_argument('--pt_val_list', type=str, default='../datasets/lists/ucf101/ucf101_rgb_val_split_1.txt')
parser.add_argument('--pt_root', type=str, default="")
parser.add_argument('--pt_logits_channel', type=int, default=1024, help='channel of the last layers')

# ============================================== fine tune ====================================
parser.add_argument('--ft_mode', type=str, help='rgb or flow')
parser.add_argument('--ft_save_model', type=str, default='checkpoints/')
parser.add_argument('--ft_data', type=str, help='time')
parser.add_argument('--ft_dataset',default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'sth_v1', 'diving48'])
parser.add_argument('--ft_root', type=str, default="")
parser.add_argument('--ft_train_list', default='data/kinetics_rgb_train_list.txt', type=str)
parser.add_argument('--ft_val_list', default='data/kinetics_rgb_val_list.txt', type=str)
parser.add_argument('--ft_flow_prefix', default="", type=str)
parser.add_argument('--ft_snapshot_pref', type=str, default="")
parser.add_argument('--ft_logs_path', type=str, default="../experiments/logs/hmdb51_self_supervised")
parser.add_argument('--ft_stride', default=1, type=int, help='stride of temporal image')
parser.add_argument('--ft_weights', default="", type=str, help='checkpoints')
parser.add_argument('--ft_spatial_size', default='224', choices=['112', '224'], help='the network input size')
parser.add_argument('--ft_data_length', default='64', help='input clip length')
parser.add_argument('--ft_clips', default='4', help='global local clips num')
# ========================= Learing Stragety =========================
parser.add_argument('--ft_dropout', '--do', default=0.64, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--ft_mixup', type=int, help ='if use mixup do data augmentation', default=0)
parser.add_argument('--ft_momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--ft_weight_decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-7)')
parser.add_argument('--ft_lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--ft_lr_steps', default=[10, 20, 25, 30, 35, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--ft_optim', default='sgd', choices=['sgd', 'adam'])
# ========================= Monitor Configs ==========================
parser.add_argument('--ft_print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--ft_eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')
parser.add_argument('--ft_epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-ft_b', '--ft_batch_size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--ft_workers', type=int, default=4)

# =====================Runtime Config ==========================
parser.add_argument('--ft_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoints (default: none)')
parser.add_argument('-ft_e', '--ft_evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--ft_start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# ============================Evaluation=============================
parser.add_argument('--ft_test_clips', type=int, default=10)
parser.add_argument('--ft_clip_size', type=int, default=64)
parser.add_argument('--ft_prefix', type=str, default='Scratch')

# ============================Feature Extract========================
parser.add_argument('--ft_front', type=str, default='Scratch')

# =============================FineTune Layers=============================
parser.add_argument('--ft_fixed', type=int, default=0)

args = parser.parse_args()