from option import args
import datetime
from pt import pretext_train
from ft import fine_tune_train_and_val
import os
from utils.recoder import Record


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.date = datetime.datetime.today().strftime('%m-%d-%H%M')
    recorder = Record(args)
    message = ""
    if args.method == 'pt':
        args.status = 'pt'
        pretext_train(args, recorder)
        print("finished pretrain with weight from: {}".format(args.ft_weights))
    elif args.method == 'ft':
        args.status = 'ft'
        fine_tune_train_and_val(args, recorder)
        print("finished finetune with weight from: {}".format(args.ft_weights))
    elif args.method == 'pt_and_ft':
        args.status = 'pt'
        checkpoints_path = pretext_train(args, recorder)
        print("finished pretrain, the weight is in: {}".format(args.ft_weights))
        args.status = 'ft'
        args.ft_weights = checkpoints_path
        fine_tune_train_and_val(args, recorder)
        print("finished finetune with weight from: {}".format(checkpoints_path))
    else:
        Exception("wrong method!")


if __name__ == '__main__':
    main()
