"""
input: 16 x 112 x 112 x 3, with no overlapped
"""
import torch.nn.parallel
import torch.optim
from utils.utils import *
import os
from data.config import data_config, augmentation_config
from data.dataloader import data_loader_init
from model.config import model_config
from bk.option_old import args
import datetime


def get_action_index(list_txt='data/classInd.txt'):
    action_label = []
    with open(list_txt) as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    for line in content:
        label, action = line.split(' ')
        action_label.append(action)
    return action_label


def plot_matrix_test(list_txt, cfu_mat="../experiments/evaluation/ucf101/_confusion.npy", date="",prefix="flow"):
    classes = get_action_index(list_txt)
    confuse_matrix = np.load(cfu_mat)
    plot_confuse_matrix(confuse_matrix, classes, date=date, prefix=prefix)
    plt.show()


def eval_video(net, video_data):
    '''
    average 10 clips, do it later
    '''
    i, datas, label = video_data
    output = None
    net.eval()
    with torch.no_grad():
        for data in datas:
            if len(data.size()) == 4:
                data = data.unsqueeze(0)
            # print(data.size())
            overlapped_clips = 1 + int((data.size(2) - args.clip_size) / 10)
            # print(overlapped_clips)
            for i in range(overlapped_clips):
                if i > 1:
                    break
                # print(data.size()) # 3 x 47 x 112 x 112
                clip_data = data[:, :, 10 * i:10 * i + args.clip_size, :, :]
                input_var = torch.autograd.Variable(clip_data)
                res = net(input_var)
                # print(torch.exp(res), label)
                res = torch.exp(res).data.cpu().numpy().copy()
                if output is None:
                    output = res / overlapped_clips
                else:
                    output += res / overlapped_clips
    return output, label


def main(prefix='flow_scratch'):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # == dataset config==
    num_class, data_length, image_tmpl = data_config(args)
    train_transforms, test_transforms, eval_transforms = augmentation_config(args)
    _, eval_data_loader, _, _, _, _ = data_loader_init(args, data_length, image_tmpl, train_transforms,
                                                                test_transforms, eval_transforms)
    model = model_config(args, num_class)
    output = []
    total_num = len(eval_data_loader)
    for i, (data, label, index) in enumerate(eval_data_loader):
        proc_start_time = time.time()
        rst = eval_video(model, (i, data, label))
        output.append(rst)
        cnt_time = time.time() - proc_start_time
        if i % 10 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                            total_num,
                                                                            float(cnt_time) / (i + 1)))
        if i > 300:
            video_pred = [np.argmax(x[0]) for x in output]
            video_labels = [x[1] for x in output]
            cf = confusion_matrix(video_labels, video_pred).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

    date = datetime.datetime.today().strftime('%m-%d-%H%M')
    # =====output: every video's num and every video's label
    # =====x[0]:softmax value x[1]:label
    if not os.path.isdir("../experiments/evaluation/{}/{}".format(args.dataset, date)):
        os.mkdir("../experiments/evaluation/{}/{}".format(args.dataset, date))
    video_pred = [np.argmax(x[0]) for x in output]
    np.save("../experiments/evaluation/{}/{}/{}_video_pred.npy".format(args.dataset, date, prefix), video_pred)
    video_labels = [x[1] for x in output]
    np.save("../experiments/evaluation/{}/{}/{}_video_labels.npy".format(args.dataset, date, prefix), video_labels)
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    np.save("../experiments/evaluation/{}/{}/{}_confusion.npy".format(args.dataset, date, prefix), cf)
    cf_name = "../experiments/evaluation/{}/{}/{}_confusion.npy".format(args.dataset, date, prefix)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    print(cls_acc)
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

    name_list = [x.strip().split()[0] for x in open(args.val_list)]
    order_dict = {e: i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]
    np.savez('../experiments/evaluation/' + args.dataset + '/' + date + "/" + prefix + args.mode + 'res',
             scores=reorder_output, labels=reorder_label)
    return cf_name, date


def plot_confuse_matrix(matrix, classes,
                        date="",
                        prefix="flow",
                        normalize=True,
                        title=None,
                        cmap=plt.cm.Blues
                        ):
    """
    :param matrix:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()

    # We change the fontsize of minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=4)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    print("date is: {}".format(date))
    plt.savefig("../experiments/evaluation/hmdb51/{}/{}confuse.png".format(date, prefix))
    return ax


if __name__ == '__main__':
    # prefix = 'TCA'
    cf_name, date = main(args.prefix)
    # cf_name = "../experiments/evaluation/hmdb51/03-24-1659/confusion.npy"
    classList = "../datasets/lists/hmdb51/hmdb51_classInd.txt"
    plot_matrix_test(classList, cf_name, date, prefix=args.prefix)