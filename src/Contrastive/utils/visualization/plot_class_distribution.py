import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_histgorm(l, num=51):
    x = np.arange(1, num+1)
    fig, axs = plt.subplots(1, 1, sharex=True)
    axs[0].hist(l, bins=num)
    plt.savefig('visualization/histgorm.png')


def get_action_index(cls_list='../../../datasets/lists/hmdb51/hmdb51_classInd.txt'):
    action_label = []
    with open(cls_list) as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    for line in content:
        label, action = line.split(' ')
        action_label.append(action)
    return action_label


def analyse_record(scratch_label, scratch_predict, ssl_label, ssl_predict, dataset='hmdb51'):
    if dataset == 'hmdb51':
        class_num = 51
    elif dataset == 'ucf101':
        class_num = 101
    else:
        Exception("not implement dataset!")
    scratch_wrong = 0
    scratch_clsses = np.zeros(class_num)
    scratch_real_classes = np.zeros(class_num)
    for i in range(len(scratch_label)):
        max_index = scratch_predict[i]
        label = scratch_label[i]
        scratch_real_classes[label] += 1
        if max_index != label:
            scratch_wrong += 1
        else:
            scratch_clsses[label] += 1

    #=============================our self-supervised learning =======================
    self_supervised_wrong = 0
    self_supervised_classes = np.zeros(class_num)
    self_supervised_real_classes = np.zeros(class_num)
    for i in range(len(ssl_label)):
        max_index = ssl_predict[i]
        label = ssl_label[i]
        self_supervised_real_classes[label] += 1
        if max_index != label:
            self_supervised_wrong += 1
        else:
            self_supervised_classes[label] += 1
    print("scratch Top-1 is: {}".format(np.mean(scratch_clsses/(scratch_real_classes+1))))
    print("SSL Top-1 is: {}".format(np.mean(self_supervised_classes/self_supervised_real_classes)))
    arr = (self_supervised_classes/self_supervised_real_classes - scratch_clsses/(scratch_real_classes+1))
    topk = arr.argsort()[-5:][::-1]
    mink = arr.argsort()[:5][::1]
    if dataset == 'hmdb51':
        classes = get_action_index()
    elif dataset == 'ucf101':
        classes = get_action_index(cls_list='../../../datasets/lists/ucf101/classInd.txt')
    else:
        Exception("not implement dataset!")
    print("five largest ======>>")
    for i in range(5):
        print(classes[topk[i]], arr[topk[i]])
    print("five minium ======>>")
    for i in range(5):
        print(classes[mink[i]], arr[mink[i]])
    # print(self_supervised_real_classes/30, self_supervised_real_classes/30)
    # print((scratch_best_clsses - scratch_clsses) / 30)
    # print(scratch__wrong)
    # print(scratch__clsses/30, real_classes/30)
    # plot_histgorm(l_clsses, m_clsses, s_clsses)
    # rows = []
    # classes = get_action_index()
    # for i in range(len(l_clsses)):
    #     rows.append((l_clsses[i]/30, classes[i],
    #                  l_clsses[i] / 30 ))
    # header = ['l', 'm', 's', 'class', 'avg']
    # with open('visualization/store.csv', 'w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(header)
    #     f_csv.writerows(rows)
    return True


if __name__ == '__main__':
    # #====================================HMDB51============================================
    # #51.3
    # ssl_label = np.load("../../../experiments/evaluation/hmdb51/03-24-1659/video_labels.npy")
    # ssl_predict = np.load("../../../experiments/evaluation/hmdb51/03-24-1659/video_pred.npy")
    # #31.9
    # scratch_label = np.load("../../../experiments/evaluation/hmdb51/04-11-0926/video_labels.npy")
    # scratch_predict = np.load("../../../experiments/evaluation/hmdb51/04-11-0926/video_pred.npy")
    # analyse_record(scratch_label, scratch_predict, ssl_label, ssl_predict)
    # ====================================UCF101============================================
    # 78.83
    ssl_label = np.load("../../../experiments/evaluation/ucf101/04-12-1132/video_labels.npy")
    ssl_predict = np.load("../../../experiments/evaluation/ucf101/04-12-1132/video_pred.npy")
    # 63.3?
    scratch_label = np.load("../../../experiments/evaluation/ucf101/04-14-1141/video_labels.npy")
    scratch_predict = np.load("../../../experiments/evaluation/ucf101/04-14-1141/video_pred.npy")
    analyse_record(scratch_label, scratch_predict, ssl_label, ssl_predict, dataset='ucf101')