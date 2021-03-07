import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
import csv


def plot_histgorm(l, num=51):
    x = np.arange(1, num+1)
    fig, axs = plt.subplots(1, 1, sharex=True)
    axs[0].hist(l, bins=num)
    plt.savefig('../../../experiments/visualization/histgorm.png')


def plot_linear(x, y, prefix='scratch', x_name='RGB Scratch', y_name='Relative Performance [SSL - RGB]'):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'ro')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('Class Average Performance-Top-1(%)')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    # predict y from the data
    x_new = np.linspace(0, 1, 100)
    y_new = model.predict(x_new[:, np.newaxis])
    ax.plot(x_new, y_new, 'b')
    #
    # # set ticks and tick labels
    # ax.set_xlim((0, 1))
    # ax.set_xticks([0, 0.5, 1])
    # ax.set_xticklabels(['0', '0.5', '1'])
    # ax.set_ylim((0, 1.5))
    # ax.set_yticks([0, 0.5, 1])
    #
    # # Only draw spine between the y-ticks
    # ax.spines['left'].set_bounds(-1, 1)
    # # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

    # plt.show()
    plt.savefig('../../../experiments/visualization/{}flow_scratch.png'.format(prefix))


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


def analyse_record(scratch_label, scratch_predict, ssl_label, ssl_predict, dataset='hmdb51', prefix='scratch'):
    if dataset == 'hmdb51':
        class_num = 51
    elif dataset == 'ucf101':
        class_num = 101
    else:
        Exception("not implement dataset!")
    # ================================= scratch =======================================
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

    # =============================our self-supervised learning =======================
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

    # ============================relative performance ==================================
    arr_x = scratch_clsses/(scratch_real_classes+1)
    arr_y = self_supervised_classes/self_supervised_real_classes
    plot_linear(arr_x, arr_y, prefix=prefix)
    return True


def analyse_relative_performance(rgb_scratch_label, rgb_scratch_predict, flow_scratch_label, flow_scratch_predict, ssl_label, ssl_predict, dataset='hmdb51', prefix='scratch'):
    if dataset == 'hmdb51':
        class_num = 51
    elif dataset == 'ucf101':
        class_num = 101
    else:
        Exception("not implement dataset!")
    # ================================= rgb scratch =======================================
    rgb_scratch_wrong = 0
    rgb_scratch_clsses = np.zeros(class_num)
    rgb_scratch_real_classes = np.zeros(class_num)
    for i in range(len(rgb_scratch_label)):
        max_index = rgb_scratch_predict[i]
        label = rgb_scratch_label[i]
        rgb_scratch_real_classes[label] += 1
        if max_index != label:
            rgb_scratch_wrong += 1
        else:
            rgb_scratch_clsses[label] += 1

    # ================================= flow scratch ====================================
    flow_scratch_wrong = 0
    flow_scratch_clsses = np.zeros(class_num)
    flow_scratch_real_classes = np.zeros(class_num)
    for i in range(len(flow_scratch_label)):
        max_index = flow_scratch_predict[i]
        label = flow_scratch_label[i]
        flow_scratch_real_classes[label] += 1
        if max_index != label:
            flow_scratch_wrong += 1
        else:
            flow_scratch_clsses[label] += 1
    # =============================our self-supervised learning =======================
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
    print("RGB scratch Top-1 is: {}".format(np.mean(rgb_scratch_clsses/(rgb_scratch_real_classes+1))))
    print("Flow scratch Top-1 is: {}".format(np.mean(flow_scratch_clsses / (flow_scratch_real_classes + 1))))
    print("SSL Top-1 is: {}".format(np.mean(self_supervised_classes/self_supervised_real_classes)))

    # ============================relative performance ==================================
    arr_x = flow_scratch_clsses/(rgb_scratch_real_classes+1)
    arr_y = (self_supervised_classes-rgb_scratch_clsses)/self_supervised_real_classes
    header = ['class', 'scratch_x', 'avg_y']
    rows = []
    classes = get_action_index()
    for i in range(len(arr_x)):
        rows.append((classes[i], arr_x[i], arr_y[i]))
    with open('store.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(rows)
    plot_linear(arr_x, arr_y, prefix=prefix)
    print("Finished!")
    return True


if __name__ == '__main__':
    # ====================================HMDB51============================================
    # (RGB Scratch)
    rgb_scratch_label = np.load("../../../experiments/evaluation/hmdb51/04-11-0926/rgb_scratch_video_labels.npy")
    rgb_scratch_predict = np.load("../../../experiments/evaluation/hmdb51/04-11-0926/rgb_scratch_video_pred.npy")
    # 32.0 (flow scratch)
    flow_scratch_label = np.load("../../../experiments/evaluation/hmdb51/08-04-1154/flow_scratch_video_labels.npy")
    flow_scratch_predict = np.load("../../../experiments/evaluation/hmdb51/08-04-1154/flow_scratch_video_pred.npy")
    # (upper bound flow) Kinetics_I3D_Flow
    flow_kinetics_i3d_label = np.load("../../../experiments/evaluation/test_output/hmdb51/flow_79.86_video_labels.npy")
    flow_kinetics_i3d_predict = np.load("../../../experiments/evaluation/test_output/hmdb51/flow_79.86_video_pred.npy")
    # 43.9 (TCA RGB)
    ssl_label = np.load("../../../experiments/evaluation/hmdb51/08-04-1519/rgb_TCA_video_labels.npy")
    ssl_predict = np.load("../../../experiments/evaluation/hmdb51/08-04-1519/rgb_TCA_video_pred.npy")
    # prefix = 'rgb_scratch_vs_flow_scratch'
    # analyse_record(flow_scratch_label, flow_scratch_predict, rgb_scratch_label, rgb_scratch_predict, prefix=prefix)
    # prefix = 'ssl-rgb_scratch_vs_flow_scratch'
    # analyse_relative_performance(rgb_scratch_label, rgb_scratch_predict, flow_scratch_label, flow_scratch_predict, ssl_label, ssl_predict, prefix=prefix)
    prefix = 'ssl-rgb_scratch_vs_rgb_scratch'
    analyse_relative_performance(rgb_scratch_label, rgb_scratch_predict, rgb_scratch_label, rgb_scratch_predict,
                                 ssl_label, ssl_predict, prefix=prefix)
    # prefix = 'ssl-rgb_scratch_vs_flow_kinetics_i3d'
    # analyse_relative_performance(rgb_scratch_label, rgb_scratch_predict, flow_kinetics_i3d_label, flow_kinetics_i3d_predict,
    #                              ssl_label, ssl_predict, prefix=prefix)
    # analyse_record(rgb_scratch_label, rgb_scratch_predict, ssl_label, ssl_predict, prefix=prefix)