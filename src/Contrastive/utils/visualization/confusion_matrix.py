import numpy as np
import matplotlib.pyplot as plt

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

def plot_matrix_test(list_txt, cfu_mat="../experiments/evaluation/ucf101/_confusion.npy", s_path="../experiments/evaluation/hmdb51/03-24-1659/confuse.png"):
    classes = get_action_index(list_txt)
    confuse_matrix = np.load(cfu_mat)
    plot_confuse_matrix(confuse_matrix, classes, s_path)
    plt.show()

def plot_confuse_matrix(matrix, classes,
                        s_path = "../experiments/evaluation/hmdb51/03-24-1659/confuse.png",
                        normalize=True,
                        title=None,
                        cmap=plt.cm.Blues
                        ):
    """
    :param matrix:
    :param classes:
    :param s_path:
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
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.tick_params(axis='both', which='minor', labelsize=1)

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
    fmt = '.2f'
    # thresh = cm.max() / 2.
    thresh = 0.2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > thresh and i != j:
                ax.text(j, i, "({},{})".format(classes[i], classes[j]) + format(cm[i, j], fmt),
                        ha="center", va="center",fontsize='smaller',
                        color="black")
    fig.tight_layout()
    plt.savefig(s_path, dpi=1024)
    return ax

if __name__ == '__main__':
    # cf_name = "../experiments/evaluation/ucf101/04-12-1132/confusion.npy"
    # classList = "../datasets/lists/ucf101/classInd.txt"
    # save_path = "../experiments/evaluation/ucf101/04-12-1132/78.8_confuse.png"
    # plot_matrix_test(classList, cf_name, save_path)
    cf_name = "../experiments/evaluation/ucf101/04-14-1141/confusion.npy"
    classList = "../datasets/lists/ucf101/classInd.txt"
    save_path = "../experiments/evaluation/ucf101/04-14-1141/63.3_confuse.png"
    plot_matrix_test(classList, cf_name, save_path)