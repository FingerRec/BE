# That's an impressive list of imports.
import numpy as np
import torch
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# %matplotlib inline

# We import seaborn to make nice plots.
import seaborn as sns
import os
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy


def load_data(file_name):
    # digits = load_digits()
    # digits.data.shape  # 1797 x 64
    # print(digits.data.shape)
    # print(digits['DESCR'])
    # return digits
    features = np.load(file_name, allow_pickle='TRUE').item()
    return features


def scatter(x, colors, num_class=10):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_class))
    # sns.palplot(sns.color_palette("hls", 10))
    # We create a scatter plot.
    labels=['brush_hair', 'cartwheel', 'catch', 'chew',
    'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword',
    'dribble']
    f = plt.figure(figsize=(8, 8))
    # print(colors.astype(np.int))
    ax = plt.subplot(aspect='equal')
    # for i in range(10):
    #     sc = ax.scatter(x[:, 0][30*i:30*(i+1)], x[:, 1][30*i:30*(i+1)], c=palette[colors.astype(np.int)][30*i:30*(i+1)],
    #                     s=40,
    #                     label=labels[i],
    #                     )
    sc = ax.scatter(x[:,0], x[:,1], c=palette[colors.astype(np.int)],
                    s=150,
                    #label=colors.astype(np.int)[30],
                    )
    # ax.legend(loc="best", title="Classes", bbox_to_anchor=(0.2, 0.4))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(num_class):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        # ax.legend(ytext, "a")
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    # ax.legend(('a','b','c','d','e'))
    return f, ax, sc, txts


def tsne_visualize(data, file_name, num_class=101):
    # nrows, ncols = 2, 5
    # plt.figure(figsize=(6,3))
    # plt.gray()
    # for i in range(ncols * nrows):
    #     ax = plt.subplot(nrows, ncols, i + 1)
    #     ax.matshow(digits.images[i,...])
    #     plt.xticks([]); plt.yticks([])
    #     plt.title(digits.target[i])
    # plt.savefig('../../../experiments/visualization/digits-generated.png', dpi=150)

    # We first reorder the data points according to the handwritten numbers.
    datas = []
    labels = []
    nums = len(data['target'])
    for i in range(num_class, num_class + 10):
        for j in range(nums):
            if data['target'][j] == i:
                datas.append(data['data'][j])
    X = np.vstack(datas)
    for i in range(num_class, num_class + 10):
        for j in range(nums):
            if data['target'][j] == i:
                labels.append(data['target'][j] - num_class)
    y = np.hstack(labels)
    # X = np.vstack([data['data'][data['target']==i].cpu()
    #                for i in range(10)])
    # y = np.hstack([data['target'][data['target']==i].cpu()
    #                for i in range(10)])
    # print(y)
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    scatter(digits_proj, y)
    plt.savefig(file_name, dpi=120)


# for begin_index in range(1,41,5):
#     front = 'self_supervised_kineticsMoCo'
#     s_path = "../../../experiments/visualization/" + front + str(begin_index) + 'to' + str(begin_index + 10)
#     if not os.path.exists(s_path):
#         os.mkdir(s_path)
#     features_file = "../../../experiments/visualization/{}_hmdb51_features.npy".format(front)
#     file_name = "{}/{}_hmdb51_tsne-generated.png".format(s_path, front)
#     data = load_data(features_file)
#     tsne_visualize(data, file_name, begin_index)

    # for front in ['scratch', 'fully_supervised_37', 'self_supervised_37','fully_supervised_scratch']:
    #     # front = 'scratch'
    #     features_file = "../../../experiments/visualization/{}_hmdb51_features.npy".format(front)
    #     file_name = "{}/{}hmdb51_tsne-generated.png".format(s_path, front)
    #     data = load_data(features_file)
    #     tsne_visualize(data, file_name, begin_index)


if __name__ == '__main__':
    # front = 'contrastive_kinetics_warpping_hmdb51'
    # front = 'contrastive_ucf101_warpping_hmdb51'
    front = 'i3d_fully_supervised_kinetics_warpping_hmdb51'
    for begin_index in range(1, 41, 5):
        s_path = "../../../experiments/visualization/TSNE/" + front + str(begin_index) + 'to' + str(begin_index + 10)
        if not os.path.exists(s_path):
            os.mkdir(s_path)
        features_file = "../../../experiments/features/{}/val_features.npy".format(front)
        file_name = "{}/{}_hmdb51_tsne-generated.png".format(s_path, front)
        data = load_data(features_file)
        tsne_visualize(data, file_name, begin_index)