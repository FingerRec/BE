"""Video retrieval experiment, top-k."""
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def topk_retrieval(feature_dir):
    """Extract features from test split and search on train split features."""
    print('Load local .npy files. from ...', feature_dir)
    train_features = np.load(os.path.join(feature_dir, 'train_features.npy'), allow_pickle=True).item()
    X_train = train_features['data']
    y_train = train_features['target']
    # X_train = np.mean(X_train, 1)
    # y_train = y_train[:, 0]
    # X_train = X_train.reshape((-1, X_train.shape[-1]))
    # y_train = y_train.reshape(-1)

    val_features = np.load(os.path.join(feature_dir, 'val_features.npy'), allow_pickle=True).item()
    X_test = val_features['data']
    y_test = val_features['target']
    # X_test = np.mean(X_test, 1)
    # y_test = y_test[:, 0]
    # X_test = X_test.reshape((-1, X_test.shape[-1]))
    # y_test = y_test.reshape(-1)

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k: 0 for k in ks}

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)  # 1530 x 3570

    for k in ks:
        top_k_indices = indices[:, :k]
        for ind, test_label in zip(top_k_indices, y_test):
            # print(ind)
            for j in range(len(ind)):
                labels = y_train[ind[j]]
                if test_label in labels:
                    topk_correct[k] += 1
                    break

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}%'.format(k, correct, total, correct / total * 100))

    with open(os.path.join(feature_dir, 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)


if __name__ == '__main__':
    # front = "contrastive_kinetics_warpping"
    # front = "triplet_ucf101_warpping_hmdb51"
    # front = "contrastive_ucf101_warpping_hmdb51"
    # front = "contrastive_ucf101_warpping_ucf101"
    # front = "triplet_ucf101_warpping_hmdb51"
    # front = "i3d_fully_supervised_kinetics_warpping_hmdb51_finetune"
    # front = "c3d_fully_supervised_ucf101_warpping_hmdb51"
    front = "c3d_c3d_contrastive_ucf101_warpping_ucf101_ucf101"
    feature_dirs = "../experiments/features/{}".format(front)
    topk_retrieval(feature_dirs)


# ==============================kinetics BE contrastive pretrain =======================
# hmdb51: 11.9 / 31.3 / 44.452 / 60.432 / 81.23
#  ucf101: 13.0/35.16/44.0/64.78/83.76

# ===============================ucf101 BE triplet pretrain===================================
# hmdb51: 3.922 / 17.386 / 29.15 / 45.359 / 69.020 (may not best)

# =============================ucf101 BE contrastive pretrain===========================
#  hmdb51: 11.4 / 31.2 / 46.5/ 60.4 / 80.876
# ucf101: 17.394 / 35.184 / 45.308 / 57.811 / 73.962
# c3d
# hmdb51: 8.23/25.88/38.10/51.96/75.0
