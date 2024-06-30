import os

import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import svm, metrics
from sklearn.cluster import KMeans

class_names = ["flooded", "no_flooded"]
sift = cv.SIFT().create()
# sift = cv.ORB().create()
qtd_clusters_kmeans = 150


def load_data_path(path, x, y):
    for img_path in os.listdir(path):
        if img_path.find("_1") < 0:
            y.append(0)
        else:
            y.append(1)
        img = cv.imread(f'{path}/{img_path}', cv.IMREAD_GRAYSCALE)
        x.append(img)


def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    load_data_path("data/train", x_train, y_train)
    load_data_path("data/test", x_test, y_test)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


# Extração de Características com SIFT
def get_sift_info(imgs):
    result = []
    for img in imgs:
        result.append(sift.detectAndCompute(img, None)[1])
    return result


# Agrupamento de Características
def group_features(features):
    all_descriptors = []
    for descriptor in features:
        if descriptor is not None:
            for des in descriptor:
                all_descriptors.append(des)

    kmeans = KMeans(n_clusters=qtd_clusters_kmeans)
    kmeans.fit(all_descriptors)
    return kmeans.cluster_centers_


def count_group_features(descriptors, bow):
    count_feat = []

    for i in range(len(descriptors)):
        feats = np.array([0] * qtd_clusters_kmeans)

        if descriptors[i] is not None:
            dist = cdist(descriptors[i], bow)
            argmin = np.argmin(dist, axis=1)

            for j in argmin:
                feats[j] += 1

        count_feat.append(feats)
    return count_feat


if __name__ == '__main__':
    print(f'---------- Loading dataset ------------')
    x_train, y_train, x_test, y_test = load_dataset()

    print(f'---------- Preparing dataset ----------')
    x_train_sift = get_sift_info(x_train)
    x_test_sift = get_sift_info(x_test)

    bow = group_features(x_train_sift)

    count_features = count_group_features(x_train_sift, bow)
    count_features_test = count_group_features(x_test_sift, bow)

    print(f'---------- Results SVM ----------------')

    # C -> Peso das classes (fator de regularizacao)
    clf = svm.SVC(C=5)
    clf.fit(count_features, y_train)

    y_pred = clf.predict(count_features_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
