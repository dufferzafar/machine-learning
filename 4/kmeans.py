from collections import Counter

import numpy as np

from sklearn.cluster import KMeans

from common import load_data, write_csv, accuracy

trX, trY, tsX = load_data()


def cluster_labels(km, Y):
    """Get labels of clusters from kmeans object according to Y."""

    # Indices of examples in each cluster
    cluster_indices = {c: np.where(km.labels_ == c)[0] for c in range(20)}

    # Assign majority label to clusters
    labels = {}
    for k, indices in cluster_indices.items():
        labels[k] = Counter(Y[indices]).most_common()[0][0]

    # Convert the labels to an np.array to make things easier later
    return np.array(list(labels.values()))


def part_a(max_iter=300):
    print("Training Kmeans (max_iter=%d)" % max_iter)

    kmeans = KMeans(n_init=10, n_clusters=20, max_iter=max_iter, random_state=0).fit(trX)
    labels = cluster_labels(kmeans, trY)

    # Find training accuracy
    trP = labels[kmeans.predict(trX)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    # Dump test labels
    # Test accuracy can only be calculated by uploading to Kaggle
    tsP = labels[kmeans.predict(tsX)]
    write_csv("kmeans_%d.csv" % max_iter, tsP)


def part_b():

    for max_iter in [10, 20, 30, 40, 50]:
        part_a(max_iter)


if __name__ == '__main__':
    part_a()
