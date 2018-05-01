from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from common import load_data, write_csv, accuracy, normalize

trX, trY, tsX = load_data()

# zero mean, unit variance
trX = normalize(trX)
tsX = normalize(tsX)


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

    print()
    print("Training Kmeans (max_iter=%d)" % max_iter)

    kmeans = KMeans(n_init=10, n_clusters=20,
                    max_iter=max_iter, random_state=0).fit(trX)
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


def plot_accuracies():
    max_iter = [10, 20, 30, 40, 50]

    # Values found by running part_b; and kaggle!

    # Before normalization
    # plt.plot(max_iter, [0.34652, 0.35422, 0.34787, 0.34687, 0.34600])
    # plt.plot(max_iter, [0.34572, 0.35220, 0.34515, 0.34645, 0.34520])

    # After normalization
    plt.plot(max_iter, [0.36303, 0.33084, 0.34694, 0.34913, 0.34914])
    plt.plot(max_iter, [0.35893, 0.32784, 0.34022, 0.34700, 0.34698])

    plt.xlabel("Maximum number of iterations")
    plt.ylabel("Accuracy")

    plt.legend(["Train", "Test"])

    plt.savefig("output/q1_a_kmeans_acc.png")
    plt.close()


if __name__ == '__main__':

    part_a(max_iter=300)
    # part_b()

    # plot_accuracies()
