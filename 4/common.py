import csv
import glob

import numpy as np

from matplotlib import pyplot as plt


def normalize(data):
    return (data - data.mean(axis=0)) / (data.std(axis=0))


def load_data():
    """Load training & test data."""

    X, Y = [], []
    for file in sorted(glob.glob("data/train/*.npy")):
        x = np.load(file)
        y = [file.split('/')[-1][:-4]] * len(x)

        X.append(x)
        Y.append(y)

    trX = np.concatenate(X, axis=0)
    trY = np.concatenate(Y, axis=0)
    tsX = np.load("data/test/test.npy")

    return trX.astype("float64"), trY, tsX.astype("float64")


def write_csv(file, labels):
    """Write test labels onto CSV file."""

    print("Writing labels to: %s" % file)

    with open(file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["ID", "CATEGORY"])

        for i, label in enumerate(labels):
            writer.writerow([i, label])


def imshow(arr, name=""):
    """Display image from array of values."""
    arr = np.asarray(arr)
    plt.imshow(arr.reshape((28, 28)), cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()


def accuracy(actual, predicted):
    """Find accuracy of the model."""

    assert len(actual) == len(predicted)

    correct = sum([a == p for (a, p) in zip(actual, predicted)])
    return correct / len(actual)
