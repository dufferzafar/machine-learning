import csv

import numpy as np
from matplotlib import pyplot as plt

from svmutil import *

DATA = "data/mnist/"


def show_img(arr):
    """Display image from array of values."""
    plt.imshow(arr.reshape((28, 28)))
    plt.axis('off')
    # TODO: Save image to file rather than showing
    plt.show()


def read_data(fn):
    y, x = [], []
    with open(DATA + fn + ".csv") as f:
        for row in csv.reader(f, delimiter=','):
            y.append(int(row[-1]))
            x.append([int(n) for n in row[:-1]])
    return y, x


def normalize(data):
    """Scale down features to range 0-1, for faster convergence."""
    X = np.asarray(data)

    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        X = (X - mn) / (mx - mn)
        X = np.nan_to_num(X, copy=False)

    return X.tolist()


def part_c():
    print("\n--- Part C ---\n")

    print("Reading Data")
    train_y, train_x = read_data("train")
    test_y, test_x = read_data("test")

    print("Normalizing")
    train_x = normalize(train_x)
    test_x = normalize(test_x)

    problem = svm_problem(train_y, train_x)
    params = svm_parameter("-q -s 0 -c 1")

    # TODO: Timing calculations
    print("Training SVM (linear kernel)")
    params.parse_options("-t 0")
    model = svm_train(problem, params)

    _, p_acc, _ = svm_predict(test_y, test_x, model)
    print("Accuracy: ", p_acc)

    print("Training SVM (gaussian kernel)")
    params.parse_options("-t 2 -g 0.05")
    model = svm_train(problem, params)

    _, p_acc, _ = svm_predict(test_y, test_x, model)
    print("Accuracy: ", p_acc)


if __name__ == '__main__':

    part_c()
