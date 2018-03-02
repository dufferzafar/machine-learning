import csv
import numpy as np

from svmutil import *  # noqa

DATA = "data/mnist/"


# TODO: Don't use numpy?
# def read_data(fn):
#     x = np.loadtxt(DATA + fn + ".csv", usecols=list(range(784)), delimiter=",")
#     y = np.loadtxt(DATA + fn + ".csv", usecols=[784], delimiter=",")
#     return x, y


def read_data(fn):
    y, x = [], []
    with open(DATA + fn + ".csv") as f:
        for row in csv.reader(f, delimiter=','):
            y.append(int(row[-1]))
            x.append([int(n) for n in row[:-1]])
    return y, x


def normalize(data):
    """Scale down features to range 0-1, for faster convergence."""
    data = np.asarray(data)
    for col in data.T:
        min_ = np.min(col)
        max_ = np.max(col)
        col = (col - min_) / (max_ - min_)
    return data.tolist()


def part_c():
    print("--- Part C ---")

    print("Reading Data")
    train_y, train_x = read_data("train")
    test_y, test_x = read_data("test")

    # TODO: Scale data to 0-1 range
    train_x = normalize(train_x)
    test_x = normalize(test_x)

    problem = svm_problem(train_y, train_x)

    # Takes around ~2 minutes with raw features
    print("Training SVM (linear kernel)")
    model = svm_train(problem, "-q -s 0 -t 0")
    _, p_acc, _ = svm_predict(test_y, test_x, model)
    print("Accuracy: ", p_acc)

    # print("Training SVM (gaussian kernel)")
    # model = svm_train(problem, "-q -s 0 -t 2 -c 1 -g 0.05")
    # _, p_acc, _ = svm_predict(*test, model)
    # print("Accuracy: ", p_acc)


if __name__ == '__main__':

    part_c()
