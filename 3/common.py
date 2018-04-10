import csv
import numpy as np


def accuracy(actual, predicted):
    """Find accuracy of the model."""

    assert len(actual) == len(predicted)

    correct = sum(a == p for (a, p) in zip(actual, predicted))
    return correct / len(actual)


def normalize(data):
    """Scale down features to range 0-1, for faster convergence."""
    X = np.asarray(data)

    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        X = (X - mn) / (mx - mn)
        X = np.nan_to_num(X, copy=False)

    return X


def svm_convert_data(input_file):
    """Convert our data set into a format that libsvm can read."""

    def read_data(fn):
        y, x = [], []
        with open(fn + ".csv") as f:
            for row in csv.reader(f, delimiter=','):
                y.append(int(row[-1]))
                x.append([int(n) for n in row[:-1]])
        return y, x

    with open(input_file + "_svm", "w") as out:
        for y, x in zip(*read_data(input_file)):
            out.write("%d " % y)
            out.write(" ".join(
                ["%d:%d" % (i, x) for i, x in enumerate(x) if x])
            )
            out.write("\n")


# if __name__ == '__main__':

#     print("Converting data into a format that SVM can read")
#     svm_convert_data("data/MNIST_test")
#     svm_convert_data("data/MNIST_train")
