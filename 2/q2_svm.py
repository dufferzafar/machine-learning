import csv
import random
import pickle

from itertools import combinations

import numpy as np

from matplotlib import pyplot as plt
from svmutil import (
    svm_parameter,
    svm_predict,
    svm_problem,
    svm_train,
)

from common import plot_confusion, miss_rate, accuracy

DATA = "data/mnist/"


def read_data(fn):
    y, x = [], []
    with open(DATA + fn + ".csv") as f:
        for row in csv.reader(f, delimiter=','):
            y.append(int(row[-1]))
            x.append([int(n) for n in row[:-1]])
    return y, x


def svm_convert_data(fn):
    """
    Convert our data set into a format that libsvm can read.
    """

    with open(DATA + fn + "-svm", "w") as out:
        for y, x in zip(*read_data(fn)):
            out.write("%d " % y)
            out.write(" ".join(
                ["%d:%d" % (i, x) for i, x in enumerate(x) if x])
            )
            out.write("\n")


def normalize(data):
    """Scale down features to range 0-1, for faster convergence."""
    X = np.asarray(data)

    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        X = (X - mn) / (mx - mn)
        X = np.nan_to_num(X, copy=False)

    return X


# This is the part a of the assignment
def pegasos(X, y, lmbd=1):
    """
    Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.

    A batch SGD algorithm to find parameters w, b in SVM.

    X: Design matrix (training data's features)
    y: Labels (should be +1 / -1)

    Differences from the Pegasos paper:

    lambda is fixed to 1
    k is 1/C
    """

    m, n = X.shape

    # Batch size
    r = 100
    C = 1.0

    # Initial guess of W
    # What if this gets changed?
    W = np.zeros(n)
    b = 0

    iters = 0
    converged = False

    while not converged:

        iters += 1

        # Because this is stochastic descent
        # we decrease eta as we go ahead
        eta = 1 / iters

        # Find indices of elements in this batch
        batch = np.array(random.sample(range(m), r))
        Xb, yb = X[batch], y[batch]

        # Find examples in this batch for which T < 1
        # Are these points the "support vectors" ?
        T = yb * ((W @ Xb.T) + b)
        Tl1 = np.where(T < 1)

        Wp = W

        W = (1 - eta) * W + eta * C * np.sum(yb[Tl1] * Xb[Tl1].T, axis=1)
        b = b + eta * C * np.sum(yb[Tl1])

        # Convergence criteria
        if all(abs(Wp - W) <= 10**-3):
            converged = True

        # Prevent us from going in infinite loops
        if iters >= 10000:
            print("Abrupt Stop")
            break

    return W, b


def pegasos_train(train_x, train_y):
    """Build 1vs1 SVM classifiers using Pegasos."""

    classifiers = {}
    for classes in combinations(set(train_y), 2):

        # Let's call the first class positive
        # and the other negative
        pos, neg = classes

        # print("Training classifier between classes %d & %d" % (pos, neg))

        # Find examples of these classes
        Xpos = train_x[np.where(train_y == pos)]
        Xneg = train_x[np.where(train_y == neg)]
        X = np.concatenate((Xpos, Xneg))

        # The data contains classes from 1 - 10
        # but SVMs deal with +1 / -1
        y = np.array([1] * len(Xpos) + [-1] * len(Xneg))

        # Fit a classifier and store the parameters
        classifiers[classes] = pegasos(X, y)

    return classifiers


def pegasos_predict(data, classifiers):
    """Make predictions on data using 1v1 classifiers."""

    # Make predictions on test set
    predictions = []

    # Iterate over training set
    for x in data:

        # Prediction for this particular example
        p = []

        # Pass each example to all classifiers
        for classes, params in classifiers.items():
            pos, neg = classes
            W, b = params

            if W.T @ x + b > 0:
                p.append(pos)
            # In case of ties, predict digit with bigger value
            elif W.T @ x + b == 0:
                p.append(max(pos, neg))
            else:
                p.append(neg)

        # Find the class with the most count
        predictions.append(max(p, key=p.count))

    return predictions


def part_b():
    print("\n--- Part B ---\n")

    print("Reading Data")
    train_y, train_x = read_data("train")
    test_y, test_x = read_data("test")

    print("Normalizing")
    # train_x = np.asarray(train_x)
    # test_x = np.asarray(test_x)
    train_x = normalize(train_x)
    test_x = normalize(test_x)
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)

    # Build 10_C_2 classifiers
    print("Building Classifiers")
    classifiers = pegasos_train(train_x, train_y)

    with open("models/svm-model-1", "wb") as m:
        pickle.dump(classifiers, m)

    print("Finding Accuracy")
    predictions_test = pegasos_predict(test_x, classifiers)
    acc = accuracy(test_y.tolist(), predictions_test)
    print("Testing Accuracy: %.2f" % (acc * 100))

    predictions_train = pegasos_predict(train_x, classifiers)
    acc = accuracy(train_y.tolist(), predictions_train)
    print("Training Accuracy: %.2f" % (acc * 100))


def part_c():
    print("\n--- Part C ---\n")

    print("Reading Data")
    train_y, train_x = read_data("train")
    test_y, test_x = read_data("test")

    print("Normalizing")
    train_x = normalize(train_x).tolist()
    test_x = normalize(test_x).tolist()

    problem = svm_problem(train_y, train_x)
    params = svm_parameter("-q -s 0 -c 1")

    # Timing calculations
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


def part_d():
    print("\n--- Part D ---\n")

    print("Reading Data")
    train_y, train_x = read_data("train")
    test_y, test_x = read_data("test")

    print("Normalizing")
    train_x = normalize(train_x)
    test_x = normalize(test_x)

    problem = svm_problem(train_y, train_x)
    params = "-q -s 0 -t 2 -g 0.05"

    results = []
    for c in [10 ** -5, 10 ** -3, 1, 5, 10]:

        c = " -c %f " % c
        print("10-fold CV using" + c)
        cv_acc = svm_train(problem, params + c + "-v 10")

        print("On test data using" + c)
        model = svm_train(problem, params + c)
        _, test_acc, _ = svm_predict(test_y, test_x, model)
        print("C, Accuracy: ", c, cv_acc, test_acc)

        results.append((c, cv_acc, test_acc[0]))


def part_d_2():
    cvals = [0.00001, 0.001, 1, 5, 10]

    # These values were found by running the libsvm CLI tools
    c_acc = [71.59, 71.59, 97.355, 97.455, 97.455]
    t_acc = [72.11, 72.11, 97.23, 97.29, 97.29]

    c_line, = plt.plot(cvals, c_acc, label="Avg. Acc. after 10 fold cross-validation.",
                       linestyle='-', color='r', marker='x')

    t_line, = plt.plot(cvals, t_acc, label="Acc. on Test Set",
                       linestyle='-', color='b', marker='o')

    plt.legend(handles=[c_line, t_line])

    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Effect of varying value of C on accuracy of Gaussian kernel SVM")
    plt.savefig("part_d_2.png")
    plt.close()


def part_e():

    def imshow(arr, name):
        """Display image from array of values."""
        arr = np.asarray(arr)
        plt.imshow(arr.reshape((28, 28)), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.savefig("miss/%s.png" % name, bbox_inches='tight', pad_inches=0)
        plt.close()

    actual, train_x = read_data("test")

    # These labels were generated by running the
    predicted = np.loadtxt(
        "models/gaussian-test-labels-5", dtype="int").tolist()
    plot_confusion(actual, predicted, range(10), "Gaussian SVM (C=5)")

    predicted = np.loadtxt(
        "models/gaussian-test-labels-10", dtype="int").tolist()
    plot_confusion(actual, predicted, range(10), "Gaussian SVM (C=10)")

    # Find the class that is the hardest to classify using misclassification rate
    mr = miss_rate(actual, predicted)
    print(mr)

    # Examples that actually belong to class 8 but we predict them to be in class 9
    # Found while making the confusion matrix in make_confusion
    a9_p8 = [151, 241, 448, 1107, 2406, 6081, 6091, 6112, 6157, 6168, 617]
    a7_p2 = [810, 1226, 1283, 1754, 1941, 2016, 2325, 2607, 3767,
             4690, 4837, 5887, 7432, 8316, 9009, 9015, 9019, 9024, 9036, 9045]

    # Let us plot these to see what they are
    for idx in a9_p8:
        imshow(train_x[idx], "a9_p8_ex_%d" % idx)

    for idx in a7_p2:
        imshow(train_x[idx], "a7_p2_ex_%d" % idx)


if __name__ == '__main__':

    # In part a we just have to implement pegasos

    part_b()

    # Convert data to a format that libsvm can recognize
    # svm_convert_data("test")
    # svm_convert_data("train")

    # I wrote these parts first and only later found out that we could
    # directly use the C programs provided with libsvm: svm-train etc.
    # Those programs are highly optimized and have very low memory footprint.

    # Look in libsvm.sh on how to call those functions

    # part_c()
    # part_d()

    # part_d_2()

    # part_e()
