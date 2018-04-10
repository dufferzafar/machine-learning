"""
Assignment 3 Question 2: Neural Networks.

Part A: Backprop with Batch Gradient Descent
Part B: Toy dataset
Part C: MNIST dataset
"""

import csv

import numpy as np

# from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# ~/.python/timtim
# from timtim import Timer as TimeIt

from neural_network import NeuralNetwork

from common import normalize
from visualization import plot_decision_boundary


def read_toy_data(neural=False):
    trX = np.loadtxt("data/toy_trainX.csv", delimiter=",")
    trY = np.loadtxt("data/toy_trainY.csv", delimiter=",")
    tsX = np.loadtxt("data/toy_testX.csv", delimiter=",")
    tsY = np.loadtxt("data/toy_testY.csv", delimiter=",")

    # Convert data in a form suitable for neural network?
    if neural:
        trX = np.array([x.reshape(-1, 1) for x in trX])
        tsX = np.array([x.reshape(-1, 1) for x in tsX])

    return trX, trY, tsX, tsY


def read_mnist_data(fn):
    X, Y = [], []

    # The CSV contains an image per row
    # with the first 784 numbers representing 28x28 pixels
    # and the last value representing the class label
    with open(fn + ".csv") as f:
        for row in csv.reader(f, delimiter=','):
            Y.append(int(row[-1]))
            X.append([int(n) for n in row[:-1]])

    X = np.array([x.reshape(-1, 1) for x in normalize(X)])
    Y = np.array([0 if y == 6 else 1 for y in Y])

    return X, Y


def part_b_1():
    toy_trainX, toy_trainY, toy_testX, toy_testY = read_toy_data()

    print()
    print("Part B 1")

    clf = LogisticRegression()
    clf.fit(toy_trainX, toy_trainY)

    print("Training Accuracy", 100 * clf.score(toy_trainX, toy_trainY))
    print("Testing Accuracy", 100 * clf.score(toy_testX, toy_testY))

    plot_decision_boundary(clf.predict, toy_trainX, toy_trainY,
                           "Toy Training Data", "b1_train")

    plot_decision_boundary(clf.predict, toy_testX, toy_testY,
                           "Toy Testing Data", "b1_test")


def part_b_2():

    toy_trainX, toy_trainY, toy_testX, toy_testY = read_toy_data()

    print()
    print("Part B 2")

    nn = NeuralNetwork(topo=[2, 5, 1])

    nn.train(toy_trainX, toy_trainY, epochs=5000, eta=4.5)

    print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
    print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))

    plot_decision_boundary(nn.predict, toy_trainX, toy_trainY,
                           "Toy Training Data", "b2_train_5")

    plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                           "Toy Testing Data", "b2_test_5")


def part_b_3():

    toy_trainX, toy_trainY, toy_testX, toy_testY = read_toy_data()

    print()
    print("Part B 3")

    # TODO: Not working with a single hidden unit - 2 1 1
    # for hidden_units in [2, 3, 10, 20, 40]:
    for hidden_units in [2, 3, 10, 20, 40]:

        print()
        print("Hidden units %d" % hidden_units)

        nn = NeuralNetwork(topo=[2, hidden_units, 1])
        nn.train(toy_trainX, toy_trainY, epochs=5000, eta=1)

        print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
        print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))

        plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                               "Toy Testing Data (%d hidden units)" % hidden_units,
                               "b3_test_%d" % hidden_units)


def part_b_4():

    toy_trainX, toy_trainY, toy_testX, toy_testY = read_toy_data()

    print()
    print("Part B 4")

    nn = NeuralNetwork(topo=[2, 5, 5, 1])

    nn.train(toy_trainX, toy_trainY, epochs=5000, eta=1)

    print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
    print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))

    plot_decision_boundary(nn.predict, toy_trainX, toy_trainY,
                           "Toy Training Data", "b4_train_5_5")
    plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                           "Toy Testing Data", "b4_test_5_5")


def part_c_1():
    # TODO: LIBSVM Linear

    print()
    print("Part C 1")

    print("Reading MNIST Data")
    trainX, trainY = read_mnist_data("data/MNIST_train")
    testX, testY = read_mnist_data("data/MNIST_test")

    nn = NeuralNetwork(topo=[784, 1])
    nn.train(trainX, trainY, eta=0,
             epochs=100, error_threshold=10**-5)

    print("Training Accuracy", 100 * nn.score(trainX, trainY))
    print("Testing Accuracy", 100 * nn.score(testX, testY))


def part_c_2():

    print()
    print("Part C 2")

    print("Reading MNIST Data")
    trainX, trainY = read_mnist_data("data/MNIST_train")
    testX, testY = read_mnist_data("data/MNIST_test")

    nn = NeuralNetwork(topo=[784, 100, 1])
    nn.train(trainX, trainY, eta=0,
             epochs=100, error_threshold=10**-5)

    print("Training Accuracy", 100 * nn.score(trainX, trainY))
    print("Testing Accuracy", 100 * nn.score(testX, testY))


def part_c_3():

    print()
    print("Part C 3")

    print("Reading MNIST Data")
    trainX, trainY = read_mnist_data("data/MNIST_train")
    testX, testY = read_mnist_data("data/MNIST_test")

    # TODO: ReLU at hidden layers, Sigmoid at output

    nn = NeuralNetwork(topo=[784, 100, 1])
    nn.train(trainX, trainY, eta=0,
             epochs=100, error_threshold=10**-5,
             use_relu=True)

    print("Training Accuracy", 100 * nn.score(trainX, trainY))
    print("Testing Accuracy", 100 * nn.score(testX, testY))


if __name__ == '__main__':
    # part_b_1()
    # part_b_2()
    # part_b_3()
    # part_b_4()

    # part_c_1()
    part_c_2()

    # part_c_3()
