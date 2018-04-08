"""
Assignment 3 Question 2: Neural Networks.

Part A: Backprop with Batch Gradient Descent
Part B: Toy dataset
Part C: MNIST dataset
"""

import numpy as np

# from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# ~/.python/timtim
# from timtim import Timer as TimeIt

from neural_network import NeuralNetwork

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
    print("Architecture: 2 - 5 - 1")

    nn = NeuralNetwork(topo=[2, 5, 1])

    nn.train(toy_trainX, toy_trainY, epochs=5000, eta=0.1)

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

    for hidden_units in [2, 3, 10, 20, 40]:

        print()
        print("Hidden units %d" % hidden_units)

        nn = NeuralNetwork(topo=[2, hidden_units, 1])
        nn.train(toy_trainX, toy_trainY, epochs=5000, eta=0.1)

        print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
        print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))

        plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                               "Toy Testing Data (%d hidden units)" % hidden_units,
                               "b3_test_%d" % hidden_units)


def part_b_4():

    toy_trainX, toy_trainY, toy_testX, toy_testY = read_toy_data()

    print()
    print("Part B 4")
    print("Architecture: 2 - 5 - 5 - 1")

    nn = NeuralNetwork(topo=[2, 5, 5, 1])

    nn.train(toy_trainX, toy_trainY, epochs=5000, eta=0.2)

    print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
    print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))

    plot_decision_boundary(nn.predict, toy_trainX, toy_trainY,
                           "Toy Training Data")
    plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                           "Toy Testing Data")


if __name__ == '__main__':
    part_b_1()
    # part_b_2()
    # part_b_3()
    # part_b_4()
