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


toy_trainX = np.loadtxt("data/toy_trainX.csv", delimiter=",")
toy_trainY = np.loadtxt("data/toy_trainY.csv", delimiter=",")

toy_testX = np.loadtxt("data/toy_testX.csv", delimiter=",")
toy_testY = np.loadtxt("data/toy_testY.csv", delimiter=",")


def part_b_1():
    clf = LogisticRegression()
    clf.fit(toy_trainX, toy_trainY)

    print("Training Accuracy", 100 * clf.score(toy_trainX, toy_trainY))
    plot_decision_boundary(clf.predict, toy_trainX, toy_trainY,
                           "Toy Training Data")

    print("Testing Accuracy", 100 * clf.score(toy_testX, toy_testY))
    plot_decision_boundary(clf.predict, toy_testX, toy_testY,
                           "Toy Testing Data")


def part_b_2():

    nn = NeuralNetwork(topo=[2, 5, 1])

    nn.train(toy_trainX, toy_trainY)

    print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
    plot_decision_boundary(nn.predict, toy_trainX, toy_trainY,
                           "Toy Training Data")

    print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))
    plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                           "Toy Testing Data")


def part_b_3():

    for hidden_units in [1, 2, 3, 10, 20, 40]:
        nn = NeuralNetwork(topo=[2, hidden_units, 1])

        print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
        print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))

        plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                               "Toy Testing Data (%d hidden units)" % hidden_units)


def part_b_4():

    nn = NeuralNetwork(topo=[2, 5, 5, 1])

    nn.train(toy_trainX, toy_trainY)

    print("Training Accuracy", 100 * nn.score(toy_trainX, toy_trainY))
    plot_decision_boundary(nn.predict, toy_trainX, toy_trainY,
                           "Toy Training Data")

    print("Testing Accuracy", 100 * nn.score(toy_testX, toy_testY))
    plot_decision_boundary(nn.predict, toy_testX, toy_testY,
                           "Toy Testing Data")


if __name__ == '__main__':
    part_b_1()
    # part_b_2()
    # part_b_3()
    # part_b_4()
