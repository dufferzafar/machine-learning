"""
Assignment 3 Question 1: Decision Trees.

Part A: Decision Tree ID3 / Entropy
Part B: Post Pruning the Tree
Part C: Dynamic Median Calculations
Part D: sklearn's Decision Tree
Part E: sklearn's Random Forest
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier

from read_data import preprocess

from decision_tree import DecisionTree

# TODO: Move this data reading into a function
train_data = preprocess("data/train.csv")
test_data = preprocess("data/test.csv")
valid_data = preprocess("data/valid.csv")


def part_a():

    print("Building Decision Tree (on training data)")
    dtree = DecisionTree(train_data)

    print("Tree height", dtree.height())
    print("Tree node count", dtree.node_count())

    print("Accuracy (training data)", dtree.score(train_data))
    print("Accuracy (testing data)", dtree.score(test_data))
    print("Accuracy (validation data)", dtree.score(valid_data))


def part_d():

    # Run with default parameters
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(train_data[:, 1:], train_data[:, 0])

    print("")
    print("Accuracy", 100 * clf.score(valid_data[:, 1:], valid_data[:, 0]))
    print("Depth", clf.tree_.max_depth)
    print("Node Count", clf.tree_.node_count)

    # Plot the effect of maximum depth on accuracy
    accuracy = []
    depths = range(3, 30)
    for d in depths:
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0,
                                     max_depth=d)
        clf.fit(train_data[:, 1:], train_data[:, 0])
        accuracy.append(100 * clf.score(valid_data[:, 1:], valid_data[:, 0]))

    plt.figure(figsize=(6 * 1.5, 4 * 1.5))
    plt.plot(depths, accuracy, linestyle='--', marker='o')
    plt.xlabel("Maximum depth of tree")
    plt.ylabel("Validation accuracy")
    plt.show()

    # Range of parameters to find best accuracy over
    parameters = {
        'max_depth': range(8, 13),
        'min_samples_split': range(5, 100, 5),
        'min_samples_leaf': range(5, 75, 5),
    }

    # Run a custom search that scores on the validation set
    results = {}
    for param in ParameterGrid(parameters):
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0,
                                     **param)
        clf.fit(train_data[:, 1:], train_data[:, 0])

        validation_score = clf.score(valid_data[:, 1:], valid_data[:, 0])
        results[validation_score] = param

    print("Parameters resulting in accuracy %r" %
          max(results), results[max(results)])


def part_e():

    parameters = {
        'n_estimators': range(5, 25),
        'max_features': range(3, 15),
        'max_depth': range(8, 15),
        'bootstrap': [True, False],
    }

    results = {}
    for param in ParameterGrid(parameters):
        clf = RandomForestClassifier(criterion='entropy', random_state=0,
                                     **param)
        clf.fit(train_data[:, 1:], train_data[:, 0])

        validation_score = clf.score(valid_data[:, 1:], valid_data[:, 0])
        results[validation_score] = param

    print("Parameters resulting in accuracy %r" %
          max(results), results[max(results)])


if __name__ == '__main__':

    part_a()

    # part_d()
    # part_e()
