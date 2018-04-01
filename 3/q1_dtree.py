"""
Assignment 3 Question 1: Decision Trees.

Part A: Decision Tree ID3 / Entropy
Part B: Post Pruning the Tree
Part C: Dynamic Median Calculations
Part D: sklearn's Decision Tree
Part E: sklearn's Random Forest
"""

from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier

from read_data import preprocess


# TODO: Move this data reading into a function
train_data = preprocess("data/train.csv")
test_data = preprocess("data/test.csv")
valid_data = preprocess("data/valid.csv")


# Internal (decision) node of the tree
Node = namedtuple("Node", [
    "attr_idx",     # Index of the attribute to make decision on
    "threshold",    # Threshold value for the attribute
    "true_branch",  # Branch where data[attr_idx] <= threshold
    "false_branch"  # Branch where data[attr_idx] >  threshold
])

# Leaf nodes
Leaf = namedtuple("Leaf", [
    "cls"           # Class to predict at this leaf
])


def entropy(Y):
    """
    Calculate entropy from class labels.
    """

    _, counts = np.unique(Y, return_counts=True)
    probabilities = counts.astype('float') / len(Y)

    return sum(-1 * p * np.log2(p) for p in probabilities if p)


def best_attribute(data):
    """
    Use information gain to decide which attribute to split on.
    """

    Y = data[:, 0]
    X = data[:, 1:]

    # Find these parameters
    best_gain = 0.0
    best_attr = 0
    best_threshold = 0
    true_data = []
    false_data = []

    # Iterate over each attribute
    for Xa in X.T:

        for val in np.unique(Xa):



def build_decision_tree(data):
    """
    Build a decision tree using ID3 / information gain.

    First column (data[:, 0]) is the output class.
    """

    Y = data[:, 0]

    # if data is "pure" i.e has examples of a single class
    # then return a leaf node predicting that class
    if len(set(Y)) == 1:
        return Leaf(cls=data[0][0])

    # Find the attribute that maximizes the gain
    gain, attr_idx, threshold, true_data, false_data = best_attribute(data)

    if gain > 0:
        # Split if gain is positive
        return Node(attr_idx, threshold,
                    build_decision_tree(true_data),
                    build_decision_tree(false_data))
    else:
        # Otherwise create a leaf node that predicts the majority class
        return Leaf(cls=np.bincount(Y).argmax())


def dtree_predict(x, dtree):
    """
    Predict a single example using dtree.
    """

    if isinstance(dtree, Leaf):
        return dtree.cls


def part_a():

    dtree = build_decision_tree(train_data)

    train_acc = dtree_score(dtree, train_data)
    test_acc = dtree_score(dtree, test_data)
    valid_acc = dtree_score(dtree, valid_data)


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

    part_d()
    part_e()
