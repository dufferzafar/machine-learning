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
from common import accuracy


# TODO: Move this data reading into a function
train_data = preprocess("data/train.csv")
test_data = preprocess("data/test.csv")
valid_data = preprocess("data/valid.csv")


# Internal (decision) node of the tree
Node = namedtuple("Node", [
    "attr_idx",  # Index of the attribute to make decision on
    "children",  # A dictionary of attribute values and child nodes
    "cls",       # Majority class at this node
])

# Leaf nodes
Leaf = namedtuple("Leaf", [
    "cls",       # Majority class at this leaf
])


def entropy(Y):
    """
    Calculate entropy from class labels.
    """

    _, counts = np.unique(Y, return_counts=True)
    probabilities = counts.astype('float') / len(Y)

    return -1 * sum(p * np.log2(p) for p in probabilities if p)


def partition(Xa):
    """
    Partition a column based on the unique values it takes.

    { value: [indices where that value occurs in the column] }
    """
    return {v: np.where(Xa == v)[0] for v in np.unique(Xa)}


def best_attribute(data):
    """
    Use information gain to decide which attribute to split on.
    """

    # Need to find these parameters
    best_gain = -1
    best_attr = -1

    Y = data[:, 0]
    X = data[:, 1:]

    # Iterate over each attribute
    for i, Xa in enumerate(X.T):

        # Create partitions over this attribute
        entropy_Y_Xa = sum((len(p) / len(Xa)) * entropy(Y[p])
                           for p in partition(Xa).values())

        gain = entropy(Y) - entropy_Y_Xa

        # TODO: In case of a tie, choose the attribute which appears first in the
        # ordering as given in the training data.
        if gain > best_gain:
            best_gain = gain

            # NOTE: +1 because the data contains output variables at 1st column
            # so attributes/features start from 2nd column
            best_attr = i + 1

    return best_gain, best_attr


def build_decision_tree(data):
    """
    Build a decision tree using ID3 / information gain.

    First column (data[:, 0]) is the output class.
    """

    Y = data[:, 0]
    majority_class = np.bincount(Y).argmax()

    # if data is "pure" i.e has examples of a single class
    # then return a leaf node predicting that class
    if len(set(Y)) <= 1:
        return Leaf(cls=majority_class)

    # if all features finished?
    # TODO: Will info gain handle attribute repetitions?

    # Find the attribute that maximizes the gain
    gain, attr_idx = best_attribute(data)

    if gain > 0:
        # Split if gain is positive
        children = {v: build_decision_tree(data[p])
                    for v, p in partition(data[:, attr_idx]).items()}

        return Node(attr_idx, children, majority_class)
    else:
        # Otherwise create a leaf node that predicts the majority class
        return Leaf(cls=majority_class)


def dtree_predict(dtree, x):
    """Predict a single example using dtree."""
    if isinstance(dtree, Leaf):
        return dtree.cls
    else:
        child = dtree.children.get(x[dtree.attr_idx])

        # If there isn't a correct outgoing edge
        # then just return the majority class
        if not child:
            return dtree.cls
        else:
            return dtree_predict(child, x)


def dtree_score(dtree, data):
    """Find accuracy of dtree over data."""
    predictions = [dtree_predict(dtree, x) for x in data]
    return accuracy(data[:, 0], predictions)


def dtree_height(dtree):
    if isinstance(dtree, Leaf):
        return 0
    else:
        return 1 + max(map(dtree_height, dtree.children.values()))


def dtree_node_count(dtree):
    if isinstance(dtree, Leaf):
        return 1
    else:
        return 1 + sum(map(dtree_node_count, dtree.children.values()))


def part_a():

    print("Building Decision Tree (on training data)")
    dtree = build_decision_tree(train_data)

    print("Tree height", dtree_height(dtree))
    print("Tree node count", dtree_node_count(dtree))

    print("Accuracy (training data)", dtree_score(dtree, train_data))
    print("Accuracy (testing data)", dtree_score(dtree, test_data))
    print("Accuracy (validation data)", dtree_score(dtree, valid_data))


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
