"""
Assignment 3 Question 1: Decision Trees.

Part A: Decision Tree ID3 / Entropy
Part B: Post Pruning the Tree
Part C: Dynamic Median Calculations
Part D: sklearn's Decision Tree
Part E: sklearn's Random Forest
"""

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm

# ~/.python/timtim
from timtim import Timer as TimeIt

from read_data import preprocess
from decision_tree import DecisionTree

# TODO: Move this data reading into a function


def read_rich_poor_data(binarize_median=True):
    return (
        preprocess("data/train.csv"),
        preprocess("data/test.csv"),
        preprocess("data/valid.csv"),
    )


def plot_node_accuracies(dtree, train_data, test_data, valid_data,
                         step=100, fn="plot"):
    """Plot accuracy of a decision tree as the number of nodes grow."""

    accuracies = {"train": [], "test": [], "valid": []}

    nodes = list(dtree.nodes())
    nodes.reverse()

    nodecounts = []
    totalnodes = len(nodes)

    for i in tqdm(range(0, len(nodes), step), ncols=80, ascii=True):

        # Remove "step" number of nodes at a time
        for node in nodes[i:i + step]:
            dtree._remove_node(node)

        # Prevent total number nodes from going below 0
        # (this will happen at the last iteration)
        totalnodes = max(totalnodes - step, 0)
        nodecounts.append(totalnodes)

        accuracies["train"].append(dtree.score(train_data))
        accuracies["test"].append(dtree.score(test_data))
        accuracies["valid"].append(dtree.score(valid_data))

    plt.plot(nodecounts, accuracies["train"])  # linestyle='-', marker='o')
    plt.plot(nodecounts, accuracies["test"])  # linestyle='-', marker='o')
    plt.plot(nodecounts, accuracies["valid"])  # linestyle='-', marker='o')

    plt.legend(['Train', 'Test', 'Validation'], loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of nodes')

    plt.savefig(fn + ".png")
    plt.close()


def part_a():

    print()
    print("Part A")

    train_data, test_data, valid_data = read_rich_poor_data()

    with TimeIt(prefix="Building Decision Tree"):
        dtree = DecisionTree(train_data)

    print()
    print("Tree height", dtree.height())
    print("Tree node count", dtree.node_count())

    print()
    print("Accuracy (training data)", dtree.score(train_data))
    print("Accuracy (testing data)", dtree.score(test_data))
    print("Accuracy (validation data)", dtree.score(valid_data))

    print()
    print("Calculating accuracy at different number of nodes")
    plot_node_accuracies(dtree, train_data, test_data, valid_data,
                         fn="output/dtree_part_a_before_pruning", step=50)


def part_b():

    print()
    print("Part B")

    train_data, test_data, valid_data = read_rich_poor_data()

    with TimeIt(prefix="Building Decision Tree"):
        dtree = DecisionTree(train_data)

    print()
    print("Pruning the tree")

    # dtree.prune_single_pass(valid_data)
    dtree.prune_brute(valid_data)

    print()
    print("Tree height", dtree.height())
    print("Tree node count", dtree.node_count())

    print()
    print("Accuracy (training data)", dtree.score(train_data))
    print("Accuracy (testing data)", dtree.score(test_data))
    print("Accuracy (validation data)", dtree.score(valid_data))

    print()
    print("Calculating accuracy at different number of nodes")
    plot_node_accuracies(dtree, train_data, test_data, valid_data,
                         fn="output/dtree_part_b_after_pruning", step=50)


def part_d():

    print()
    print("Part C")

    train_data, test_data, valid_data = read_rich_poor_data()

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

    print()
    print("Part D")

    train_data, test_data, valid_data = read_rich_poor_data()

    # Run with default parameters
    clf = RandomForestClassifier(criterion='entropy', random_state=0)
    clf.fit(train_data[:, 1:], train_data[:, 0])
    print("Random forest accuracy (default parameters)",
          clf.score(valid_data[:, 1:], valid_data[:, 0]))

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
    part_b()

    # part_d()
    # part_e()
