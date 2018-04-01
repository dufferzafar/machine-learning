
import graphviz

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid

from matplotlib import pyplot as plt

from read_data import attributes, preprocess


def part_d():

    # Read the data
    train_data = preprocess("data/train.csv")
    # test_data = preprocess("data/test.csv")
    valid_data = preprocess("data/valid.csv")

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
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0)
        clf.fit(train_data[:, 1:], train_data[:, 0])
        accuracy.append(100 * clf.score(valid_data[:, 1:], valid_data[:, 0]))

    plt.figure(figsize=(6 * 1.5, 4 * 1.5))
    plt.plot(depths, accuracy, linestyle='--', marker='o')
    plt.xlabel("Maximum depth of tree")
    plt.ylabel("Validation accuracy")
    plt.show()

    # Sort features by their relative importance
    # list(sorted(zip(attributes, clf.feature_importances_), key=lambda x: x[1]))

    # Export the tree in graphviz format
    # dot_data = tree.export_graphviz(
    # clf, out_file="default_tree.dot",
    # feature_names=attributes[1:],
    # class_names=["poor", "rich"],
    # filled=True, rounded=True,
    # special_characters=True
    # )
    # graphviz.Source(dot_data)

    # Range of parameters to find best accuracy over
    parameters = {
        'max_depth': range(8, 13),
        'min_samples_split': range(5, 100, 5),
        'min_samples_leaf': range(5, 75, 5),
    }

    # Use cross-validation on training set to find best parameters
    # clf = GridSearchCV(DecisionTreeClassifier(criterion="entropy"), parameters)
    # clf.fit(train_data[:, 1:], train_data[:, 0])
    # clf.best_params_

    # Run a custom search that scores on the validation set
    results = {}
    for param in ParameterGrid(parameters):
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0, **param)
        clf.fit(train_data[:, 1:], train_data[:, 0])

        validation_score = clf.score(valid_data[:, 1:], valid_data[:, 0])
        results[validation_score] = param

    print("Parameters resultin in accuracy %r" % max(results), results[max(results)])


def part_e():

    parameters = {
        'n_estimators': range(5, 25),
        'max_features': range(3, 15),
        'max_depth': range(8, 15),
        'bootstrap': [True, False],
    }

    results = {}
    for param in ParameterGrid(parameters):
        clf = RandomForestClassifier(criterion='entropy', random_state=0, **param)
        clf.fit(train_data[:, 1:], train_data[:, 0])

        validation_score = clf.score(valid_data[:, 1:], valid_data[:, 0])
        results[validation_score] = param

    print("Parameters resultin in accuracy %r" % max(results), results[max(results)])


if __name__ == '__main__':

    part_d()
    part_e()
