def accuracy(actual, predicted):
    """Find accuracy of the model."""

    # TODO: Build a confusion matrix

    # TODO: Convert the confusion matrix into a plot
    # https://stackoverflow.com/questions/2148543
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    assert len(actual) == len(predicted)

    correct = sum([a == p for (a, p) in zip(actual, predicted)])
    return correct / len(actual)


def make_confusion(actual, predicted):
    """Create a confusion matrix from two list of labels."""

    assert len(actual) == len(predicted)

    # TODO: Not proud of this code - Make Pythonic!

    # Using a dict ensures that classes can be anything
    cm = {
        c: {cc: 0 for cc in set(actual)}
        for c in set(actual)
    }

    # Update entries with counts
    for a, p in zip(actual, predicted):
        cm[a][p] += 1

    # Now convert into a matrix form
    # TODO: See if seaborn can take dictionaries as input
    cmm = []
    for a in sorted(cm):
        cmm.append([cm[a][p] for p in sorted(cm[a])])

    return cmm
