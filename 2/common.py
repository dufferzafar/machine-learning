def accuracy(actual, predicted):
    """Find accuracy of the model."""

    # TODO: Build a confusion matrix

    # TODO: Convert the confusion matrix into a plot
    # https://stackoverflow.com/questions/2148543
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    assert len(actual) == len(predicted)

    correct = sum([a == p for (a, p) in zip(actual, predicted)])
    return correct / len(actual)
