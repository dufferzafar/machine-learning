import matplotlib.pyplot as plt
import seaborn as sn


def accuracy(actual, predicted):
    """Find accuracy of the model."""

    assert len(actual) == len(predicted)

    correct = sum([a == p for (a, p) in zip(actual, predicted)])
    return correct / len(actual)


def make_confusion(actual, predicted, dict_=False):
    """Create a confusion matrix from two list of labels."""

    assert len(actual) == len(predicted)

    # Not proud of this code - Make Pythonic!

    # Using a dict ensures that classes can be anything
    cm = {
        c: {cc: 0 for cc in set(actual)}
        for c in set(actual)
    }

    # Update entries with counts
    for i, (a, p) in enumerate(zip(actual, predicted)):
        cm[a][p] += 1

        # These checks were added AFTER generating the confusion matrix
        # to find out examples that are being misclassified.
        # if a == 9 and p == 8:
        # if a == 7 and p == 2:
        #     print(i)

    # If a dict is required, return!
    if dict_:
        return cm

    # Now convert into a matrix form
    # See if seaborn can take dictionaries as input
    cmm = []
    for a in sorted(cm):
        cmm.append([cm[a][p] for p in sorted(cm[a])])

    return cmm


def plot_confusion(actual, predicted, ticks, title):
    """Plot the confusion matrix."""

    cm = make_confusion(actual, predicted)
    acc = " - %.2f%% accuracy" % (accuracy(actual, predicted) * 100)

    # Could replace the above line by sklearn.metrics.confusion_matrix
    # cm = confusion_matrix(actual, predicted)

    plt.figure(figsize=(10, 7))

    ax = sn.heatmap(cm, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=ticks, yticklabels=ticks)
    ax.set(xlabel="Predicted", ylabel="Actual")

    # Move X-Axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title(title + acc, y=1.10)

    plt.savefig(title + ".png")
    plt.close()


def miss_rate(actual, predicted):
    """Find the misclassification rate."""

    cm = make_confusion(actual, predicted, dict_=True)

    mr = {}
    for a in cm:
        mr[a] = sum([v for p, v in cm[a].items() if p != a]) / sum(cm[a].values())
        mr[a] = round(mr[a] * 100, 2)

    return mr
