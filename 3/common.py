import numpy as np


def accuracy(actual, predicted):
    """Find accuracy of the model."""

    assert len(actual) == len(predicted)

    correct = sum(a == p for (a, p) in zip(actual, predicted))
    return correct / len(actual)


def normalize(data):
    """Scale down features to range 0-1, for faster convergence."""
    X = np.asarray(data)

    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        X = (X - mn) / (mx - mn)
        X = np.nan_to_num(X, copy=False)

    return X
