import numpy as np


def normalize(x):
    """Normalize a column of a numpy array."""

    mu = np.mean(x)
    std = np.std(x)

    def norm(z):
        return (z - mu) / std

    return np.vectorize(norm)(x)


def read_data(xf, yf, delimiter=','):
    """Read a file from data/ directory."""

    x = np.loadtxt(xf, delimiter=delimiter)

    try:
        m, n = x.shape[0], x.shape[1]
    except IndexError:
        m, n = (x.shape[0], 1)

    # The intercept term
    x0 = np.ones(m)

    # Normalization
    x = normalize(x)

    X = np.c_[x0, x]
    y = np.loadtxt(yf)

    return X, y, m, n
