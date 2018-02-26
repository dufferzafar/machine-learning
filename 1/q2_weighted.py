"""Q2: Locally weighted linear regression."""

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from common import read_data, save


X, y, m, n = read_data("data/weightedX.csv",
                       "data/weightedY.csv")

# Find some query points in the range of x
s = np.amin(X[:, 1])
e = np.amax(X[:, 1])
query_points = np.linspace(s, e)


def part_a():
    theta = LA.inv(X.T @ X) @ X.T @ y

    print("Parameters:", theta)

    yy = list(map(lambda x: theta @ x, X))

    data, = plt.plot(X[:, 1], y, 'rx', label="Data")
    line, = plt.plot(X[:, 1], yy, 'b', label="Hypothesis (unweighted)")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(handles=[data, line])

    save(plt, "q2_a.png")

    plt.show()


def weight_matrix(X, x, tau):
    """Calculate the weight matrix for a given query point x."""
    return np.diag(np.exp(-1 * ((x - X[:, 1])**2 / (2 * tau ** 2))))


def weighted_regression(tau):
    """Run Weighted regression on the query points."""

    result = []
    for x in query_points:
        W = weight_matrix(X, x, tau)
        theta = LA.inv(X.T @ W @ X) @ X.T @ W @ y
        result.append(theta @ np.array([1, x]))

    # Plot it!
    data, = plt.plot(X[:, 1], y, 'rx', label="Data")

    taut = r'$\tau = $' + str(tau)
    line, = plt.plot(query_points, result, 'b', label="Hypothesis for " + taut)

    plt.legend(handles=[data, line], loc=4)

    plt.title(taut)

    plt.xlabel("X")
    plt.ylabel("Y")

    return plt


def part_b():
    plt = weighted_regression(0.8)
    save(plt, "q2_b_0.8.png")
    plt.show()


def part_c():
    for tau in [0.1, 0.3, 2, 10]:
        print("Tau: ", tau)

        plt = weighted_regression(tau)
        save(plt, "q2_b_%.1f.png" % tau)
        plt.close()
        plt.show()


if __name__ == '__main__':
    part_a()
    part_b()
    part_c()
