"""Q3: Logistic Regression using Newton's Method."""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from common import normalize, save


# Use read_data once normalize is fixed?
xf, yf = "data/logisticX.csv", "data/logisticY.csv"
X = np.loadtxt(xf, delimiter=',')
m, n = X.shape

x0 = np.ones(m)
x1 = normalize(X[:, 0])
x2 = normalize(X[:, 1])

X = np.c_[x0, x1, x2]
y = np.loadtxt(yf)


def g(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    """Hypothesis function for logistic regression."""
    return g(X @ theta)


def gradient(X, y, theta):
    """Gradient of the log-likelihood function."""
    return X.T @ (y - h(X, theta))


def hessian(X, theta):
    """Hessian of the log-likelihood function."""

    # The diagonal sigmoid matrix
    # https://stats.stackexchange.com/a/316933
    # -1 * g(X, theta) * (1 - g(X, theta)) * X @ X.T
    G = h(X, theta)
    D = np.diag(G * (1 - G))

    return X.T @ D @ X


def L(X, y, theta):
    """log-likelihood of logistic regression."""
    return -1 * (np.sum(y * np.log(h(X, theta)) + (1 - y) * np.log(1 - h(X, theta))))


def part_a(theta=np.zeros(n + 1)):
    """Newton's method."""

    iters = 0
    converged = False
    theta = np.zeros(n + 1)

    Ln = L(X, y, theta)
    print("Initial Error: ", Ln)

    while not converged:
        theta += np.linalg.pinv(hessian(X, theta)) @ gradient(X, y, theta)

        Lp = Ln
        Ln = L(X, y, theta)

        if Lp - Ln < 10**-12:
            converged = True

        iters += 1

    print("Final Error: ", Ln)
    print("Number of iterations: ", iters)
    print("Parameters: ", theta)

    return theta


def part_b(theta):

    # Use vectorization to remove this and the corresponding map?
    def find_x2(x, t=theta):
        """
        Find x2 from equation of the line.

            > theta @ X = 0
            > t0 + t1 x1 + t2 x2 = 0
            > t2 x2 = -t0 -t1 x1
            > x2 = (-t0 -t1 x1) / t2
        """
        return (-t[0] - t[1] * x[1]) / t[2]

    colors = ["r" if cls else "b" for cls in y]

    yy = list(map(find_x2, X))

    plt.scatter(X[:, 1], X[:, 2], c=colors)
    line, = plt.plot(X[:, 1], yy, 'g', label="Decision Boundary")

    cls0 = mpatches.Patch(color='blue', label='Class 0')
    cls1 = mpatches.Patch(color='red', label='Class 1')

    plt.xlabel(r'Feature 0 ($X_0)$')
    plt.ylabel(r'Feature 1 ($X_1)$')

    plt.legend(handles=[cls0, cls1, line])

    save(plt, "q3_b.png")

    plt.show()


if __name__ == '__main__':
    theta = part_a()
    part_b(theta)
