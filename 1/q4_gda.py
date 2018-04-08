"""Q4: Gaussian Discriminant Analysis."""

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

# Used for legends
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from common import normalize, save

# Use read_data once normalize is fixed?
xf, yf = "data/q4x.dat", "data/q4y.dat"
X = np.loadtxt(xf)
m, n = X.shape

x0 = normalize(X[:, 0])
x1 = normalize(X[:, 1])
X = np.c_[x0, x1]

y = np.loadtxt(yf, dtype='str')

# Data of a specific class
df = np.c_[X, y]
X_alaska = X[np.where(df[:, 2] == "Alaska")]
X_canada = X[np.where(df[:, 2] == "Canada")]

colors = ["r" if cls == "Canada" else "b" for cls in y]


def part_a():
    phi = len(X_alaska) / m

    mu = [np.mean(X_alaska, axis=0),
          np.mean(X_canada, axis=0)]

    # Iterative way to find sigma
    # sigma = np.zeros(shape=(n, n))
    # for i in range(m):
    #     mu_i = mu[classes.index(y[i])]
    #     sigma += np.outer((X[i] - mu_i), (X[i] - mu_i))
    # sigma = sigma / m

    # Vectorized way ftw!
    X = np.concatenate(
        (X_alaska - mu[0], X_canada - mu[1])
    )
    sigma = X.T @ X / m

    print("\n Part A \n")
    print("Phi =", phi)
    print("Mu_0 =", mu[0])
    print("Mu_1 =", mu[1])
    print("Sigma =", sigma)

    return phi, mu, sigma


def part_b():
    """Plot data with proper colors."""

    plt.scatter(X[:, 0], X[:, 1], c=colors)

    cls0 = mpatches.Patch(color='blue', label="Alaska")
    cls1 = mpatches.Patch(color='red', label="Canada")

    plt.xlabel(r'Feature 0 ($X_0)$')
    plt.ylabel(r'Feature 1 ($X_1)$')
    plt.legend(handles=[cls0, cls1])

    save(plt, "q4_b.png")
    plt.show()


def part_c(phi, mu, sigma):
    """Plot the linear decision boundary."""

    sigma_i = LA.inv(sigma)

    # Parameters of the line equation: AX - B = 0
    A = 2 * (mu[0] - mu[1]).T @ sigma_i

    B = (
        (mu[0].T @ sigma_i @ mu[0]) -
        (mu[1].T @ sigma_i @ mu[1]) -
        2 * np.log((1 / phi) - 1)
    )

    # Plot data points
    X0, X1 = X[:, 0], X[:, 1]
    plt.scatter(X0, X1, c=colors)

    # Plot line
    X1 = (B - A[0] * X0) / (A[1])
    line, = plt.plot(X0, X1, "g", label="Decision Boundary")

    cls0 = mpatches.Patch(color='blue', label="Alaska")
    cls1 = mpatches.Patch(color='red', label="Canada")

    plt.xlabel(r'Feature 0 ($X_0)$')
    plt.ylabel(r'Feature 1 ($X_1)$')
    plt.legend(handles=[cls0, cls1, line])

    save(plt, "q4_c.png")

    plt.show()

    return (X0, X1)

    # Alternative way of plotting a line using a mesh

    # p, q = np.mgrid[-2.5:2.5:50j, -3:3:50j]
    # M = np.c_[p.flatten(), q.flatten()]
    # line = (A @ M.T + B).reshape(p.shape)

    # Plot data points
    # X0, X1 = X[:, 0], X[:, 1]
    # plt.scatter(X0, X1, c=colors)
    # plt.contour(p, q, line, [0])  # [0] ensures only first contour
    # plt.show()


def part_d():
    """GDA with different covariance matrices."""

    phi = len(X_alaska) / m

    mu = [X_alaska.mean(axis=0),
          X_canada.mean(axis=0)]

    sigma = [
        # The outer product!
        (X_alaska - mu[0]).T @ (X_alaska - mu[0]) / len(X_alaska),
        (X_canada - mu[1]).T @ (X_canada - mu[1]) / len(X_canada)
    ]

    print("\n Part D \n")
    print("Phi =", phi)
    print("Mu_0 =", mu[0])
    print("Mu_1 =", mu[1])
    print("Sigma_0 =", sigma[0])
    print("Sigma_1 =", sigma[1])

    return phi, mu, sigma


def part_e(phi, mu, sigma, line):
    """Decision Boundar for GDA with different covariance matrices."""

    # Inverse and Determinant of sigma_0/1
    sigma_i = [LA.inv(sig) for sig in sigma]
    sigma_d = [LA.det(sig) for sig in sigma]

    # Parameters for equation of the decision boundary: X'AX + BX + C = 0
    # Compare these with the straight line equations
    A = (sigma_i[0] - sigma_i[1])
    B = -2 * ((mu[0].T @ sigma_i[0]) - (mu[1].T @ sigma_i[1]))
    C = (
        (mu[0].T @ sigma_i[0] @ mu[0]) -
        (mu[1].T @ sigma_i[1] @ mu[1]) -
        2 * np.log(((1 / phi) - 1) * (sigma_d[1] / sigma_d[0]))
    )

    # Mesh of points
    p, q = np.mgrid[-2.5:2.5:50j, -3:3:50j]
    M = np.c_[p.flatten(), q.flatten()]

    def bdry(x):
        return x.T @ A @ x + B @ x + C

    # The quadratic decision boundary
    quad = np.array([bdry(m) for m in M]).reshape(p.shape)

    # Plot data points
    X0, X1 = X[:, 0], X[:, 1]
    plt.scatter(X0, X1, c=colors)

    plt.contour(p, q, quad, [0], colors="y")
    line, = plt.plot(line[0], line[1], color="g", label="Linear Decision Boundary")

    # If line were calculated via vectorized method!
    # plt.contour(p, q, line, [0], colors="g")

    cls0 = mpatches.Patch(color='blue', label="Alaska")
    cls1 = mpatches.Patch(color='red', label="Canada")
    qdb = mlines.Line2D(color='yellow', label="Quadratic Decision Boundary",
                        xdata=[], ydata=[])

    plt.xlabel(r'Feature 0 ($X_0)$')
    plt.ylabel(r'Feature 1 ($X_1)$')

    plt.legend(handles=[cls0, cls1, qdb, line])

    save(plt, "q4_d.png")
    plt.show()


if __name__ == '__main__':
    phi, mu, sigma = part_a()

    part_b()

    line = part_c(phi, mu, sigma)

    phi, mu, sigma = part_d()

    part_e(phi, mu, sigma, line)
