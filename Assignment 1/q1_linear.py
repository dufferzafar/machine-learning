"""Q1: Linear regression."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa

from common import read_data, save


X, y, m, n = read_data("data/linearX.csv",
                       "data/linearY.csv")


def J(theta):
    """Cost function for linear regression."""
    return np.sum((X @ theta - y) ** 2) / (2)


def part_a(eta=0.001):
    # Theta - the prameters we are looking for
    # Intialized to a vector of all zeros
    theta = np.zeros(n + 1)

    # Count the number of iterations it took for convergence
    iters = 0

    # Error at zero theta
    Jn = J(theta)
    converged = False

    # Error values that the algorithm goes through (used for plotting)
    J_values = []

    print("Learning Rate: ", eta)
    print("Initial Error: ", Jn)

    while (not converged):
        # The gradient descent update rule
        theta -= eta * (X @ theta - y) @ X

        # Calculate new error and check convergence
        Jp = Jn
        Jn = J(theta)
        J_values.append((theta, Jn))

        if abs(Jp - Jn) < 10 ** -15:
            converged = True

        iters += 1

        # To avoid infinite loops break when iterations reach a high number
        if iters == 500:
            print("\n> Early Stop \n")
            break

    print("Final Error: ", Jn)
    print("Final Parameters: ", theta)
    print("Number of iterations: ", iters)
    print("Stopping Criteria: (J_new - J_old) < 10 ** -15")

    return theta, J_values


def part_b(theta):
    data, = plt.plot(X[:, 1], y, 'rx', label="Data")
    yy = list(map(lambda x: theta @ x, X))
    line, = plt.plot(X[:, 1], yy, 'b', label="Hypothesis")

    plt.xlabel("Acidity of wine (normalised)")
    plt.ylabel("Density of wine")

    plt.legend(handles=[data, line])

    save(plt, "q1_b.png")
    plt.show()


def part_c():
    # Create a mesh using numpy awesome-sauce!
    T0, T1 = np.mgrid[-1:3:50j, -2:2.5:50j]
    mesh = np.c_[T0.flatten(), T1.flatten()]

    # Compute J_values for the grid
    J_values = (
        np.array([J(point) for point in mesh])
        .reshape(50, 50)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(T0, T1, J_values, cmap=cm.RdBu_r)
    ax.set_xlabel(r'$\theta_0$', labelpad=10)
    ax.set_ylabel(r'$\theta_1$', labelpad=10)
    ax.set_zlabel(r'$J(\theta)$', labelpad=10)

    save(plt, "q1_a.png")

    # TODO: 0.2 second time gap?
    plt.show()


def part_d():
    T0, T1 = np.mgrid[-1:11:0.1, -1:11:0.1]
    mesh = np.c_[T0.flatten(), T1.flatten()]

    J_values = (
        np.array([J(point) for point in mesh])
        .reshape(T0.shape)
    )

    # TODO: 0.2 second time gap?
    plt.contour(T0, T1, J_values, np.arange(0, 50, 3), colors="k")
    plt.contour(T0, T1, J_values, colors="k")

    return plt


def part_e():
    for eta in [0.001, 0.005, 0.009, 0.013, 0.017, 0.021, 0.025]:
        print("\n --- \n Eta: %.2f \n --- \n" % eta)

        theta, J_values = part_a(eta)

        plt = part_d()

        for t, j in J_values:
            plt.plot([t[0]], [t[1]], markerfacecolor='r',
                     marker='o', markersize=3)

        plt.show()


if __name__ == '__main__':
    theta, _ = part_a()
    part_b(theta)
    # part_c()

    # plt = part_d()
    # plt.show()

    # part_e()
