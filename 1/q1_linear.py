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
    J_trace = np.array([])

    print("Learning Rate: ", eta)
    print("Initial Error: ", Jn)

    while (not converged):
        J_trace = np.append(J_trace, [theta[0], theta[1], Jn])

        # The gradient descent update rule
        theta -= eta * (X @ theta - y) @ X

        # Calculate new error and check convergence
        Jp = Jn
        Jn = J(theta)

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

    return theta, J_trace.reshape(iters, 3)


def part_b(theta):

    yy = list(map(lambda x: theta @ x, X))

    data, = plt.plot(X[:, 1], y, 'rx', label="Data")
    line, = plt.plot(X[:, 1], yy, 'b', label="Hypothesis")

    plt.xlabel("Acidity of wine (normalised)")
    plt.ylabel("Density of wine")

    plt.legend(handles=[data, line])

    save(plt, "q1_b.png")
    plt.show()


def part_c(Jt=None):
    # Create a mesh using numpy awesome-sauce!
    T0, T1 = np.mgrid[0:2:50j, -1:1:50j]
    mesh = np.c_[T0.flatten(), T1.flatten()]

    # Compute J_values for the grid
    J_values = (
        np.array([J(point) for point in mesh])
        .reshape(50, 50)
    )

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(T0, T1, J_values, cmap=cm.RdBu_r)
    ax.set_xlabel(r'$\theta_0$', labelpad=10)
    ax.set_ylabel(r'$\theta_1$', labelpad=10)
    ax.set_zlabel(r'$J(\theta)$', labelpad=10)

    plt.show()

    if Jt is not None:

        # To draw line between points
        # ax.plot(Jt[:, 0], Jt[:, 1], Jt[:, 2], linestyle='-',
        #         color='r', marker='o', markersize=2.5)

        for jt in Jt:
            ax.plot([jt[0]], [jt[1]], [jt[2]], linestyle='-',
                    color='r', marker='o', markersize=2.5)

            plt.pause(0.2)

    save(plt, "q1_a.png")


def part_d(Jt=None, show=False):
    T0, T1 = np.mgrid[0:2:50j, -1:1:50j]
    mesh = np.c_[T0.flatten(), T1.flatten()]

    J_values = (
        np.array([J(point) for point in mesh])
        .reshape(T0.shape)
    )

    plt.ion()
    plt.contour(T0, T1, J_values, 25, colors="k")
    plt.xlabel(r'$\theta_0$', labelpad=10)
    plt.ylabel(r'$\theta_1$', labelpad=10)

    if show:
        plt.show()

    if Jt is not None:

        # To draw line with points
        # plt.plot(Jt[:, 0], Jt[:, 1], linestyle='-',
        #          color='r', marker='o', markersize=2.5)

        for jt in Jt:
            plt.plot([jt[0]], [jt[1]], linestyle='-',
                     color='r', marker='o', markersize=2.5)

            if show:
                plt.pause(0.02)

    if show:
        save(plt, "q1_d_0.001.png")
        plt.close()

    return plt


def part_e():
    for eta in [0.001, 0.005, 0.009, 0.013, 0.017, 0.021, 0.025]:
        print("\n --- \n Eta: %.2f \n --- \n" % eta)

        theta, J_trace = part_a(eta)

        plt = part_d(J_trace)
        plt.title(r"$Contours (\eta=$" + str(eta) + ")")

        save(plt, "q1_e_%.3f.png" % eta)
        plt.show()
        plt.close()


if __name__ == '__main__':
    theta, J_trace = part_a(eta=0.001)
    part_b(theta)

    part_c()
    part_c(J_trace)

    part_d(J_trace, show=True)

    part_e()
