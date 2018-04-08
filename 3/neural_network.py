"""
A neural network implementation trained using mini-batch gradient descent.

https://en.wikipedia.org/wiki/Backpropagation
"""

import sys

import numpy as np

from common import accuracy


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def one_hot_encode(cls, num_classes):
    """Encode a class as a 1-hot vector."""
    y = np.zeros((num_classes, 1))
    y[cls] = 1.0
    return y


class Sigmoid:

    """Sigmoid function."""

    @staticmethod
    def f(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def df(x):
        return x * (1 - x)


# TODO: Make ReLU activation function
class ReLU:

    """ReLU function."""

    @staticmethod
    def f(x):
        return np.maximum(0.0, x)

    @staticmethod
    def df(x):
        return 0 if x < 0 else 1


class QuadCost:

    """Quadratic Cost."""

    @staticmethod
    def f(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def df(a, y):
        return (a - y) * Sigmoid.df(a)


class NeuralNetwork():

    def __init__(self,
                 topo=[2, 5, 2],
                 activation_func=Sigmoid):

        # Can be Sigmoid or ReLU
        self.activation_func = activation_func

        self.topo = topo
        self.nlayers = len(topo)

        # These are a neural net's parameters
        # Look into how we could maintain weights & biases in single structure
        self.weights = []
        self.biases = []

        # Parameters are initialzed by random values
        for j, k in zip(topo[:-1], topo[1:]):
            self.weights.append(np.random.randn(k, j) / np.sqrt(j))
            self.biases.append(np.random.randn(k, 1))

    def feedforward(self, a, return_lists=False):

        # Net input & output (also called activation) at each layer
        # The first layer has no input and the output is just the data as it is
        inputs = []
        outputs = [a]

        # Feed the data forward
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            a = self.activation_func.f(z)

            inputs.append(z)
            outputs.append(a)

        if return_lists:
            # The lists are needed during backprop
            return inputs, outputs
        else:
            # Last layer's activations are the actual output of the net
            return outputs[-1]

    def backprop(self, x, y):
        """
        Compute updated gradients at each layer using Backpropagation.
        """

        # Wikipedia calls these Net_j and O_j
        inputs, outputs = self.feedforward(x, return_lists=True)

        # Create empty lists to hold gradient updates
        dw = [0] * (self.nlayers - 1)
        db = [0] * (self.nlayers - 1)

        # Compute the updates for other layers - moving backwards
        for L in range(1, self.nlayers):

            # del_Oj / del_Netj = Oj (1 - Oj)
            del_out = self.activation_func.df(outputs[-L])

            # At the last layer
            if -L == -1:
                delta = del_out * (outputs[-1] - y)
            else:
                delta = del_out * (self.weights[-L + 1].T @ delta)

            # Gradient updates at this layer
            dw[-L] = delta @ outputs[-L - 1].T
            db[-L] = delta

        return dw, db

    def train(self, X, y, eta=0.05, batch_size=100, epochs=10):
        """
        Train the network using mini-batch gradient descent.
        """

        # TODO: Stopping criteria based on validation set accuracy

        # Encode data to work with the net
        X = np.array([x.reshape(-1, 1) for x in X])
        idx = np.arange(len(X))

        # If last layer has more than 1 layer, then one-hot-encode the target values
        if self.topo[-1] > 1:
            y = np.array([one_hot_encode(c, self.topo[-1]) for c in y])

        sys.stdout.write("\n")

        # Assume infinite error at beginning
        epoch = 0
        error = np.inf

        # Go over the data these many times
        while True:

            epoch += 1
            sys.stdout.write("\rEpoch: %d / %d; " % (epoch, epochs))

            # Adpative learning rate
            if eta == 0:
                eta = 1 / np.sqrt(epoch)

            # shuffle the indices of the data at each epoch
            np.random.shuffle(idx)

            # Iterate over batches of data
            for i in range(0, len(X), batch_size):

                batch = idx[i:i + batch_size]
                Xb, yb = X[batch], y[batch]

                gradients = [self.backprop(x, y) for x, y in zip(Xb, yb)]
                dw = list(sum(np.array(w) for w, _ in gradients))
                db = list(sum(np.array(b) for _, b in gradients))

                # Update the parameters of each layer
                for l in range(self.nlayers - 1):
                    self.weights[l] -= (eta / len(Xb)) * dw[l]
                    self.biases[l] -= (eta / len(Xb)) * db[l]

            error_old = error
            error = self.total_error(X, y)

            sys.stdout.write("Error: %.5f" % error)

            # Early stopping
            if abs(error_old - error) <= 10**-5:
                print("\nStopping criteria reached")
                break

            elif epoch == epochs:
                print("\nMaximum epochs reached")
                break

        sys.stdout.write("\n\n")

    def total_error(self, X, Y):
        """Average error"""
        return sum(QuadCost.f(self.feedforward(x), y) for x, y in zip(X, Y)) / len(X)

    def score(self, X, y):
        """Calculate accuracy of this net on data."""
        return accuracy(y, self.predict(X))

    def predict(self, X):
        """Predict classes for data."""

        # If there is a single output unit then use thresholding
        if self.topo[-1] == 1:
            return np.array([int(self.feedforward(x.reshape(-1, 1)) > 0.5) for x in X])
        else:
            return np.array([self.feedforward(x.reshape(-1, 1)).argmax() for x in X])