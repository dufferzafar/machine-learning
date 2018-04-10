"""
A neural network implementation trained using mini-batch gradient descent.
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


class ReLU:

    """ReLU function."""

    @staticmethod
    def f(x):
        # Return x, where x > 0
        return x * (x > 0)

    @staticmethod
    def df(x):
        # Return 1, where x > 0
        return 1 * (x > 0)


class QuadCost:

    """Quadratic Cost."""

    @staticmethod
    def f(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def df(a, y):
        return (a - y) * Sigmoid.df(a)


class NeuralNetwork():

    def __init__(self, topo, use_relu=False):

        print()
        print("NN Architecture: ", " - ".join(map(str, topo)))

        if use_relu:
            print("Using ReLU activation at hidden layers.")

        self.use_relu = use_relu

        self.topo = topo
        self.nlayers = len(topo)

        # These are a neural net's parameters
        # Look into how we could maintain weights & biases in single structure
        self.weights = []

        # In my testing:
        # things were working just fine even without having any Biases at all.
        self.biases = []

        # Parameters are initialzed by random values
        for j, k in zip(topo[:-1], topo[1:]):
            self.weights.append(np.random.randn(k, j) / np.sqrt(j))
            self.biases.append(np.random.randn(k, 1))

    def feed_forward(self, a, return_list=False):
        """Feed data forward and return output(s) of all layers or just the last one."""

        # Output (also called activation) at each layer (net inputs are not really required)
        # The first layer has no input and the output is just the data as it is
        outputs = [a]

        # Feed the data forward
        for j in range(self.nlayers - 1):

            w = self.weights[j]
            b = self.biases[j]

            net_j = w @ a + b

            # ReLU is only used in hidden layers
            if self.use_relu and j < self.nlayers - 2:
                a = ReLU.f(net_j)
            else:
                a = Sigmoid.f(net_j)

            outputs.append(a)

        if return_list:
            # This list is needed during backprop
            return outputs
        else:
            # Last layer's activations are the actual output of the net
            return outputs[-1]

    def back_propagation(self, outputs, target):
        """
        Compute gradients at each layer using Backpropagation.

        Only requires the outputs at each layer.

        https://en.wikipedia.org/wiki/Backpropagation
        """

        # Create empty lists to hold gradients
        dw = [0] * (self.nlayers - 1)
        db = [0] * (self.nlayers - 1)

        # Computation is done moving backwards
        for j in range(1, self.nlayers):

            # ReLU is only used in hidden layers
            if self.use_relu and -j != -1:
                del_out = ReLU.df(outputs[-j])
            else:
                # del_Oj / del_Netj = Oj (1 - Oj)
                del_out = Sigmoid.df(outputs[-j])

            # At the last layer
            if -j == -1:
                delta = del_out * (outputs[-1] - target)
            else:
                delta = del_out * (self.weights[-j + 1].T @ delta)

            # Gradients at this layer
            dw[-j] = delta @ outputs[-j - 1].T
            db[-j] = delta

        return dw, db

    def train(self, X, y, eta=0.05, batch_size=100, epochs=10, error_threshold=10**-6):
        """
        Train the network using mini-batch gradient descent.
        """

        # TODO: Stopping criteria based on validation set accuracy

        # Encode data to work with the net
        X = np.array([x.reshape(-1, 1) for x in X])
        idx = np.arange(len(X))

        if not batch_size:
            batch_size = len(X)

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

                gradient_updates = []

                # Go over each example and compute gradients for them
                for xb, yb in zip(Xb, yb):
                    layer_outputs = self.feed_forward(xb, return_list=True)
                    dw, db = self.back_propagation(layer_outputs, target=yb)

                    gradient_updates.append((dw, db))

                # Mean of the gradients of this batch
                dw = list(sum(np.array(w) for w, _ in gradient_updates) / len(Xb))
                db = list(sum(np.array(b) for _, b in gradient_updates) / len(Xb))

                # Update the parameters of each layer - gradient descent step!
                for l in range(self.nlayers - 1):
                    self.weights[l] -= eta * dw[l]
                    self.biases[l] -= eta * db[l]

            error_old = error
            error = self.total_error(X, y)

            sys.stdout.write("Error: %.5f" % error)

            # Early stopping
            if abs(error_old - error) <= error_threshold:
                print("\nStopping criteria reached")
                break

            elif epoch == epochs:
                print("\nMaximum epochs reached")
                break

        sys.stdout.write("\n\n")

    def total_error(self, X, Y):
        """Average error"""
        return sum(QuadCost.f(self.feed_forward(x), y) for x, y in zip(X, Y)) / len(X)

    def score(self, X, y):
        """Calculate accuracy of this net on data."""
        return accuracy(y, self.predict(X))

    def predict(self, X):
        """Predict classes for data."""

        # If there is a single output unit then use thresholding
        if self.topo[-1] == 1:
            return np.array([int(self.feed_forward(x.reshape(-1, 1)) > 0.5) for x in X])
        else:
            return np.array([self.feed_forward(x.reshape(-1, 1)).argmax() for x in X])
