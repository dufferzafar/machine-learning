
import numpy as np


class Sigmoid:

    """Sigmoid function."""

    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def df(x):
        return x * (1 - x)


# TODO: Make ReLU activation function


class NeuralNetwork():

    def __init__(self,
                 topo=[2, 3, 1],
                 activation_func=Sigmoid):

        # Can be Sigmoid or ReLU
        self.activation_func = activation_func

        # These are a neural net's parameters
        # TODO: Don't maintain weights / biases separately?
        self.weights = []
        self.biases = []

        # Parameters are initialzed by random values
        for j, k in zip(topo[:-1], topo[1:]):
            self.weights.append(np.random.randn(k, j))
            self.biases.append(np.random.randn(k))

    def feedforward(self, a, return_list=False):

        # The activation of first layer is the input as it is
        activations = []

        # Feed the data forward
        for w, b in zip(self.weights, self.biases):
            a = self.activation_func.f(w @ a.T + b)
            activations.append(a)

        # Return activations of all layers or just the last layer
        if return_list:
            return activations
        else:
            # Last layer's activation is the output of the net
            return activations[-1]

    def train(X, y, eta=0.5, batch=100):
        """
        Train the network using mini-batch gradient descent.
        """
        pass

    def score(X, y):
        """Calculate the accuracy of the net on data."""
        pass

    def predict(X):
        """Predict classes for data."""
        pass
