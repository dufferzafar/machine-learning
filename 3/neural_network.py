"""
A neural network implementation trained using mini-batch gradient descent.

https://en.wikipedia.org/wiki/Backpropagation
"""


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
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def df(x):
        return x * (1 - x)


# TODO: Make ReLU activation function


class NeuralNetwork():

    def __init__(self,
                 topo=[2, 3, 2],
                 activation_func=Sigmoid):

        # Can be Sigmoid or ReLU
        self.activation_func = activation_func

        self.nlayers = len(topo)

        # These are a neural net's parameters
        # TODO: Don't maintain weights / biases separately?
        self.weights = []
        self.biases = []

        # Parameters are initialzed by random values
        for j, k in zip(topo[:-1], topo[1:]):
            self.weights.append(np.random.randn(k, j))
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

        # Compute updates at the last layer
        # NOTE: This will change if the cost function changes
        # delta = (outputs[-1] - y) * self.activation_func.df(outputs[-1])
        # dw[-1] = delta @ outputs[-2].T
        # db[-1] = delta

        # Create empty lists to hold gradient updates
        dw = [0] * self.nlayers
        db = [0] * self.nlayers

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

    def train(self, X, y, eta=0.05, batch_sz=100, epochs=10):
        """
        Train the network using mini-batch gradient descent.
        """

        # TODO: Convergence using validation set accuracy

        # Encode data to work with the net
        X = [x.reshape(-1, 1) for x in X]
        y = [one_hot_encode(c, 2) for c in y]

        m, n = X.shape

        # Go over the data these many times
        for _ in range(epochs):

            # Iterate over batches of data
            for i in range(0, m, batch_sz):
                Xb, yb = X[i:i + batch_sz], y[i:i + batch_sz]

                gradients = [self.backprop(x, y) for x, y in zip(Xb, yb)]
                dw = [w for w, _ in gradients]
                db = [b for _, b in gradients]

                # Sum the gradients properly
                # dw = []
                # db = []

                # assert dw.shape == self.weights.shape

                # Update the parameters of each layer
                for l in range(self.nlayers):
                    self.weights[l] -= (eta / len(Xb)) * dw[l]
                    self.biases[l] -= (eta / len(Xb)) * db[l]

    def score(self, X, y):
        """Calculate accuracy of this net on data."""
        return accuracy(y, self.predict(X))

    def predict(self, X):
        """Predict classes for data."""
        return [self.feedforward(x).argmax() for x in X]
