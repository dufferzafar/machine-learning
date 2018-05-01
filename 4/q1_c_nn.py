import numpy as np

from sklearn.model_selection import (
    ParameterGrid,
    train_test_split,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from common import load_data, write_csv, normalize


trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)

# zero mean, unit variance
trX = normalize(trX)
tsX = normalize(tsX)


class Sketches(Dataset):

    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = -1

        if self.Y is not None:
            y = self.Y[idx]

        return x, y


class Net(nn.Module):
    def __init__(self, hidden_size=50):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, hidden_size)
        self.l2 = nn.Linear(hidden_size, 20)

    def forward(self, x):
        x = self.l1(x)
        x = F.sigmoid(x)

        x = self.l2(x)
        x = F.log_softmax(x, dim=0)

        return x


def train(net, train_data, dev_data=None,
          max_epochs=100, learning_rate=0.001, quiet=False):

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(max_epochs):
        for i, (images, labels) in enumerate(train_data):

            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28 * 28).float())
            labels = Variable(labels)

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward Propagation
            outputs = net(images)

            # Calculate loss at output layer
            loss = criterion(outputs, labels)

            # Backpropagate the loss
            loss.backward()

            # Update weights
            optimizer.step()

        if not quiet:
            print(
                "\r",
                "Epoch [%3d/%3d]" % (epoch + 1, max_epochs),
                "| Train Loss: %.4f" % loss.data[0],
                "| Train Acc: %.4f" % predict(net, train_data, return_acc=True),
                "| Dev Acc: %.4f" % predict(net, dev_data, return_acc=True),
                sep=" ",
                end="",
                flush=True,
            )

        # if not quiet and not (epoch + 1) % 10:
        #     print("\n")

    if not quiet:
        print("\n")


def predict(net, data, return_acc=False):

    predictions = []
    correct, total = 0, 0

    for images, labels in data:
        images = Variable(images.view(-1, 28 * 28).float())
        outputs = net(images)

        _, batch_predictions = torch.max(outputs.data, 1)

        predictions.extend(list(batch_predictions))

        total += labels.size(0)
        correct += (batch_predictions == labels).sum()

    if return_acc:
        return correct / total
    else:
        return predictions


def simple_run(split=True):

    # Hyper Parameters
    hidden_size = 1000
    max_epochs = 30
    learning_rate = 0.0005

    batch_size = 100

    # Data
    if split:
        trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)
        trD = DataLoader(Sketches(trX_, trY_), batch_size, shuffle=True)
        tvD = DataLoader(Sketches(tvX_, tvY_), batch_size, shuffle=False)
    else:
        trD = DataLoader(Sketches(trX, trYi), batch_size, shuffle=True)
        tvD = None

    # Build the network
    net = Net(hidden_size)

    print(
        "\n",
        "Hyperparameters:",
        "hidden_size: ", hidden_size,
        "max_epochs: ", max_epochs,
        "learning_rate: ", learning_rate,
        "batch_size: ", batch_size,
        "\n",
    )

    print(net)

    # Train it
    train(net, trD, tvD, max_epochs, learning_rate)

    # Turn shuffle off when computing predictions
    tsD = DataLoader(Sketches(tsX), batch_size, shuffle=False)
    tsP = classes[predict(net, tsD)]
    write_csv("neural_net_%d.csv" % hidden_size, tsP)


def grid_search():

    trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)

    # Data
    batch_size = 100

    trD = DataLoader(Sketches(trX_, trY_),
                     batch_size, shuffle=True)

    tvD = DataLoader(Sketches(tvX_, tvY_),
                     batch_size, shuffle=False)

    parameters = {
        'hidden_size': range(50, 100, 10),
        'max_epochs': range(25, 50, 5),
        'learning_rate': [0.01, 0.005, 0.001],
    }

    results = {}
    for params in ParameterGrid(parameters):

        print("Training net", params)

        # Build the network
        net = Net(params.pop("hidden_size"))

        # Train it
        train(net, trD, tvD, **params, quiet=True)

        # Store
        dev_score = predict(net, tvD, return_acc=True)
        results[dev_score] = params


if __name__ == '__main__':

    simple_run(split=True)

    # grid_search()
