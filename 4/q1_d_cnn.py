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


class ConvNet(nn.Module):

    def __init__(self, channels=32, kernel_size=5, hidden_size=512):
        super(ConvNet, self).__init__()

        self.conv = nn.Conv2d(1, channels, kernel_size=kernel_size)
        self.mp = nn.MaxPool2d(kernel_size=2)

        # Reduce image size by 1/4th
        img_width = (28 + 1 - kernel_size) // 2
        num_inputs = channels * (img_width ** 2)

        # A hidden layer in between
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 20)

    def forward(self, x):
        in_size = x.size(0)

        # Convolution & Maxpool
        x = self.conv(x)
        x = self.mp(x)
        x = F.relu(x)

        # Flatten the input now
        x = x.view(in_size, -1)

        # Fully Connected Hidden Layer
        x = self.fc1(x)
        x = F.relu(x)

        # Output Layer
        x = self.fc2(x)
        x = F.log_softmax(x, dim=0)

        return x


def train(net, train_data, dev_data=None,
          max_epochs=100, learning_rate=0.001, quiet=False):

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(max_epochs):
        for i, (images, labels) in enumerate(train_data):

            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 1, 28, 28).float())
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
                # "\r",
                "Epoch [%3d/%3d]" % (epoch + 1, max_epochs),
                "| Train Loss: %.4f" % loss.data[0],
                "| Train Acc: %.4f" % predict(
                    net, train_data, return_acc=True),
                "| Dev Acc: %.4f" % predict(net, dev_data, return_acc=True),
                # sep=" ",
                # end="",
                flush=True,
            )

        # if not quiet and not (epoch + 1) % 10:
        #     print("\n")

    if not quiet:
        print("\n")


def predict(net, data, return_acc=False):

    if not data:
        return -1

    predictions = []
    correct, total = 0, 0

    for images, labels in data:
        images = Variable(images.view(-1, 1, 28, 28).float())
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

    # Network parameters
    channels = 64
    kernel_size = 5
    hidden_size = 512

    # Training parameters
    max_epochs = 50
    batch_size = 512
    learning_rate = 0.001

    # Data
    if split:
        trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)
        trD = DataLoader(Sketches(trX_, trY_), batch_size, shuffle=True)
        tvD = DataLoader(Sketches(tvX_, tvY_), batch_size, shuffle=False)
    else:
        trD = DataLoader(Sketches(trX, trYi), batch_size, shuffle=True)
        tvD = None

    # Build the network
    net = ConvNet(channels, kernel_size, hidden_size)

    print(
        "\n",
        "Hyperparameters:",
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
    write_csv("conv_net.csv", tsP)


def grid_search():

    trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)

    # Data
    batch_size = 100

    trD = DataLoader(Sketches(trX_, trY_),
                     batch_size, shuffle=True)

    tvD = DataLoader(Sketches(tvX_, tvY_),
                     batch_size, shuffle=False)

    parameters = {
        'channels': [10, 20, 32, 48, 64],
        'kernel_size': [4, 5, 6, 7, 8],
        'hidden_size': [256, 512, 1024],
        'learning_rate': [0.005, 0.001],
        'max_epochs': [75],
    }

    results = {}
    for params in ParameterGrid(parameters):

        print("Training net", params)

        # Build the network
        net = ConvNet(
            params.pop("channels"),
            params.pop("kernel_size"),
            params.pop("hidden_size"),
        )

        print(net)

        # Train it
        train(net, trD, tvD, **params, quiet=True)

        # Store
        dev_score = predict(net, tvD, return_acc=True)
        results[dev_score] = params


if __name__ == '__main__':

    simple_run(split=True)

    # simple_run(split=False)

    # grid_search()
