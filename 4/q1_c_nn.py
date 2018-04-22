import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from common import load_data, accuracy, write_csv


# Hyper Parameters
hidden_size = 500
max_epochs = 100
batch_size = 100
learning_rate = 0.001

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


class Sketches(Dataset):

    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y = -1

        if self.Y is not None:
            y = self.Y[idx]

        return x, y


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = F.sigmoid(x)
        x = self.l2(x)
        return x


def train(net, train_data, dev_data=None):

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
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

        print(
            "\r",
            "Epoch [%d/%d]" % (epoch + 1, max_epochs),
            "| Train Loss: %.4f" % loss.data[0],
            "| Train Acc: %.4f" % predict(net, train_data, return_acc=True),
            "| Dev Acc: %.4f" % predict(net, dev_data, return_acc=True),
            sep=" ",
            end="",
        )

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


if __name__ == '__main__':

    trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)

    trD = DataLoader(Sketches(trX_, trY_),
                     batch_size, shuffle=True)

    tvD = DataLoader(Sketches(tvX_, tvY_),
                     batch_size, shuffle=False)

    # Build the network
    net = Net(784, hidden_size, 20)

    # Train it
    train(net, trD, tvD)

    # Turn shuffle off when computing predictions
    train_data = DataLoader(dataset=TrainingData(),
                            batch_size=batch_size, shuffle=False)

    test_data = DataLoader(dataset=TestingData(),
                           batch_size=batch_size, shuffle=False)

    trP = classes[predict(net, train_data)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    tsP = classes[predict(net, test_data)]
    write_csv("neural_net_%d.csv" % hidden_size, tsP)
