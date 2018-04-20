import sys

import numpy as np

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


class TrainingData(Dataset):

    def __len__(self):
        return len(trX)

    def __getitem__(self, idx):
        return trX[idx] / 255, trYi[idx]


class TestingData(Dataset):

    def __len__(self):
        return len(tsX)

    def __getitem__(self, idx):
        return tsX[idx] / 255, -1


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


def train(net, data):

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # print(len(data), batch_size)

    for epoch in range(max_epochs):
        for i, (images, labels) in enumerate(data):

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

            if (i + 1) % 100 == 0:
                sys.stdout.write('\rEpoch [%d/%d], Batch [%d/%d], Loss: %.5f'
                                 % (epoch + 1, max_epochs, (i + 1) // 100, len(data) // 100, loss.data[0]))

    print("\n")


def predict(net, data):
    predictions = []
    for images, _ in data:
        images = Variable(images.view(-1, 28 * 28).float())
        outputs = net(images)
        _, batch_predictions = torch.max(outputs.data, 1)

        predictions.extend(list(batch_predictions))

    return predictions


if __name__ == '__main__':
    train_data = DataLoader(dataset=TrainingData(),
                            batch_size=batch_size, shuffle=True)

    net = Net(784, hidden_size, 20)

    train(net, train_data)

    # Turn shuffle off when computing predictions
    train_data = DataLoader(dataset=TrainingData(),
                            batch_size=batch_size, shuffle=False)

    test_data = DataLoader(dataset=TestingData(),
                           batch_size=batch_size, shuffle=False)

    trP = classes[predict(net, train_data)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    tsP = classes[predict(net, test_data)]
    write_csv("neural_net_%d.csv" % hidden_size, tsP)
