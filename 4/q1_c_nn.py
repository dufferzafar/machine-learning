import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from common import load_data, accuracy, write_csv

# Hyper Parameters
hidden_size = 500
num_epochs = 10
batch_size = 100
learning_rate = 0.0001

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


class TrainingData(Dataset):

    def __len__(self):
        return len(trX)

    def __getitem__(self, idx):
        return trX[idx], trYi[idx]


class TestingData(Dataset):

    def __len__(self):
        return len(tsX)

    def __getitem__(self, idx):
        return tsX[idx], -1


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


def train(net, data):

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data):

            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28 * 28).float())
            labels = Variable(labels, requires_grad=False)

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
                print('Epoch [%d/%d], Batch [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, (i + 1) // batch_size, len(data) // batch_size, loss.data[0]))


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

    test_data = DataLoader(dataset=TestingData(),
                           batch_size=batch_size, shuffle=True)

    net = Net(784, hidden_size, 20)

    train(net, train_data)

    trP = classes[predict(net, train_data)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    tsP = classes[predict(net, test_data)]
    write_csv("neural_net_%d.csv" % hidden_size, tsP)
