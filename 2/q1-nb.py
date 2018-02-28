"""IMDB Movie review classification using Naive Bayes."""

import numpy as np


# Load dataset
DATA = "data/imdb/"


def read_file(fn):
    with open(DATA + fn) as f:
        return list(map(lambda l: l.strip(), f.readlines()))


def read_data():
    train_x = read_file("imdb_train_text.txt")
    train_y = read_file("imdb_train_labels.txt")

    # test_x = read_file("imdb_test_text.txt")
    # test_y = read_file("imdb_test_labels.txt")

    return train_x, train_y
    # return train_x, train_y, test_x, test_y

print(set(train_labels))

# Calculate prior probablities of all
