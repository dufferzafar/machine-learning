"""
Find accuracy from two files that contain a label per line.
"""

import sys

from common import accuracy

if __name__ == '__main__':
    with open(sys.argv[1], "r") as f:
        actual = list(map(lambda line: line.strip(), f.readlines()))

    with open(sys.argv[2], "r") as f:
        predicted = list(map(lambda line: line.strip(), f.readlines()))

    print("Actual labels: ", sys.argv[1])
    print("Predicted labels: ", sys.argv[2])
    print("Accuracy: ", accuracy(actual, predicted) * 100, "%")
