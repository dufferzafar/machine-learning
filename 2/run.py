#!/home/dufferzafar/.apps/anaconda3/bin/python

# Fix this python
# !/usr/bin/env python3

"""
Run assignment's Python code based on input parameters.
"""

import sys
import pickle
import csv

from q1_nb import clean_line, classify, NBmodel  # noqa
from q2_svm import normalize, pegasos_predict


def svm_convert_data(inp):
    """
    Convert our data set into a format that libsvm can read.
    """
    with open(inp + "-svm-fmt", "w") as o:
        with open(inp) as f:
            for row in csv.reader(f, delimiter=','):
                o.write("0 ")
                o.write(" ".join(
                    ["%d:%d" % (i, int(x)) for i, x in enumerate(row) if int(x)])
                )
                o.write("\n")


if __name__ == '__main__':
    args = sys.argv

    ques = args[1]
    model = args[2]
    inp = args[3]
    out = args[4]

    if ques == "1":

        print("Model\t", model)
        print("Data\t", inp)
        print("Labels\t", out)

        with open(model, "rb") as m:
            model = pickle.load(m)

        with open(inp) as f:
            test_x = map(clean_line, f.readlines())

        with open(out, "w") as o:
            for review in test_x:
                o.write(classify(review, model))
                o.write("\n")

    elif ques == "2":

        # libsvm parts
        if not model.endswith("1"):
            print("First, converting input data to libsvm format")
            svm_convert_data(inp)
            exit()

        print("Pegasos!")

        print("Model\t", model)
        print("Data\t", inp)
        print("Labels\t", out)

        with open(model, "rb") as m:
            classifiers = pickle.load(m)

        with open(inp) as f:
            test_x = []
            for row in csv.reader(f, delimiter=','):
                test_x.append([int(n) for n in row])

        test_x = normalize(test_x)

        with open(out, "w") as o:
            o.write("\n".join(map(str, pegasos_predict(test_x, classifiers))))
