#!/home/dufferzafar/.apps/anaconda3/bin/python

# TODO: Fix this python
# !/usr/bin/env python3

"""
Run assignment's Python code based on input parameters.
"""

import sys
import pickle

from q1_nb import clean_line, classify, NBmodel  # noqa

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
