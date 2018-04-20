import sys

from collections import Counter

from common import write_csv

# Various algorithms and their scores on the test set
files = {
    "kmeans_20.csv": 0.35220,
    "pca_50_svm_rbf_C1.csv": 0.81892,
    # "pca_50_svm_linear.csv": 0.69345,
    "logreg.csv": 0.63092,
    # "xgb.csv": 0.62302,
}


def soft_file_vote(files):
    # Go over lines in all files at once
    file_objects = (open("output/" + f) for f in files)
    weights = files.values()

    for idx, lines in enumerate(zip(*file_objects)):
        if not idx:  # Skip header row
            continue

        labels = map(lambda l: l.strip().split(",")[1], lines)

        c = Counter()
        for lbl, wt in zip(labels, weights):
            c.update({lbl: wt})

        label, _ = c.most_common(1)[0]

        yield label


if __name__ == '__main__':
    labels = list(soft_file_vote(files))
    write_csv(sys.argv[1], labels)
