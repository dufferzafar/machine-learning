import sys

from collections import Counter

from common import write_csv

# Various algorithms and their scores on the test set
files = {
    "keras_vgg_13_918.csv": 0.918,
    "keras_vgg_13_91867.csv": 0.91867,
    "keras_vgg_19_small_912.csv": 0.91242,
    "keras_alexnet.csv": 0.90270,  # deeper_cnn
    "keras_vgg_13_909.csv": 0.909,
    # "majority_keras_mnist_deeper_cnn_svm.csv": 0.9034,
    # "keras_vgg_13_899.csv": 0.899,
    # "keras_mnist_cnn.csv": 0.87797,
    # "conv_net_854.csv": 0.85422,
    # "pca_200_svm_rbf.csv": 0.83005,
    # "keras_vgg_13_898.csv": 0.898,

    # "conv_net_0.7_845.csv": 0.84555,
    # "neural_net_500.csv": 0.78410,
    # "logreg.csv": 0.63092,
    # "kmeans_20.csv": 0.35220,
    # "pca_50_svm_rbf_C1.csv": 0.81892,
    # "pca_50_svm_linear.csv": 0.69345,
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

# def correlations():


if __name__ == '__main__':
    labels = list(soft_file_vote(files))
    write_csv(sys.argv[1], labels)
