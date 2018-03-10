"""IMDB Movie review classification using Naive Bayes."""

# Python's stdlib
import math
import random
import re

from contexttimer import Timer as TimeIt
from collections import Counter, defaultdict

from common import accuracy, plot_confusion

alnum = re.compile(r'[^A-Za-z]+')


# Data directory
DATA = "data/imdb/"


def clean_line(l):
    # l = l.strip().lower().replace("<br />", "")
    # return re.sub(alnum, ' ', l).strip()
    return l.strip().lower()


def read_file(fn, clean=lambda l: l.strip()):
    # print("reading ", fn)
    with open(DATA + fn) as f:
        # NOTE: Could remove the list here!
        return list(map(clean, f.readlines()))


def read_data(typ, stem=False):
    stem = "_stem" if stem else ""
    train_x = read_file("imdb_%s_text%s.txt" % (typ, stem), clean_line)
    train_y = read_file("imdb_%s_labels.txt" % typ)
    return train_x, train_y


# Data
train_x, train_y = read_data("train", stem=True)
test_x, test_y = read_data("test", stem=True)


def classify(review, priors, wrd_cnt, wrd_cnt_tot, len_vocab):
    """Classify a review into a class."""

    # Start with only the priors
    probs = priors.copy()

    # Find the probabilities of this review belonging to each class
    for cls in priors.keys():

        for word in review.split():

            # Count of a word may be zero for two reasons:
            # 1 - Word did not occur in reviews of that class
            # 2 - Word did not occur in the entire vocabulary
            cnt = wrd_cnt[cls].get(word, 0)

            # We handle both the cases similarly
            # by Laplace Smoothing
            p = (cnt + 1) / (wrd_cnt_tot[cls] + len_vocab)

            # We use summation of logs to handle underflow issues
            # with low valued probabilities
            probs[cls] += math.log(p)

    # Return the class with maximum probability
    return max(probs, key=probs.get)


def train():
    """Process the training data and return parameters of the model."""

    # Prior probabilites of the data
    cnts = Counter(train_y)
    total = sum(cnts.values())
    priors = {cls: cnt / total for cls, cnt in cnts.items()}

    # Store counts of each word in documents of each class
    wrd_cnt = defaultdict(Counter)
    for r, c in zip(train_x, train_y):
        wrd_cnt[c].update(r.split())

    # Total words in documents of a class
    wrd_cnt_tot = {cls: sum(ctr.values()) for cls, ctr in wrd_cnt.items()}

    # Build a vocabulary of all words in the dataset
    vocab = set()
    for ctr in wrd_cnt.values():
        vocab |= set(ctr.keys())

    return priors, wrd_cnt, wrd_cnt_tot, len(vocab)


def part_a():
    with TimeIt(prefix="Training Naive Bayes"):
        model = train()

    print("")

    ratings = list(sorted(map(int, set(train_y))))

    with TimeIt(prefix="Finding Training Accuracy"):
        predicted = [classify(review, *model) for review in train_x]
        train_acc = accuracy(train_y, predicted)

    print("\nTraining Accuracy: %.3f\n" % (train_acc * 100))
    plot_confusion(train_y, predicted, ratings, "Naive Bayes (Training Data)")

    with TimeIt(prefix="Finding Testing Accuracy"):
        predicted = [classify(review, *model) for review in test_x]
        test_acc = accuracy(test_y, predicted)

    plot_confusion(test_y, predicted, ratings, "Naive Bayes (Testing Data)")
    print("\nTesting Accuracy: %.3f\n" % (test_acc * 100))


def part_b():
    ratings = list(set(train_y))
    random_predictions = [random.choice(ratings) for _ in test_x]
    rand_acc = accuracy(test_y, random_predictions)

    print("\nRandom Accuracy: %.3f\n" % (rand_acc * 100))

    majority_class = Counter(train_y).most_common(1)[0][0]

    majority_predictions = [majority_class] * len(test_x)
    maj_acc = accuracy(test_y, majority_predictions)

    print("\nMajority Accuracy: %.3f\n" % (maj_acc * 100))


if __name__ == '__main__':

    part_a()

