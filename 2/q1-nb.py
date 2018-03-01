"""IMDB Movie review classification using Naive Bayes."""

# Python's stdlib
import math
import re

from collections import Counter, defaultdict

alnum = re.compile(r'[^A-Za-z]+')


# Data directory
DATA = "data/imdb/"


def clean_line(l):
    l = l.strip().lower().replace("<br />", "")
    return re.sub(alnum, ' ', l).strip()


def read_file(fn, clean=lambda l: l.strip()):
    with open(DATA + fn) as f:
        # NOTE: Could remove the list here!
        return list(map(clean, f.readlines()))


def read_data(typ):
    train_x = read_file("imdb_%s_text.txt" % typ, clean_line)
    train_y = read_file("imdb_%s_labels.txt" % typ)
    return train_x, train_y


def calc_priors(labels):
    """Calculate the prior probabilities from a list of labels."""

    cnts = Counter(labels)
    total = sum(cnts.values())
    return {cls: cnt / total for cls, cnt in cnts.items()}


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


def accuracy(reviews, labels, *args):
    """Find accuracy of the model."""

    # TODO: Build a confusion matrix

    right, wrong = 0, 0

    for r, c in zip(reviews, labels):

        if c == classify(r, *args):
            right += 1
        else:
            wrong += 1

    return right / (right + wrong)


def train():
    """Process the training data and return parameters of the model."""

    train_x, train_y = read_data("train")

    # Prior probabilites of the data
    priors = calc_priors(list(train_y))

    # Store counts of words in documents of each class
    wrd_cnt = defaultdict(Counter)

    for review, rating in zip(train_x, train_y):
        wrd_cnt[rating].update(review.split())

    wrd_cnt_tot = {cls: sum(ctr.values()) for cls, ctr in wrd_cnt.items()}

    # Build a vocabulary of all words in the dataset
    vocab = set()
    for ctr in wrd_cnt.values():
        vocab |= set(ctr.keys())

    len_vocab = len(vocab)

    # Why keep a huge list in memory
    del vocab

    return train_x, train_y, (priors, wrd_cnt, wrd_cnt_tot, len_vocab)


if __name__ == '__main__':

    train_x, train_y, args = train()

    acc = accuracy(train_x, train_y, *args)

    print("Accuracy: %f" % acc)
