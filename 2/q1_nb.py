"""IMDB Movie review classification using Naive Bayes."""

# Python's stdlib
import math
import random
import re
import pickle

from contexttimer import Timer as TimeIt
from collections import Counter, defaultdict, namedtuple

from common import accuracy, plot_confusion

alnum = re.compile(r'[^A-Za-z]+')


# Data directory
DATA = "data/imdb/"

NBmodel = namedtuple(
    "NBmodel", ["priors", "wrd_cnt", "wrd_cnt_tot", "len_vocab"]
)


def clean_line(l):
    # l = l.strip().lower().replace("<br />", "")
    # return re.sub(alnum, ' ', l).strip()
    return l.strip().lower()


def read_file(fn, clean=lambda l: l.strip()):
    with open(DATA + fn) as f:
        return list(map(clean, f.readlines()))


def read_data(typ, stem=False):
    stem = "_stem" if stem else ""
    train_x = read_file("imdb_%s_text%s.txt" % (typ, stem), clean_line)
    train_y = read_file("imdb_%s_labels.txt" % typ)
    return train_x, train_y


def bigrams(s):
    return zip(s.split(" ")[:-1], s.split(" ")[1:])


# Data
train_x, train_y = read_data("train", stem=True)
test_x, test_y = read_data("test", stem=True)


def classify(review, m):
    """Classify a review into a class."""

    # Start with only the priors
    probs = m.priors.copy()

    # Find the probabilities of this review belonging to each class
    for cls in m.priors.keys():

        for word in review.split():

            # Count of a word may be zero for two reasons:
            # 1 - Word did not occur in reviews of that class
            # 2 - Word did not occur in the entire vocabulary
            cnt = m.wrd_cnt[cls].get(word, 0)

            # We handle both the cases similarly
            # by Laplace Smoothing
            p = (cnt + 1) / (m.wrd_cnt_tot[cls] + m.len_vocab)

            # We use summation of logs to handle underflow issues
            # with low valued probabilities
            probs[cls] += math.log(p)

        # Also consider bigrams
        # for bi in bigrams(review):
        #     cnt = m.wrd_cnt[cls].get(bi, 0)
        #     p = (cnt + 1) / (m.wrd_cnt_tot[cls] + m.len_vocab)
        #     probs[cls] += math.log(p)

    # Return the class with maximum probability
    return max(probs, key=probs.get)


def train():
    """Process the training data and return parameters of the model."""

    # Prior probabilites of the data
    cnts = Counter(train_y)
    total = sum(cnts.values())
    priors = {cls: cnt / total for cls, cnt in cnts.items()}

    # Store counts of each word in documents of each class
    term_freq = defaultdict(Counter)

    # Store the number of documents that contain this word
    doc_freq = Counter()

    for r, c in zip(train_x, train_y):
        wrds = r.split()
        term_freq[c].update(wrds)
        doc_freq.update(set(wrds))

        # bis = list(bigrams(r))
        # term_freq[c].update(bis)
        # doc_freq.update(set(bis))

    # Convert raw term frequencies into TF-IDF scores
    total_docs = len(train_x)
    for cls, ctr in term_freq.items():
        for wrd in ctr:
            ctr[wrd] = ctr[wrd] * math.log(total_docs / doc_freq[wrd])

    # Total words in documents of a class
    wrd_cnt_tot = {cls: sum(ctr.values()) for cls, ctr in term_freq.items()}

    # Build a vocabulary of all words in the dataset
    vocab = set()
    for ctr in term_freq.values():
        vocab |= set(ctr.keys())

    return NBmodel(priors, term_freq, wrd_cnt_tot, len(vocab))


def part_a():
    with TimeIt(prefix="Training Naive Bayes"):
        model = train()

    with open("models/naive-bayes-model-3", "wb") as out:
        pickle.dump(model, out)

    ratings = list(sorted(map(int, set(train_y))))

    with TimeIt(prefix="Finding Training Accuracy"):
        predicted = [classify(review, model) for review in train_x]
        train_acc = accuracy(train_y, predicted)

    print("\nTraining Accuracy: %.3f\n" % (train_acc * 100))
    plot_confusion(list(map(int, train_y)), list(map(int, predicted)), ratings, "Naive Bayes (Training Data)")

    with TimeIt(prefix="Finding Testing Accuracy"):
        predicted = [classify(review, model) for review in test_x]
        test_acc = accuracy(test_y, predicted)

    plot_confusion(list(map(int, test_y)), list(map(int, predicted)), ratings, "Naive Bayes (Testing Data)")
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


def cheating():
    """See what accuracy does sklearn give."""
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 3))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    print("Training sklearn's Naive Bayes using TF-IDF")
    clf.fit(train_x, train_y)

    predicted = clf.predict(train_x)
    train_acc = accuracy(train_y, predicted)
    print("\nTraining Accuracy: %.3f\n" % (train_acc * 100))

    predicted = clf.predict(test_x)
    test_acc = accuracy(test_y, predicted)
    print("Testing Accuracy: %.3f\n" % (test_acc * 100))


if __name__ == '__main__':

    part_a()
    # part_b()

    # cheating()
