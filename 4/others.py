import numpy as np

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from common import load_data, write_csv, accuracy

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


def xgb():
    print("Training an XGB Classifier")
    gbm = XGBClassifier()
    gbm.fit(trX, trYi)

    # Find training accuracy
    trP = classes[gbm.predict(trX)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    # Dump test labels
    tsP = classes[gbm.predict(tsX)]
    write_csv("xgb.csv", tsP)


def logreg():
    # Takes ~46 minutes to finish
    print("Training Logistic Regression")
    clf = LogisticRegression()
    clf.fit(trX, trY)

    print("Training Accuracy:", clf.score(trX, trY))

    tsP = clf.predict(tsX)
    write_csv("logreg.csv", tsP)


if __name__ == '__main__':
    xgb()
    # logreg()
    # voting_ensemble()
