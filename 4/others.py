import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from common import load_data, write_csv, accuracy

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)

print(len(tsX))


def xgb():
    print("Training an XGB Classifier")

    params = {
        "max_depth": 5,
        "n_setimators": 150,
    }

    trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)

    gbm = XGBClassifier(**params)
    gbm.fit(trX_, trY_, eval_set=[(tvX_, tvY_)], verbose=True)

    # Find training accuracy
    trP = classes[gbm.predict(trX)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    # Dump test labels
    tsP = classes[gbm.predict(tsX)]
    write_csv("xgb_d5_n150.csv", tsP)


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
