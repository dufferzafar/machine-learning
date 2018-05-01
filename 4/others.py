import random

import cv2
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

from common import load_data, write_csv, accuracy

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)

print(len(tsX))


def xgb():
    print("Training an XGB Classifier")

    params = {
        "max_depth": 8,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "n_jobs": -1,
        "subsample": 0.8,
        "nthread": 4,
    }

    trX_, tvX_, trY_, tvY_ = train_test_split(trX, trYi, test_size=0.3)

    gbm = XGBClassifier(**params)
    print(gbm.get_xgb_params())

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


def clean_imgs():
    r, c = 4, 5
    f, axs = plt.subplots(2 * r, c, figsize=(12, 18), facecolor='white')

    kernel = np.ones((2, 1))
    img_idx = random.randrange(5000)

    for i in range(0, 2 * r, 2):
        for j in range(c):

            img = trX[img_idx]
            img_ = cv2.erode(img, kernel)
            # img_ = cv2.dilate(img_, kernel)

            axs[i, j].imshow(img.reshape(28, 28))
            axs[i, j].set_title(trY[img_idx])
            axs[i, j].axis('off')

            axs[i + 1, j].imshow(img_.reshape(28, 28))
            axs[i + 1, j].set_title(trY[img_idx])
            axs[i + 1, j].axis('off')

            img_idx += 5000

    f.savefig("output/cleaning_erosion.png")


if __name__ == '__main__':
    # xgb()
    # logreg()
    clean_imgs()
