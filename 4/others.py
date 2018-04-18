import numpy as np

from xgboost import XGBClassifier

from common import load_data, write_csv, accuracy

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


def xgb():
    gbm = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    gbm.fit(trX, trYi)

    # Find training accuracy
    trP = classes[gbm.predict(trX)]
    print("Training Accuracy: ", 100 * accuracy(trY, trP))

    # Dump test labels
    tsP = classes[gbm.predict(tsX)]
    write_csv("xgb.csv", tsP)
