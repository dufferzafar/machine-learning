import numpy as np
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from common import load_data, write_csv

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


def pca_plot():
    print("Fitting PCA")
    pca = PCA().fit(trX)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")

    plt.title("Deciding the number of components")

    # plt.show()
    plt.savefig("output/q1_b_pca.png")
    plt.close()


def pca_svm_linear():

    pipeline = Pipeline([
        # Normalize data to zero mean & unit variance
        ('scaler', StandardScaler()),

        # Project data down to 50 principal components
        ('pca', PCA(n_components=50)),

        # Learn a hyperplane in that space
        ('clf', SVC(verbose=True)),
    ])

    pipeline.fit(trX, trY)

    print("Training Accuracy: ", pipeline.score(trX, trY))

    tsP = pipeline.predict(tsX)
    write_csv("1_b_pca_svm.csv", tsP)


if __name__ == '__main__':
    pca_svm()
