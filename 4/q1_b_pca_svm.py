import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from common import load_data, write_csv

trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)

# Scale the data
# trX /= 255
# tsX /= 255

svm_pipeline = Pipeline([
    # Normalize data to zero mean & unit variance
    ('scaler', StandardScaler()),

    # Project data down to 50 principal components
    ('pca', PCA(n_components=50)),

    # Learn a hyperplane in that space
    ('clf', SVC(kernel="linear",
                decision_function_shape="ovo")),
])


def pca_plot():

    print("\nFitting PCA")
    pca = PCA().fit(trX)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")

    plt.title("Deciding the number of components")

    plt.savefig("output/q1_b_pca.png")
    plt.close()


def pca_svm_linear():

    print(svm_pipeline)

    print("\nFitting PCA (50) + SVM (Linear)")
    svm_pipeline.fit(trX, trY)

    print("Training Accuracy: ", svm_pipeline.score(trX, trY))

    tsP = svm_pipeline.predict(tsX)
    write_csv("pca_50_svm_linear.csv", tsP)


def pca_svm_linear_grid_search():

    print(svm_pipeline)

    print("\nRunning a grid search over C\n")

    params = dict(clf__C=[0.001, 0.01, 0.1, 1, 10, 100])

    grid_search = GridSearchCV(svm_pipeline, n_jobs=-1,
                               param_grid=params, verbose=True)

    grid_search.fit(trX, trY)

    print("Best Parameter:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    # print(pd.DataFrame(grid_search.cv_results_))


def pca_svm_rbf():

    svm_pipeline.set_params(pca__n_components=250)
    svm_pipeline.set_params(clf__kernel="rbf")

    print(svm_pipeline)

    print("\nFitting PCA (250) + SVM (RBF)")
    svm_pipeline.fit(trX, trY)

    print("Training Accuracy: ", svm_pipeline.score(trX, trY))

    tsP = svm_pipeline.predict(tsX)
    write_csv("pca_250_svm_rbf.csv", tsP)


if __name__ == '__main__':
    # pca_plot()

    pca_svm_linear()

    # pca_svm_linear_grid_search()

    # pca_svm_rbf()
