import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip


def plot_iris(X_transformed, y, y_names, *, title):
    scatter = plt.scatter(*X_transformed.T, c=y)
    scatter_objects, _ = scatter.legend_elements()
    plt.legend(scatter_objects, y_names, loc='lower left', title='Classes')
    plt.title(title)
    plt.show()


def pca_through_covariance_matrix(X, n_components=2):
    X = X.copy()
    n_samples, n_features = X.shape

    # center data - i.e. subtract off the mean for each dimension
    X -= np.mean(X, axis=0)
    assert np.allclose(X.mean(axis=0), np.zeros(X.shape[1]))

    # compute covariance matrix
    C = (X.T @ X) / (n_samples - 1)

    # find eigenvalues and eigenvectors (in this order)
    w, v = np.linalg.eig(C)

    # project the original data set
    X_transformed = X @ v

    # get variance explained
    C_transformed = (X_transformed.T @ X_transformed) / (n_samples - 1)
    explained_variance = C_transformed.diagonal()
    explained_variance_ratio = explained_variance / explained_variance.sum()

    # select only first two principal components
    X_transformed = X_transformed[:, :n_components]

    # note: In this implementation, we using full array of explaind variance.
    #       But implementation of Scikit-Learn is croping the array
    #       and storing only explained variance of n_components.

    return X_transformed, explained_variance, explained_variance_ratio


def pca_through_svd(X, n_components=2):
    """
    Pieces of code were taken from scikit-learn library.
    See documentation page https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    X = X.copy()
    n_samples, n_features = X.shape

    # center data - i.e. subtract off the mean for each dimension
    X -= np.mean(X, axis=0)
    assert np.allclose(X.mean(axis=0), np.zeros(X.shape[1]))

    # compute singular value decomposition (SVD)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output

    # create output variables (these are present in scikit-learn PCA object after it is fitted)
    components = V[:n_components]
    singular_values = S[:n_components]
    # X_transformed = X * V = U * S * V^T * V = U * S
    X_transformed = U[:, :n_components] * S[:n_components]

    # get variance explained
    explained_variance = (S ** 2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    # note: In this implementation, we using full array of explaind variance.
    #       But implementation of Scikit-Learn is croping the array
    #       and storing only explained variance of n_components.

    return X_transformed, explained_variance, explained_variance_ratio


def pca_scikit_learn(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    return X_transformed, pca.explained_variance_, pca.explained_variance_ratio_


# %% load data
dataset = datasets.load_iris()

X = dataset.data
y = dataset.target
y_names = dataset.target_names


# %% custom pca (covariance matrix)
X_transformed_cov, explained_variance_cov, explained_variance_ratio_cov = pca_through_covariance_matrix(X)
plot_iris(X_transformed_cov, y, y_names, title='PCA custom code - computed through Covariance matrix')


# %% custom pca (with SVD) - utilized code from scikit-learn
X_transformed_svd, explained_variance_svd, explained_variance_ratio_svd = pca_through_svd(X)
plot_iris(X_transformed_svd, y, y_names, title='PCA custom code - computed through SVD')


# %% pca from scikit-learn - this implementation utilizes SVD
X_transformed_sklearn, explained_variance_sklearn, explained_variance_ratio_sklearn = pca_scikit_learn(X)
plot_iris(X_transformed_sklearn, y, y_names, title='Scikit-learn PCA')


# %% compare variables
# compare output data
print('Output of custom SVD implementation is same as output of Scikit-Learn implementation:',
      (X_transformed_svd == X_transformed_sklearn).all())

print('Output of custom Covariance Matrix implementation is same as output of custom SVD implementation:',
      np.allclose(X_transformed_cov, X_transformed_svd))

print('Output of custom Covariance Matrix implementation is rotated on y-axis'
      ' compared to output of custom SVD implementation:',
      np.allclose(X_transformed_cov * [1, -1], X_transformed_svd))

# compare explained variance
print('Explained variance of custom Covariance Matrix implementation'
      ' is same as explained variance of custom SVD implementation:',
      np.allclose(explained_variance_cov, explained_variance_svd))

print('Explained variance of custom Covariance Matrix implementation'
      ' is same as explained variance of Scikit-Learn implementation:',
      np.allclose(explained_variance_cov[:2], explained_variance_sklearn))
