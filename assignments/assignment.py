# %%
import numpy as np
from typing import Any


# TODO: implement the PCA with numpy
# Note that you are not allowed to use any existing PCA implementation from sklearn or other libraries.
class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        """_summary_

        Parameters
        ----------
        n_components : int
            The number of principal components to be computed. This value should be less than or equal to the number of features in the dataset.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    # TODO: implement the fit method
    def fit(self, X: np.ndarray):
        """
        Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # following from exercise 1
        # center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # Compute covariance matrix
        cov_mat = np.cov(X_centered.T)
        # compute eigen_values and eigen_vectors
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        # sorting eigen values and eigen vectors
        eig_index = np.argsort(eigen_vals)[::-1]
        eigen_vecs = eigen_vecs[:, eig_index]
        # select top N components
        self.components = eigen_vecs[:, : self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components)
        return X_transformed


# TODO: implement the LDA with numpy
# Note that you are not allowed to use any existing LDA implementation from sklearn or other libraries.
class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Hint:
        -----
        To implement LDA with numpy, follow these steps:
        1. Compute the mean vectors for each class.
        2. Compute the within-class scatter matrix.
        3. Compute the between-class scatter matrix.
        4. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices.
        5. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix W.
        6. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
        """
        number_of_features = X.shape[1]
        unique_labels = np.unique(y)
        num_of_class = len(unique_labels)

        # within class matix
        Sw = np.zeros((number_of_features, number_of_features))
        # between class matrix
        Sb = np.zeros((number_of_features, number_of_features))

        # overall mean computation
        mean_overall = np.mean(X, axis=0)
        self.means_ = np.zeros((num_of_class, number_of_features))

        for index, label in enumerate(unique_labels):
            X_class = X[y == label]
            mean_class = np.mean(X_class, axis=0)
            self.means_[index] = mean_class

            # within class matrix
            Sw = Sw + np.dot((X_class - mean_class).T, (X_class - mean_class))

            # Between class matrix
            n_class = X_class.shape[0]
            mean_diff = (mean_class - mean_overall).reshape(number_of_features, 1)
            Sb = Sb + n_class * np.dot(mean_diff, mean_diff.T)

        # solve the eigen value
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

        # sort eigen-vectors by eigen values in descending order
        index = np.argsort(eigvals[::-1])
        eigvecs = eigvecs[:, index]

        # select the top components eigen vectors
        self.scalings_ = eigvecs[:, : self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        X_transformed = np.dot(X, self.scalings_)

        return X_transformed


# TODO: Generating adversarial examples for PCA.
# We will generate adversarial examples for PCA. The adversarial examples are generated by creating two well-separated clusters in a 2D space. Then, we will apply PCA to the data and check if the clusters are still well-separated in the transformed space.
# Your task is to generate adversarial examples for PCA, in which
# the clusters are well-separated in the original space, but not in the PCA space. The separabilit of the clusters will be measured by the K-means clustering algorithm in the test script.
#
# Hint:
# - You can place the two clusters wherever you want in a 2D space.
# - For example, you can use `np.random.multivariate_normal` to generate the samples in a cluster. Repeat this process for both clusters and concatenate the samples to create a single dataset.
# - You can set any covariance matrix, mean, and number of samples for the clusters.
class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        """Generate adversarial examples for PCA

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_features : int
            The number of features.

        Returns
        -------
        X: ndarray of shape (n_samples, n_features)
            Transformed values.

        y: ndarray of shape (n_samples,)
            Cluster IDs. y[i] is the cluster ID of the i-th sample.

        """
        mean1 = np.zeros(n_features)
        mean2 = np.array([2] * n_features)

        # Spreading of the clusters along different directions
        cov1 = np.diag(np.linspace(1, 2, n_features))
        cov2 = np.diag(np.linspace(1, 3, n_features))

        # samples for each cluster
        cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples // 2)
        cluster2 = np.random.multivariate_normal(
            mean2, cov2, n_samples - n_samples // 2
        )

        # Combined samples
        X = np.vstack((cluster1, cluster2))
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

        return X, y


# %%
