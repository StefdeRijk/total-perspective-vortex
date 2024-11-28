import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters = None
        self.n_classes = None
        self.mean = None
        self.std = None

    def calculate_covariance(self, X, y):
        _, n_channels, _ = X.shape
        covs = []

        for l in self.n_classes:
            lX = X[np.where(y == l)]
            lX = lX.transpose([1, 0, 2])
            lX = lX.reshape(n_channels, -1)
            covs.append(np.cov(lX))

        return np.asarray(covs)

    @staticmethod
    def calculate_eigenvalues(covs):
        eigenvalues, eigenvectors = [], []

        for idx, cov in enumerate(covs):
            for iidx, compCov in enumerate(covs):
                if idx < iidx:
                    eigVals, eigVects = linalg.eig(cov, cov + compCov)
                    sorted_indices = np.argsort(np.abs(eigVals - 0.5))[::-1]
                    eigenvalues.append(eigVals[sorted_indices])
                    eigenvectors.append(eigVects[:, sorted_indices])

        return eigenvalues, eigenvectors

    def pick_filters(self, eigenvectors):
        filters = []

        for EigVects in eigenvectors:
            if filters == []:
                filters = EigVects[:, :self.n_components]
            else:
                filters = np.concatenate([filters, EigVects[:, :self.n_components]], axis=1)

        self.filters = filters.T

    def fit(self, X, y):
        self.n_classes = np.unique(y)

        if len(self.n_classes) < 2:
            raise ValueError("n_classes must be >= 2")

        covs = self.calculate_covariance(X, y)

        eigenvalues, eigenvectors = self.calculate_eigenvalues(covs)

        self.pick_filters(eigenvectors)

        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])

        X = (X ** 2).mean(axis=2)

        X -= self.mean
        X /= self.std

        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
