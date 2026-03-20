import numpy as np

from .base import BaseScorer


class MahalanobisScorer(BaseScorer):
    """
    Anomaly score = Mahalanobis distance from the normal class distribution.
    Models correlations between feature dimensions using the empirical covariance.
    Covariance is regularized for numerical stability.
    """

    name = "mahalanobis"

    def __init__(self, reg: float = 1e-5):
        self.reg = reg

    def fit(self, train_embeddings: np.ndarray) -> None:
        self._mean = train_embeddings.mean(axis=0)
        cov = np.cov(train_embeddings, rowvar=False)
        cov += np.eye(cov.shape[0]) * self.reg
        self._inv_cov = np.linalg.pinv(cov)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        diff = test_embeddings - self._mean
        # Efficient per-sample Mahalanobis: sqrt(diff @ inv_cov @ diff.T) diagonal
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff))
