import numpy as np

from .base import BaseScorer


class EuclideanScorer(BaseScorer):
    """
    Anomaly score = L2 distance from the test embedding to the mean of all
    normal training embeddings. The training mean acts as a single prototype
    of the normal class; no neighbours or clusters are needed at test time.
    """

    name = "euclidean"

    def fit(self, train_embeddings: np.ndarray) -> None:
        if train_embeddings.ndim == 3:
            N, P, D = train_embeddings.shape
            train_embeddings = train_embeddings.reshape(-1, D)
        self._mean = train_embeddings.mean(axis=0)  # (D,)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        return np.linalg.norm(test_embeddings - self._mean, axis=1)

    def score_patches(
        self,
        train_patches: np.ndarray,  # (N_train, n_patches, D) — unused, mean computed in fit
        test_patches: np.ndarray,   # (N_test,  n_patches, D)
    ) -> np.ndarray:                # (N_test,  n_patches)
        # self._mean (D,) broadcasts across (N_test, n_patches, D)
        return np.linalg.norm(test_patches - self._mean, axis=2)