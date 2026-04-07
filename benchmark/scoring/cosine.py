import numpy as np

from .base import BaseScorer
from ._faiss_utils import build_index, query_index


class CosineScorer(BaseScorer):
    """
    Anomaly score = cosine distance to the nearest normal training patch.
    Uses L2-normalised vectors so cosine distance = L2 distance / 2.
    """

    name = "cosine"

    def fit(self, train_embeddings: np.ndarray) -> None:
        if train_embeddings.ndim == 3:
            N, P, D = train_embeddings.shape
            train_embeddings = train_embeddings.reshape(-1, D)
        # Index is built on normalised vectors; query vectors are normalised at search time.
        self._index = build_index(train_embeddings, normalize=True)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists = query_index(self._index, test_embeddings, k=1, normalize=True)
        return dists[:, 0]

    def score_patches(
        self,
        train_patches: np.ndarray,  # (N_train, n_patches, D) — unused, index built in fit
        test_patches: np.ndarray,   # (N_test,  n_patches, D)
    ) -> np.ndarray:                # (N_test,  n_patches)
        N_test, n_patches, D = test_patches.shape
        dists = query_index(self._index, test_patches.reshape(-1, D), k=1, normalize=True)
        return dists[:, 0].reshape(N_test, n_patches)