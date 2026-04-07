import numpy as np

from .base import BaseScorer
from ._faiss_utils import build_index, query_index


class KNNScorer(BaseScorer):
    name = "knn"

    def __init__(self, k: int = 5, aggregation: str = "mean"):
        if aggregation not in ("mean", "max"):
            raise ValueError("aggregation must be 'mean' or 'max'")
        self.k = k
        self.aggregation = aggregation

    def fit(self, train_embeddings: np.ndarray) -> None:
        if train_embeddings.ndim == 3:
            N, P, D = train_embeddings.shape
            train_embeddings = train_embeddings.reshape(-1, D)
        self._index = build_index(train_embeddings)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists = query_index(self._index, test_embeddings, k=self.k)
        return dists.mean(axis=1) if self.aggregation == "mean" else dists.max(axis=1)

    def score_patches(
        self,
        train_patches: np.ndarray,  # (N_train, n_patches, D) — unused, index built in fit
        test_patches: np.ndarray,   # (N_test,  n_patches, D)
    ) -> np.ndarray:                # (N_test,  n_patches)
        N_test, n_patches, D = test_patches.shape
        dists = query_index(self._index, test_patches.reshape(-1, D), k=self.k)
        agg = dists.mean(axis=1) if self.aggregation == "mean" else dists.max(axis=1)
        return agg.reshape(N_test, n_patches)