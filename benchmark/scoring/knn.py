import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseScorer


class KNNScorer(BaseScorer):
    name = "knn"

    def __init__(self, k: int = 5, aggregation: str = "mean"):
        if aggregation not in ("mean", "max"):
            raise ValueError("aggregation must be 'mean' or 'max'")
        self.k = k
        self.aggregation = aggregation

    def fit(self, train_embeddings: np.ndarray) -> None:
        k = min(self.k, len(train_embeddings))
        self._nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        self._nn.fit(train_embeddings)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists, _ = self._nn.kneighbors(test_embeddings)
        return dists.mean(axis=1) if self.aggregation == "mean" else dists.max(axis=1)
