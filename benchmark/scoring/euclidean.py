import numpy as np
from scipy.spatial.distance import cdist

from .base import BaseScorer


class EuclideanScorer(BaseScorer):
    """
    Anomaly score = minimum Euclidean distance to any normal training sample.
    """

    name = "euclidean"

    def fit(self, train_embeddings: np.ndarray) -> None:
        self._train = train_embeddings

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists = cdist(test_embeddings, self._train, metric="euclidean")
        return dists.min(axis=1)
