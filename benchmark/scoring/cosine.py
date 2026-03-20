import numpy as np
from sklearn.preprocessing import normalize

from .base import BaseScorer


class CosineScorer(BaseScorer):
    """
    Anomaly score = 1 - max cosine similarity to any normal training sample.
    Low similarity to all normal samples indicates anomaly.
    """

    name = "cosine"

    def fit(self, train_embeddings: np.ndarray) -> None:
        self._train_norm = normalize(train_embeddings)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        test_norm = normalize(test_embeddings)
        # (N_test, N_train) — each row is similarities to all train samples
        sims = test_norm @ self._train_norm.T
        return 1.0 - sims.max(axis=1)
