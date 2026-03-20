import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseScorer


def _greedy_coreset(embeddings: np.ndarray, ratio: float, seed: int = 42) -> np.ndarray:
    """
    Greedy coreset subsampling (PatchCore-style).
    Iteratively selects the point farthest from the already-selected subset,
    producing a diverse, representative memory bank of size ceil(N * ratio).
    """
    n = len(embeddings)
    target = max(1, int(np.ceil(n * ratio)))
    if target >= n:
        return embeddings

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    # min distance from each point to the current selected set
    min_dists = np.linalg.norm(embeddings - embeddings[selected[0]], axis=1)

    while len(selected) < target:
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        dists_to_new = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        np.minimum(min_dists, dists_to_new, out=min_dists)

    return embeddings[selected]


class MemoryBankScorer(BaseScorer):
    """
    PatchCore-style memory bank anomaly scoring.
    Subsamples normal training embeddings into a compact coreset via greedy
    farthest-point selection, then scores test samples by nearest-neighbor distance
    to the memory bank.
    """

    name = "memory_bank"

    def __init__(self, coreset_ratio: float = 0.1, k: int = 1):
        self.coreset_ratio = coreset_ratio
        self.k = k

    def fit(self, train_embeddings: np.ndarray) -> None:
        bank = _greedy_coreset(train_embeddings, ratio=self.coreset_ratio)
        k = min(self.k, len(bank))
        self._nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        self._nn.fit(bank)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists, _ = self._nn.kneighbors(test_embeddings)
        return dists.mean(axis=1)
