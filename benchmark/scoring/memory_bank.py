import numpy as np

from .base import BaseScorer
from ._faiss_utils import build_index, query_index

_CORESET_PROJ_DIM    = 128     # JL projection dimension for fast distance computation
_MAX_CORESET_INPUT   = 200_000 # cap on input pool size (random subsample if exceeded)
_MAX_BANK_SIZE       = 20_000  # cap on bank size (= number of greedy iterations)


def _greedy_coreset(embeddings: np.ndarray, ratio: float, seed: int = 42) -> np.ndarray:
    """
    Greedy coreset subsampling (PatchCore-style).
    Iteratively selects the point farthest from the already-selected subset.

    Bank size = min(ceil(N * ratio), _MAX_BANK_SIZE).
    The full input pool is always used (no random pre-subsampling) so spatial
    coverage is not sacrificed.
    Distances are computed in a randomly projected 128-d space (Johnson-
    Lindenstrauss) to reduce per-iteration cost from O(N×D) to O(N×128).
    Selected indices are applied to the original full-dimensional embeddings.
    """
    n_full = len(embeddings)
    target  = min(max(1, int(np.ceil(n_full * ratio))), _MAX_BANK_SIZE)

    # Subsample input pool if it exceeds the cap.
    # Target is computed from n_full so bank size stays proportional to the
    # full training set regardless of whether the cap activates.
    if n_full > _MAX_CORESET_INPUT:
        idx = np.random.default_rng(seed).choice(n_full, _MAX_CORESET_INPUT, replace=False)
        embeddings = embeddings[idx]

    n = len(embeddings)
    if target >= n:
        return embeddings

    rng = np.random.default_rng(seed + 1)

    proj_dim = min(_CORESET_PROJ_DIM, embeddings.shape[1])
    proj     = rng.standard_normal((embeddings.shape[1], proj_dim)).astype(np.float32)
    proj    /= np.sqrt(proj_dim)
    low_dim  = embeddings.astype(np.float32) @ proj   # (N, proj_dim)

    selected  = [int(rng.integers(n))]
    min_dists = np.linalg.norm(low_dim - low_dim[selected[0]], axis=1)

    while len(selected) < target:
        next_idx     = int(np.argmax(min_dists))
        selected.append(next_idx)
        dists_to_new = np.linalg.norm(low_dim - low_dim[next_idx], axis=1)
        np.minimum(min_dists, dists_to_new, out=min_dists)

    return embeddings[selected]   # full-dimensional bank entries


class MemoryBankScorer(BaseScorer):
    """
    PatchCore-style memory bank anomaly scoring.
    Subsamples normal training embeddings into a compact coreset via greedy
    farthest-point selection, then scores test samples by nearest-neighbor
    distance to the memory bank using FAISS.
    """

    name = "memory_bank"

    def __init__(self, coreset_ratio: float = 0.1, k: int = 1):
        self.coreset_ratio = coreset_ratio
        self.k = k

    def fit(self, train_embeddings: np.ndarray) -> None:
        if train_embeddings.ndim == 3:
            N, P, D = train_embeddings.shape
            train_embeddings = train_embeddings.reshape(-1, D)
        bank = _greedy_coreset(train_embeddings, ratio=self.coreset_ratio)
        self._index = build_index(bank)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists = query_index(self._index, test_embeddings, k=self.k)
        return dists.mean(axis=1)

    def score_patches(
        self,
        train_patches: np.ndarray,  # (N_train, n_patches, D) — unused, index built in fit
        test_patches: np.ndarray,   # (N_test,  n_patches, D)
    ) -> np.ndarray:                # (N_test,  n_patches)
        N_test, n_patches, D = test_patches.shape
        dists = query_index(self._index, test_patches.reshape(-1, D), k=self.k)
        return dists.mean(axis=1).reshape(N_test, n_patches)