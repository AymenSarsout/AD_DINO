import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .base import BaseScorer
from ._faiss_utils import build_index, query_index


class KMeansScorer(BaseScorer):
    """
    Anomaly score = distance to the nearest k-means centroid fitted on normal training samples.
    Uses MiniBatchKMeans for scalability on large training sets.
    """

    name = "kmeans"

    def __init__(self, n_clusters: int = 384):
        self.n_clusters = n_clusters

    def fit(self, train_embeddings: np.ndarray) -> None:
        if train_embeddings.ndim == 3:
            N, P, D = train_embeddings.shape
            train_embeddings = train_embeddings.reshape(-1, D)
        n_clusters = min(self.n_clusters, len(train_embeddings))
        self._kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=5,
            batch_size=8192,
        )
        self._kmeans.fit(train_embeddings)
        self._index = build_index(self._kmeans.cluster_centers_)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists = query_index(self._index, test_embeddings, k=1)
        return dists[:, 0]

    def score_patches(
        self,
        train_patches: np.ndarray,  # (N_train, n_patches, D) — unused, index built in fit
        test_patches: np.ndarray,   # (N_test,  n_patches, D)
    ) -> np.ndarray:                # (N_test,  n_patches)
        N_test, n_patches, D = test_patches.shape
        dists = query_index(self._index, test_patches.reshape(-1, D), k=1)
        return dists[:, 0].reshape(N_test, n_patches)

