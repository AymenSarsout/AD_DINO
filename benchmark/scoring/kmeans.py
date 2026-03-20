import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans

from .base import BaseScorer


class KMeansScorer(BaseScorer):
    """
    Anomaly score = distance to the nearest k-means centroid fitted on normal training samples.
    Uses MiniBatchKMeans for scalability on large training sets.
    """

    name = "kmeans"

    def __init__(self, n_clusters: int = 32):
        self.n_clusters = n_clusters

    def fit(self, train_embeddings: np.ndarray) -> None:
        n_clusters = min(self.n_clusters, len(train_embeddings))
        self._kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=1024,
        )
        self._kmeans.fit(train_embeddings)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        dists = cdist(test_embeddings, self._kmeans.cluster_centers_, metric="euclidean")
        return dists.min(axis=1)

