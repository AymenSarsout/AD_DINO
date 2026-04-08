from abc import ABC, abstractmethod

import numpy as np


# To add a new scorer:
# 1. Create a new file in this directory
# 2. Subclass BaseScorer and implement the three abstract methods below.
# 3. Set `name` (used as the key in results).
# 4. Register it in the SCORERS dict in run_benchmark.py and run_experiments.py.
# See knn.py for a minimal reference implementation.

class BaseScorer(ABC):
    name: str

    @abstractmethod
    def fit(self, train_embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def score(self, test_embeddings: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def score_patches(
        self,
        train_patches: np.ndarray,  # (N_train, n_patches, D)
        test_patches: np.ndarray,   # (N_test,  n_patches, D)
    ) -> np.ndarray: ...            # (N_test,  n_patches)
