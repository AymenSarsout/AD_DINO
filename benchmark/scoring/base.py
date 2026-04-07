from abc import ABC, abstractmethod

import numpy as np


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
