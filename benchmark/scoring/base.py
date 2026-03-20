from abc import ABC, abstractmethod

import numpy as np


class BaseScorer(ABC):
    name: str

    @abstractmethod
    def fit(self, train_embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def score(self, test_embeddings: np.ndarray) -> np.ndarray: ...
