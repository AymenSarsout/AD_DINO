from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor(ABC):
    name: str
    image_size: int

    @abstractmethod
    def load_model(self, device: str) -> None: ...

    @abstractmethod
    def extract_batch(self, images: torch.Tensor) -> np.ndarray: ...

    @abstractmethod
    def extract_patches_batch(self, images: torch.Tensor) -> np.ndarray: ...
    """Return patch token embeddings for a batch; shape (B, n_patches, D)."""

    def extract(self, dataloader: DataLoader, device: str = "cpu") -> np.ndarray:
        self.load_model(device)
        all_embeddings = []
        for batch in tqdm(dataloader, desc=f"Extracting [{self.name}]", leave=False):
            images = batch[0].to(device)
            embeddings = self.extract_batch(images)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)

    def extract_patches(self, dataloader: DataLoader, device: str = "cpu") -> np.ndarray:
        """Extract patch embeddings for all images; returns (N, n_patches, D)."""
        self.load_model(device)
        all_patches = []
        for batch in tqdm(dataloader, desc=f"Extracting patches [{self.name}]", leave=False):
            images = batch[0].to(device)
            all_patches.append(self.extract_patches_batch(images))
        return np.concatenate(all_patches, axis=0)


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _emb_path(self, extractor: str, dataset: str, split: str) -> Path:
        return self.cache_dir / extractor / f"{dataset}_{split}.npy"

    def _lbl_path(self, extractor: str, dataset: str, split: str) -> Path:
        return self.cache_dir / extractor / f"{dataset}_{split}_labels.npy"

    def exists(self, extractor: str, dataset: str, split: str) -> bool:
        return self._emb_path(extractor, dataset, split).exists()

    def save(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        extractor: str,
        dataset: str,
        split: str,
    ) -> None:
        path = self._emb_path(extractor, dataset, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        np.save(self._lbl_path(extractor, dataset, split), labels)

    def load(self, extractor: str, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        embeddings = np.load(self._emb_path(extractor, dataset, split))
        labels = np.load(self._lbl_path(extractor, dataset, split))
        return embeddings, labels

    # ------------------------------------------------------------------
    # Patch embedding cache (for spatial anomaly maps / AUPRO)
    # ------------------------------------------------------------------

    def _patch_path(self, extractor: str, dataset: str, split: str) -> Path:
        return self.cache_dir / extractor / f"{dataset}_{split}_patches.npy"

    def patches_exist(self, extractor: str, dataset: str, split: str) -> bool:
        return self._patch_path(extractor, dataset, split).exists()

    def save_patches(
        self,
        patches: np.ndarray,
        extractor: str,
        dataset: str,
        split: str,
    ) -> None:
        path = self._patch_path(extractor, dataset, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, patches)

    def load_patches(self, extractor: str, dataset: str, split: str) -> np.ndarray:
        return np.load(self._patch_path(extractor, dataset, split))

    def save_labels(self, labels: np.ndarray, extractor: str,
                    dataset: str, split: str) -> None:
        path = self._lbl_path(extractor, dataset, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, labels)

    def load_labels(self, extractor: str, dataset: str,
                    split: str) -> np.ndarray:
        return np.load(self._lbl_path(extractor, dataset, split))
