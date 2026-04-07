import numpy as np
import torch
from transformers import AutoModel

from .base import FeatureExtractor


class DINOv2Extractor(FeatureExtractor):
    name = "dinov2"
    image_size = 224
    patch_size = 14
    MODEL_ID = "facebook/dinov2-small"

    def __init__(self):
        self.model = None
        self._device = None

    def load_model(self, device: str) -> None:
        if self.model is not None and self._device == device:
            return
        self.model = AutoModel.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.model = self.model.to(device)
        self._device = device

    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            outputs = self.model(pixel_values=images)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
        return cls_tokens.cpu().float().numpy()

    @property
    def n_patches(self) -> int:
        """Number of patch tokens per image.
        For DINOv2-giant: (224 // 14)^2 = 256 patches."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        patch_size = getattr(self.model.config, 'patch_size', 14)
        return (self.image_size // patch_size) ** 2

    def extract_patches_batch(self, images: torch.Tensor) -> np.ndarray:
        """Return patch token embeddings, shape (B, n_patches, D).

        For a 224x224 image with patch_size=14 the grid is 16x16 = 256 patches.
        Index 0 of last_hidden_state is the CLS token; indices 1: are patch tokens
        in row-major (top-left → bottom-right) order.
        """
        with torch.inference_mode():
            outputs = self.model(pixel_values=images)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # skip CLS
        return patch_tokens.cpu().float().numpy()

