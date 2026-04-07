import numpy as np
import torch
from transformers import AutoModel

from .base import FeatureExtractor


class DINOv3Extractor(FeatureExtractor):
    name = "dinov3"
    image_size = 256
    patch_size = 16
    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

    def __init__(self):
        self.model = None
        self._device = None
        self._num_register_tokens = 0

    def load_model(self, device: str) -> None:
        if self.model is not None and self._device == device:
            return
        self.model = AutoModel.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.model = self.model.to(device)
        self._device = device
        self._num_register_tokens = getattr(
            self.model.config, 'num_register_tokens', 0
        )

    @property
    def n_patches(self) -> int:
        """Number of patch tokens per image after register token removal.
        Computed from image_size and patch_size in the model config."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        patch_size = getattr(self.model.config, 'patch_size', 16)
        return (self.image_size // patch_size) ** 2

    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            outputs = self.model(pixel_values=images)
            cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token.cpu().float().numpy()

    def extract_patches_batch(self, images: torch.Tensor) -> np.ndarray:
        """Return patch token embeddings, shape (B, n_patches, D)."""
        with torch.inference_mode():
            outputs = self.model(pixel_values=images)
            skip = 1 + self._num_register_tokens  # CLS + registers
            patch_tokens = outputs.last_hidden_state[:, skip:, :]
        return patch_tokens.cpu().float().numpy()


class DINOv3CLSRegExtractor(DINOv3Extractor):
    """DINOv3 global embedding: mean of the CLS token and all register tokens.

    DINOv3 offloads global/redundant information into its 4 register tokens,
    so the CLS token alone under-represents the model's global understanding.
    Averaging CLS + registers recovers a richer 384-d global descriptor that
    is directly comparable in dimension to DINOv2's CLS token.
    """

    name = "dinov3_cls_reg"

    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            outputs = self.model(pixel_values=images)
            n_global = 1 + self._num_register_tokens       # CLS + all registers
            global_tokens = outputs.last_hidden_state[:, :n_global, :]  # (B, n_global, D)
            global_emb = global_tokens.mean(dim=1)         # (B, D)
        return global_emb.cpu().float().numpy()