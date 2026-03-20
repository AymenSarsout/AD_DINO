import numpy as np
import torch
from transformers import AutoModel

from .base import FeatureExtractor


class DINOv2Extractor(FeatureExtractor):
    name = "dinov2"
    image_size = 224
    MODEL_ID = "facebook/dinov2-giant"

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
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().float().numpy()
