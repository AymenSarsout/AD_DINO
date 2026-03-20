import numpy as np
import torch
from transformers import AutoModel

from .base import FeatureExtractor


class DINOv3Extractor(FeatureExtractor):
    name = "dinov3"
    image_size = 256
    MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

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
            if outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().float().numpy()
