from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATASETS = [
    "BraTS2021_slice",
    "Chest-RSNA",
    "OCT2017",
    "RESC",
    "camelyon16_256",
    "hist_DIY",
    "MLL23",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class BMADDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str,
        image_size: int = 224,
        normalize: bool = True,
    ):
        self.root = Path(root) / dataset_name
        self.split = split
        t = [transforms.Resize((image_size, image_size), antialias=True), transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = transforms.Compose(t)

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        def collect(directory: Path) -> list[Path]:
            src = directory / "img" if (directory / "img").exists() else directory
            return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        if split == "train":
            self.image_paths.extend(collect(self.root / "train" / "good"))
            self.labels.extend([0] * len(self.image_paths))
        else:
            good_dir = self.root / "test" / "good"
            if good_dir.exists():
                imgs = collect(good_dir)
                self.image_paths.extend(imgs)
                self.labels.extend([0] * len(imgs))

            for name in ("Ungood", "ungood"):
                anomaly_dir = self.root / "test" / name
                if anomaly_dir.exists():
                    imgs = collect(anomaly_dir)
                    self.image_paths.extend(imgs)
                    self.labels.extend([1] * len(imgs))
                    break

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found for {dataset_name}/{split} under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

    @property
    def label_array(self) -> np.ndarray:
        return np.array(self.labels, dtype=np.int32)


MLL23_NORMAL_CLASSES = [
    "lymphocyte",
    "plasma_cell",
    "lymphocyte_large_granular",
    "hairy_cell",
    "smudge_cell",
    "neutrophil_segmented",
    "eosinophil",
    "monocyte",
    "myeloblast",
    "promyelocyte",
    "myelocyte",
    "promyelocyte_atypical",
    "normoblast",
]

MLL23_ANOMALY_CLASSES = [
    "lymphocyte_reactive",
    "lymphocyte_neoplastic",
    "metamyelocyte",
    "basophil",
    "neutrophil_band",
]


class MLL23Dataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 224,
        train_ratio: float = 0.8,
        seed: int = 42,
        normalize: bool = True,
    ):
        self.root = Path(root) / "MLL23"
        t = [transforms.Resize((image_size, image_size), antialias=True), transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = transforms.Compose(t)

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        rng = np.random.default_rng(seed)

        # Normal classes: split 80/20 into train/test
        for cls in MLL23_NORMAL_CLASSES:
            images = sorted(
                p for p in (self.root / cls).iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            indices = rng.permutation(len(images))
            cutoff = int(len(images) * train_ratio)
            selected = indices[:cutoff] if split == "train" else indices[cutoff:]
            for i in selected:
                self.image_paths.append(images[i])
                self.labels.append(0)

        # Anomaly classes: all images go to test only
        if split == "test":
            for cls in MLL23_ANOMALY_CLASSES:
                images = sorted(
                    p for p in (self.root / cls).iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                )
                self.image_paths.extend(images)
                self.labels.extend([1] * len(images))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found for MLL23/{split} under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

    @property
    def label_array(self) -> np.ndarray:
        return np.array(self.labels, dtype=np.int32)


def get_dataloader(
    root: str | Path,
    dataset_name: str,
    split: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
) -> DataLoader:
    if dataset_name == "MLL23":
        dataset = MLL23Dataset(root, split, image_size, normalize=normalize)
    else:
        dataset = BMADDataset(root, dataset_name, split, image_size, normalize=normalize)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
