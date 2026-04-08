from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# To add a new dataset:
# 1. Add its name to the DATASETS list above.
# 2. Place it under Datasets/<name>/ following the BMAD layout:
#      train/good/         — normal training images
#      test/good/          — normal test images
#      test/Ungood/        — anomalous test images
#      test/Ungood/label/  — (optional) pixel-level masks
#    BMADDataset will handle it automatically.
# 3. If your dataset has a non-standard folder structure (like MLL23),
#    write a dedicated Dataset subclass and add a branch for it in get_dataloader().

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


def _find_mask_dir(anomaly_dir: Path) -> Path | None:
    """Return the mask directory for a given anomaly image directory, or None."""
    candidates = [
        anomaly_dir / "label" / "img",
        anomaly_dir / "label",
        anomaly_dir.parent / "label" / anomaly_dir.name,
    ]
    for c in candidates:
        if c.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in c.iterdir()
        ):
            return c
    return None


def _find_mask_for_image(img_path: Path, mask_dir: Path) -> Path | None:
    """Return the mask file with the same stem as img_path inside mask_dir, or None."""
    for ext in IMAGE_EXTENSIONS:
        candidate = mask_dir / (img_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


class BMADDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str,
        image_size: int = 224,
        patch_size: int = 14,
        normalize: bool = True,
        return_masks: bool = False,
    ):
        self.root = Path(root) / dataset_name
        self.split = split
        self.return_masks = return_masks
        self.image_size = image_size

        # Resize shorter edge to image_size,
        # then center-crop to the nearest multiple of patch_size.
        crop_size = (image_size // patch_size) * patch_size
        t = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
        if normalize:
            t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = transforms.Compose(t)

        # Nearest-neighbor resize;
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        self.mask_paths: list[Path | None] = []

        def collect(directory: Path) -> list[Path]:
            src = directory / "img" if (directory / "img").exists() else directory
            return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        if split == "train":
            imgs = collect(self.root / "train" / "good")
            self.image_paths.extend(imgs)
            self.labels.extend([0] * len(imgs))
            self.mask_paths.extend([None] * len(imgs))
        else:
            # "valid" split: datasets use either "valid/" or "val/" folder name.
            if split == "valid":
                split_dir = next(
                    (self.root / d for d in ("valid", "val") if (self.root / d).exists()),
                    None,
                )
                if split_dir is None:
                    raise RuntimeError(
                        f"No valid/val directory found for {dataset_name} under {self.root}"
                    )
            else:
                split_dir = self.root / "test"

            good_dir = split_dir / "good"
            if good_dir.exists():
                imgs = collect(good_dir)
                self.image_paths.extend(imgs)
                self.labels.extend([0] * len(imgs))
                self.mask_paths.extend([None] * len(imgs))

            for name in ("Ungood", "ungood"):
                anomaly_dir = split_dir / name
                if anomaly_dir.exists():
                    imgs = collect(anomaly_dir)
                    mask_dir = _find_mask_dir(anomaly_dir)
                    for img_path in imgs:
                        self.image_paths.append(img_path)
                        self.labels.append(1)
                        mask_path = _find_mask_for_image(img_path, mask_dir) if mask_dir else None
                        self.mask_paths.append(mask_path)
                    break

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found for {dataset_name}/{split} under {self.root}")

    @property
    def has_masks(self) -> bool:
        """True if at least one anomaly mask file was found on disk."""
        return any(p is not None for p in self.mask_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(img)
        label = self.labels[idx]

        if not self.return_masks:
            return image, label

        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask_pil = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask_pil)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return image, label, mask

    @property
    def label_array(self) -> np.ndarray:
        return np.array(self.labels, dtype=np.int32)


MLL23_NORMAL_CLASSES = [
    "lymphocyte",
    "plasma_cell",
    "lymphocyte_large_granular",
    "neutrophil_segmented",
    "eosinophil",
    "monocyte",
    "normoblast",
]

MLL23_ANOMALY_CLASSES = [
    "myeloblast",           # 8,606 — hallmark of AML/ALL
    "promyelocyte_atypical",# 2,033 — hallmark of APL
    "promyelocyte",         #   745 — immature myeloid, leukaemia-relevant
    "myelocyte",            #   747 — immature myeloid, CML-relevant
    "metamyelocyte",        #   483 — immature myeloid, elevated in CML
]


class MLL23Dataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 224,
        patch_size: int = 14,
        train_ratio: float = 0.8,
        seed: int = 42,
        normalize: bool = True,
        subset_fraction: float = 0.7,
    ):
        self.root = Path(root) / "MLL23"
        crop_size = (image_size // patch_size) * patch_size
        t = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
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
            if subset_fraction < 1.0:
                keep = max(1, int(len(selected) * subset_fraction))
                selected = selected[:keep]
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
                if subset_fraction < 1.0:
                    keep = max(1, int(len(images) * subset_fraction))
                    images = [images[i] for i in rng.permutation(len(images))[:keep]]
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
    patch_size: int = 14,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    return_masks: bool = False,
    mll23_subset_fraction: float = 0.7,
) -> DataLoader:
    if dataset_name == "MLL23":
        dataset = MLL23Dataset(root, split, image_size, patch_size=patch_size, normalize=normalize,
                               subset_fraction=mll23_subset_fraction)
    else:
        dataset = BMADDataset(root, dataset_name, split, image_size, patch_size=patch_size,
                              normalize=normalize, return_masks=return_masks)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
