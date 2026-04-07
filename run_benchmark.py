import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from benchmark.datasets.loader import DATASETS, BMADDataset, get_dataloader
from benchmark.evaluation.metrics import evaluate, evaluate_pixel_auroc, find_best_f1_threshold, summarize_results
from benchmark.evaluation.aupro import compute_aupro, patch_scores_to_maps
from benchmark.feature_extraction.base import EmbeddingCache
from benchmark.feature_extraction.dinov2 import DINOv2Extractor
from benchmark.feature_extraction.dinov3 import DINOv3Extractor
from benchmark.scoring.cosine import CosineScorer
from benchmark.scoring.euclidean import EuclideanScorer
from benchmark.scoring.kmeans import KMeansScorer
from benchmark.scoring.knn import KNNScorer
from benchmark.scoring.mahalanobis import MahalanobisScorer
from benchmark.scoring.memory_bank import MemoryBankScorer

EXTRACTORS = {
    "dinov2": DINOv2Extractor,
    "dinov3": DINOv3Extractor,
}

SCORERS = {
    "cosine": CosineScorer,
    "euclidean": EuclideanScorer,
    "knn": KNNScorer,
    "mahalanobis": MahalanobisScorer,
    "kmeans": KMeansScorer,
    "memory_bank": MemoryBankScorer,
}


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def extract_patches_or_load(
    extractor,
    cache: EmbeddingCache,
    data_root: str,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str,
    mll23_subset_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract or load cached patch embeddings; returns (patches, labels).

    patches shape: (N, n_patches, D)
    labels shape:  (N,)
    """
    if cache.patches_exist(extractor.name, dataset_name, split):
        print(f"    [{extractor.name}] Loading cached {split} patch embeddings...")
        patches = cache.load_patches(extractor.name, dataset_name, split)
        labels = cache.load_labels(extractor.name, dataset_name, split)
        return patches, labels

    print(f"    [{extractor.name}] Extracting {split} patch embeddings...")
    loader = get_dataloader(
        data_root, dataset_name, split,
        image_size=extractor.image_size,
        patch_size=extractor.patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mll23_subset_fraction=mll23_subset_fraction,
    )
    patches = extractor.extract_patches(loader, device=device)
    labels = loader.dataset.label_array
    cache.save_patches(patches, extractor.name, dataset_name, split)
    cache.save_labels(labels, extractor.name, dataset_name, split)
    return patches, labels


def _load_all_masks(dataset: BMADDataset, batch_size: int) -> np.ndarray:
    """Load every mask from a mask-enabled BMADDataset; returns (N, H, W) uint8."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts = []
    for _, _, mask in loader:
        parts.append(mask.numpy()[:, 0, :, :])  # (B, H, W)
    return np.concatenate(parts, axis=0).astype(np.uint8)



def run(args) -> dict:
    device = get_device()
    print(f"Device: {device}\n")

    cache = EmbeddingCache(args.cache_dir)
    results: dict = {}

    for dataset_name in args.datasets:
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        results[dataset_name] = {}

        for ext_name in args.extractors:
            extractor = EXTRACTORS[ext_name]()

            # ----------------------------------------------------------------
            # Check for pixel-level masks (BMAD datasets only, not MLL23).
            # Loaded at extractor.image_size so masks match anomaly maps.
            # ----------------------------------------------------------------
            has_masks = False
            test_masks: np.ndarray | None = None
            if dataset_name != "MLL23":
                try:
                    mask_ds = BMADDataset(
                        args.data_root, dataset_name, "test",
                        image_size=extractor.image_size,
                        patch_size=extractor.patch_size,
                        return_masks=True,
                    )
                    if mask_ds.has_masks:
                        has_masks = True
                        test_masks = _load_all_masks(mask_ds, args.batch_size)
                        n_with_mask = int((test_masks.sum(axis=(1, 2)) > 0).sum())
                        print(f"  Pixel masks available: {n_with_mask} anomaly images with non-zero masks")
                except Exception as exc:
                    print(f"  Mask loading skipped: {exc}")
            results[dataset_name][ext_name] = {}

            # ------------------------------------------------------------
            # Patch embeddings — always extracted; drive image-level scoring.
            # Labels are saved/loaded alongside patches.
            # ------------------------------------------------------------
            print(f"    [{ext_name}] Loading/extracting patch embeddings...")
            train_patches, _ = extract_patches_or_load(
                extractor, cache, args.data_root, dataset_name, "train",
                args.batch_size, args.num_workers, device, args.mll23_subset_fraction,
            )
            test_patches, test_labels = extract_patches_or_load(
                extractor, cache, args.data_root, dataset_name, "test",
                args.batch_size, args.num_workers, device, args.mll23_subset_fraction,
            )

            # ----------------------------------------------------------------
            # Val patches — BMAD datasets only (MLL23 has no val split).
            # Used to find the F1 threshold; kept outside the scorer loop
            # so patches are loaded once and reused across all scorers.
            # ----------------------------------------------------------------
            val_patches, val_labels = None, None
            if dataset_name != "MLL23":
                val_patches, val_labels = extract_patches_or_load(
                    extractor, cache, args.data_root, dataset_name, "valid",
                    args.batch_size, args.num_workers, device,
                )

            for scorer_name in args.scorers:
                print(f"    [{ext_name}] Scoring with {scorer_name}...", end=" ", flush=True)
                scorer = SCORERS[scorer_name]()

                #_, n_patches, D = train_patches.shape
                #train_flat = train_patches.reshape(-1, D)

                tracemalloc.start()
                t0 = time.perf_counter()
                try:
                    scorer.fit(train_patches)
                    fit_time = time.perf_counter() - t0
                    _, fit_peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    fit_peak = fit_peak_bytes / 1024 ** 2

                    tracemalloc.start()
                    t0 = time.perf_counter()
                    patch_scores = scorer.score_patches(train_patches, test_patches)
                    image_scores = patch_scores.max(axis=1)
                    score_time = time.perf_counter() - t0
                    _, score_peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    score_peak = score_peak_bytes / 1024 ** 2

                    # BMAD datasets: find threshold on val, apply to test.
                    # MLL23: no val split — find threshold on test (optimistic).
                    if val_patches is not None:
                        val_scores = scorer.score_patches(train_patches, val_patches).max(axis=1)
                        threshold = find_best_f1_threshold(val_scores, val_labels)
                    else:
                        threshold = None

                    metrics = evaluate(image_scores, test_labels, threshold=threshold)
                    metrics["fit_time_s"] = round(fit_time, 4)
                    metrics["score_time_s"] = round(score_time, 4)
                    metrics["fit_peak_mb"] = round(fit_peak, 2)
                    metrics["score_peak_mb"] = round(score_peak, 2)

                    # --------------------------------------------------------
                    # AUPRO — each scorer produces its own spatial anomaly maps
                    # via score_patches(), giving scorer-specific AUPRO values.
                    # --------------------------------------------------------
                    if has_masks and test_masks is not None:
                        anomaly_maps = patch_scores_to_maps(
                            patch_scores, image_size=extractor.image_size
                        )
                        metrics["pixel_AUROC"] = evaluate_pixel_auroc(anomaly_maps, test_masks)
                        metrics["AUPRO"] = compute_aupro(anomaly_maps, test_masks)

                    results[dataset_name][ext_name][scorer_name] = metrics
                except Exception as exc:
                    print(f"    FAILED: {exc}")
                    results[dataset_name][ext_name][scorer_name] = {"error": str(exc)}

                log = (
                    f"AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}"
                    f"  F1={metrics['F1']:.4f}"
                )
                if "pixel_AUROC" in metrics:
                    log += f"  pxAUROC={metrics['pixel_AUROC']:.4f}"
                if "AUPRO" in metrics:
                    log += f"  AUPRO={metrics['AUPRO']:.4f}"
                log += (
                    f"  fit={metrics['fit_time_s']:.3f}s  score={metrics['score_time_s']:.3f}s"
                    f"  fit_mem={metrics['fit_peak_mb']:.1f}MB"
                    f"  score_mem={metrics['score_peak_mb']:.1f}MB"
                )
                print(log)

    return results


def save_results(results: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medical Imaging Anomaly Detection Benchmark"
    )
    parser.add_argument(
        "--data-root",
        default="./Datasets",
        help="Root directory containing BMAD datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS,
        metavar="DATASET",
        help=f"Datasets to evaluate. Choices: {DATASETS}",
    )
    parser.add_argument(
        "--extractors", nargs="+", default=list(EXTRACTORS), choices=list(EXTRACTORS),
        metavar="EXTRACTOR",
        help="Feature extractors to use (dinov2, dinov3)",
    )
    parser.add_argument(
        "--scorers", nargs="+", default=list(SCORERS), choices=list(SCORERS),
        metavar="SCORER",
        help=f"Anomaly scoring methods. Choices: {list(SCORERS)}",
    )
    parser.add_argument("--cache-dir", default="./cache", help="Embedding cache directory")
    parser.add_argument(
        "--output", default="./results/benchmark_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--mll23-subset-fraction", type=float, default=0.7, dest="mll23_subset_fraction",
        metavar="FRACTION",
        help="Fraction of MLL23 images to use per class (0.0–1.0). Default: 0.7.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run(args)
    print("\n" + summarize_results(results))
    save_results(results, args.output)