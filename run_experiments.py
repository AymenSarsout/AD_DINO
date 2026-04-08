import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.datasets.loader import DATASETS, BMADDataset, get_dataloader
from benchmark.evaluation.aupro import compute_aupro, patch_scores_to_maps
from benchmark.evaluation.metrics import evaluate, evaluate_pixel_auroc, find_best_f1_threshold
from benchmark.feature_extraction.base import EmbeddingCache
from benchmark.feature_extraction.dinov2 import DINOv2Extractor
from benchmark.feature_extraction.dinov3 import DINOv3Extractor, DINOv3CLSRegExtractor
from benchmark.scoring.cosine import CosineScorer
from benchmark.scoring.euclidean import EuclideanScorer
from benchmark.scoring.kmeans import KMeansScorer
from benchmark.scoring.knn import KNNScorer
from benchmark.scoring.mahalanobis import MahalanobisScorer
from benchmark.scoring.memory_bank import MemoryBankScorer

EXTRACTORS = {
    "dinov2":        DINOv2Extractor,
    "dinov3":        DINOv3Extractor,
    "dinov3_cls_reg": DINOv3CLSRegExtractor,  # Exp 2: CLS + register tokens
}

# Hyperparameter configurations for patch-based scoring (Part 1)
EXPERIMENTS: dict[str, list[tuple]] = {
    "knn": [
        (KNNScorer, {"k": 1,  "aggregation": "mean"}),
        (KNNScorer, {"k": 3,  "aggregation": "mean"}),
        (KNNScorer, {"k": 5,  "aggregation": "mean"}),   # default
        (KNNScorer, {"k": 10, "aggregation": "mean"}),
        (KNNScorer, {"k": 20, "aggregation": "mean"}),
        (KNNScorer, {"k": 1,  "aggregation": "max"}),
        (KNNScorer, {"k": 3,  "aggregation": "max"}),
        (KNNScorer, {"k": 5,  "aggregation": "max"}),
        (KNNScorer, {"k": 10, "aggregation": "max"}),
        (KNNScorer, {"k": 20, "aggregation": "max"}),
    ],
    "kmeans": [
        (KMeansScorer, {"n_clusters": 384}),
        (KMeansScorer, {"n_clusters": 256}),
        (KMeansScorer, {"n_clusters": 512}),   # default
        (KMeansScorer, {"n_clusters": 64}),
        (KMeansScorer, {"n_clusters": 128}),
    ],
    "mahalanobis": [
        #(MahalanobisScorer, {"n_components": 384, "reg": 1e-1}),
        (MahalanobisScorer, {"n_components": 256, "reg": 1e-1}),
        (MahalanobisScorer, {"n_components": 128, "reg": 1e-1}),

        #(MahalanobisScorer, {"n_components": 384, "reg": 1e-2}),
        (MahalanobisScorer, {"n_components": 256, "reg": 1e-2}),
        (MahalanobisScorer, {"n_components": 128, "reg": 1e-2}),  # default

        #(MahalanobisScorer, {"n_components": 384, "reg": 1e-3}),
        (MahalanobisScorer, {"n_components": 128, "reg": 1e-3}),
        (MahalanobisScorer, {"n_components": 256, "reg": 1e-3}),
    ],
    "memory_bank": [
        #(MemoryBankScorer, {"coreset_ratio": 0.01, "k": 1}),
        #(MemoryBankScorer, {"coreset_ratio": 0.05, "k": 1}),
        #(MemoryBankScorer, {"coreset_ratio": 0.10, "k": 1}),   # default
        #(MemoryBankScorer, {"coreset_ratio": 0.25, "k": 1}),
        #(MemoryBankScorer, {"coreset_ratio": 0.50, "k": 1}),
        #(MemoryBankScorer, {"coreset_ratio": 0.10, "k": 3}),
        #(MemoryBankScorer, {"coreset_ratio": 0.10, "k": 5}),
    ],
}

# All scorers with default params for mean-pooled comparison (Part 2)
MEAN_POOLED_SCORERS = {
    "cosine":      (CosineScorer,      {}),
    "euclidean":   (EuclideanScorer,   {}),
    "knn":         (KNNScorer,         {"k": 5, "aggregation": "mean"}),
    "mahalanobis": (MahalanobisScorer, {"reg": 1e-2}),
    "kmeans":      (KMeansScorer,      {"n_clusters": 384}),
    "memory_bank": (MemoryBankScorer,  {"coreset_ratio": 0.10, "k": 1}),
}


def _experiment_label(scorer_cls, params: dict) -> str:
    base = scorer_cls.name
    parts = [f"{k}={v}" for k, v in params.items()]
    return f"{base}[{', '.join(parts)}]"


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fit_and_score_patches(
    scorer_cls,
    params: dict,
    train_patches: np.ndarray,          # (N_train, n_patches, D)
    test_patches: np.ndarray,           # (N_test,  n_patches, D)
    test_labels: np.ndarray,
    val_patches: np.ndarray | None,     # (N_val, n_patches, D) or None
    val_labels: np.ndarray | None,      # (N_val,) or None
    test_masks: np.ndarray | None,      # (N_test, H, W) uint8, or None
    image_size: int = 224,
) -> dict:

    scorer = scorer_cls(**params)
    #_, n_patches, D = train_patches.shape

    tracemalloc.start()
    t0 = time.perf_counter()
    scorer.fit(train_patches)
    fit_time = time.perf_counter() - t0
    _, fit_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    patch_scores = scorer.score_patches(train_patches, test_patches)
    image_scores = patch_scores.max(axis=1)
    score_time = time.perf_counter() - t0
    _, score_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if val_patches is not None:
        val_scores = scorer.score_patches(train_patches, val_patches).max(axis=1)
        threshold = find_best_f1_threshold(val_scores, val_labels)
    else:
        threshold = None  # MLL23: test-optimal threshold

    metrics = evaluate(image_scores, test_labels, threshold=threshold)
    metrics["fit_time_s"]    = round(fit_time, 4)
    metrics["score_time_s"]  = round(score_time, 4)
    metrics["fit_peak_mb"]   = round(fit_peak_bytes / 1024 ** 2, 2)
    metrics["score_peak_mb"] = round(score_peak_bytes / 1024 ** 2, 2)

    if test_masks is not None:
        anomaly_maps = patch_scores_to_maps(patch_scores, image_size=image_size)
        metrics["pixel_AUROC"] = evaluate_pixel_auroc(anomaly_maps, test_masks)
        metrics["AUPRO"]       = compute_aupro(anomaly_maps, test_masks)

    return metrics


def _fit_and_score_mean_pooled(
    scorer_cls,
    params: dict,
    train_emb: np.ndarray,   # (N_train, D)
    test_emb: np.ndarray,    # (N_test,  D)
    test_labels: np.ndarray,
) -> dict:
    """Fit and score on mean-pooled image-level embeddings."""
    scorer = scorer_cls(**params)

    tracemalloc.start()
    t0 = time.perf_counter()
    scorer.fit(train_emb)
    fit_time = time.perf_counter() - t0
    _, fit_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    scores = scorer.score(test_emb)
    score_time = time.perf_counter() - t0
    _, score_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = evaluate(scores, test_labels)
    metrics["fit_time_s"]    = round(fit_time, 4)
    metrics["score_time_s"]  = round(score_time, 4)
    metrics["fit_peak_mb"]   = round(fit_peak_bytes / 1024 ** 2, 2)
    metrics["score_peak_mb"] = round(score_peak_bytes / 1024 ** 2, 2)
    return metrics


def run_scorer_experiments(args) -> dict:
    """Same pipeline as run_benchmark.py but sweeping scorer hyperparameters."""
    cache = EmbeddingCache(args.cache_dir)
    results = {}

    experiment_names = args.experiments
    if not experiment_names:
        return results

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")

        for ext_name in args.extractors:
            print(f"  Extractor: {ext_name}")

            if not cache.patches_exist(ext_name, dataset_name, "train"):
                print(f"    [SKIP] No cached patch embeddings for {dataset_name}/{ext_name}. "
                      f"Run run_benchmark.py first.")
                continue

            train_patches = cache.load_patches(ext_name, dataset_name, "train")
            test_patches  = cache.load_patches(ext_name, dataset_name, "test")
            test_labels   = cache.load_labels(ext_name, dataset_name, "test")

            # Val patches — BMAD datasets only (MLL23 has no val split).
            val_patches, val_labels = None, None
            if dataset_name != "MLL23" and cache.patches_exist(ext_name, dataset_name, "valid"):
                val_patches = cache.load_patches(ext_name, dataset_name, "valid")
                val_labels  = cache.load_labels(ext_name, dataset_name, "valid")
                print(f"    [{ext_name}] Loaded val patches ({len(val_labels)} samples)")
            elif dataset_name != "MLL23":
                print(f"    [{ext_name}] No val cache found — run run_benchmark.py first to cache val patches.")

            extractor   = EXTRACTORS[ext_name]()
            test_masks  = None
            if dataset_name != "MLL23":
                try:
                    from torch.utils.data import DataLoader
                    mask_ds = BMADDataset(
                        args.data_root, dataset_name, "test",
                        image_size=extractor.image_size,
                        patch_size=extractor.patch_size,
                        return_masks=True,
                    )
                    if mask_ds.has_masks:
                        parts = []
                        for _, _, mask in DataLoader(mask_ds, batch_size=32,
                                                     shuffle=False, num_workers=0):
                            parts.append(mask.numpy()[:, 0, :, :])
                        test_masks = np.concatenate(parts, axis=0).astype(np.uint8)
                        print(f"    Loaded pixel masks for {dataset_name}")
                except Exception as exc:
                    print(f"    Mask loading skipped: {exc}")

            for exp_name in experiment_names:
                for scorer_cls, params in EXPERIMENTS[exp_name]:
                    label = _experiment_label(scorer_cls, params)
                    key = (dataset_name, ext_name, label)

                    print(f"    {label}...", end=" ", flush=True)
                    try:
                        metrics = _fit_and_score_patches(
                            scorer_cls, params, train_patches, test_patches, test_labels,
                            val_patches=val_patches, val_labels=val_labels,
                            test_masks=test_masks, image_size=extractor.image_size,
                        )
                        results[key] = {"params": params, "metrics": metrics}
                    except Exception as exc:
                        print(f"FAILED: {exc}")
                        results[key] = {"params": params, "metrics": {"error": str(exc)}}
                        continue

                    log = (f"AUROC={metrics['AUROC']:.4f}  F1={metrics['F1']:.4f}"
                           f"  fit={metrics['fit_time_s']:.3f}s  score={metrics['score_time_s']:.3f}s"
                           f"  fit_mem={metrics['fit_peak_mb']:.1f}MB")
                    if "pixel_AUROC" in metrics:
                        log += f"  pxAUROC={metrics['pixel_AUROC']:.4f}"
                    if "AUPRO" in metrics:
                        log += f"  AUPRO={metrics['AUPRO']:.4f}"
                    print(log)

    return results




# ---------------------------------------------------------------------------
# Part 2 – Mean-pooled embedding comparison
# ---------------------------------------------------------------------------

def run_mean_pooled_experiments(args) -> dict:
    """
    Experiment 2 — Global representation comparison.

    Compares three global descriptors across all scorers (default hyperparams):
      - dinov2       : CLS token only (384-d)
      - dinov3       : CLS token only (384-d)
      - dinov3_cls_reg: mean(CLS + 4 register tokens) (384-d)

    dinov3_cls_reg is always included regardless of --extractors, since it is
    the whole point of this experiment.
    """
    device = get_device()
    print(f"\nDevice: {device}")

    mean_pooled_cache = EmbeddingCache(Path(args.cache_dir) / "mean_pooled")
    results = {}

    # Always include dinov3_cls_reg for the global-representation comparison.
    exp2_extractors = list(dict.fromkeys(args.extractors + ["dinov3_cls_reg"]))

    print(f"\n{'='*60}")
    print("Experiment 2 — Global representation comparison")
    print(f"Extractors: {exp2_extractors}")

    for dataset_name in args.datasets:
        print(f"\n  Dataset: {dataset_name}")

        for ext_name in exp2_extractors:
            print(f"    Extractor: {ext_name}")
            extractor = EXTRACTORS[ext_name]()

            for split in ("train", "test"):
                if mean_pooled_cache.exists(ext_name, dataset_name, split):
                    print(f"      [{split}] Loading from cache...")
                else:
                    print(f"      [{split}] Extracting mean-pooled embeddings...")
                    loader = get_dataloader(
                        args.data_root, dataset_name, split,
                        image_size=extractor.image_size,
                        patch_size=extractor.patch_size,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                    embeddings = extractor.extract(loader, device=device)
                    labels = loader.dataset.label_array
                    mean_pooled_cache.save(embeddings, labels, ext_name, dataset_name, split)

            train_emb, _          = mean_pooled_cache.load(ext_name, dataset_name, "train")
            test_emb, test_labels = mean_pooled_cache.load(ext_name, dataset_name, "test")

            for scorer_name, (scorer_cls, params) in MEAN_POOLED_SCORERS.items():
                label = f"mean_pooled/{_experiment_label(scorer_cls, params)}"
                key = (dataset_name, ext_name, label)

                print(f"      {scorer_name}...", end=" ", flush=True)
                metrics = _fit_and_score_mean_pooled(
                    scorer_cls, params, train_emb, test_emb, test_labels
                )
                results[key] = {"params": params, "metrics": metrics}

                print(
                    f"AUROC={metrics['AUROC']:.4f}  F1={metrics['F1']:.4f}"
                    f"  fit={metrics['fit_time_s']:.3f}s  score={metrics['score_time_s']:.3f}s"
                    f"  fit_mem={metrics['fit_peak_mb']:.1f}MB"
                )

    return results




# ---------------------------------------------------------------------------
# Part 3 – Global (CLS) + local (patch) score fusion  [BMAD datasets only]
# ---------------------------------------------------------------------------


FUSION_GLOBAL = {
    "dinov2": "dinov2",
    "dinov3": "dinov3",
}

# Alpha grid: 0.00, 0.05, …, 1.00
_ALPHA_GRID = np.arange(0.0, 1.05, 0.05)


def _z_normalize(ref_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    mu    = float(ref_scores.mean())
    sigma = float(ref_scores.std())
    if sigma < 1e-8:
        sigma = 1.0
    return (test_scores - mu) / sigma


def _find_best_alpha(
    global_scores: np.ndarray,
    local_scores:  np.ndarray,
    labels:        np.ndarray,
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    best_alpha, best_auroc = 0.5, 0.0
    for alpha in _ALPHA_GRID:
        combined = alpha * global_scores + (1.0 - alpha) * local_scores
        try:
            auroc = float(roc_auc_score(labels, combined))
        except Exception:
            continue
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = float(alpha)
    return best_alpha, best_auroc


def run_fusion_experiments(args) -> dict:
    """
    Experiment 3 — Global (CLS) + local (patch) score fusion.

    Only runs on BMAD datasets (MLL23 excluded — it has no val split, which is
    required for both z-score normalization reference and alpha grid search).

    For each (dataset, extractor, scorer):
      1. Fit a local scorer on patch embeddings; fit a global scorer on CLS embeddings.
      2. Z for both test score distributions using the mean and std of scores
         computed on *normal*
      3. Evaluate two fusion strategies:
           fusion_max     : score = max(global_norm, local_norm)
           fusion_weighted: score = α · global_norm + (1−α) · local_norm,
                            where α is chosen by AUROC grid-search on the val split.
      4. Report the optimal α for each configuration.
    """
    device            = get_device()
    cache             = EmbeddingCache(args.cache_dir)
    mean_pooled_cache = EmbeddingCache(Path(args.cache_dir) / "mean_pooled")
    results           = {}

    bmad_datasets     = [d for d in args.datasets if d != "MLL23"]
    fusion_extractors = [e for e in args.extractors if e in FUSION_GLOBAL]

    if not bmad_datasets:
        print("  No BMAD datasets selected. Skipping Experiment 3.")
        return results
    if not fusion_extractors:
        print("  No fusion-compatible extractors found. Skipping Experiment 3.")
        return results

    print(f"\n{'='*60}")
    print("Experiment 3 — Global + local score fusion  (BMAD only)")
    print(f"Extractors : {fusion_extractors}")
    print(f"Global map : { {e: FUSION_GLOBAL[e] for e in fusion_extractors} }")
    print(f"Alpha grid : 0.00 – 1.00  (step 0.05, optimised on val AUROC)")

    for dataset_name in bmad_datasets:
        print(f"\n  Dataset: {dataset_name}")

        for ext_name in fusion_extractors:
            global_ext_name = FUSION_GLOBAL[ext_name]
            print(f"    Extractor: {ext_name}  (global: {global_ext_name})")

            # ----------------------------------------------------------------
            # Patch embeddings
            # ----------------------------------------------------------------
            if not cache.patches_exist(ext_name, dataset_name, "train"):
                print(f"      [SKIP] No cached patch embeddings. Run run_benchmark.py first.")
                continue
            train_patches = cache.load_patches(ext_name, dataset_name, "train")
            test_patches  = cache.load_patches(ext_name, dataset_name, "test")
            test_labels   = cache.load_labels(ext_name, dataset_name, "test")
            val_patches   = cache.load_patches(ext_name, dataset_name, "valid")
            val_labels    = cache.load_labels(ext_name, dataset_name, "valid")

            # ----------------------------------------------------------------
            # CLS embeddings — train and test
            # ----------------------------------------------------------------
            if not mean_pooled_cache.exists(global_ext_name, dataset_name, "train"):
                print(f"      [SKIP] No cached CLS embeddings. Run --mean-pooled first.")
                continue
            train_cls, _ = mean_pooled_cache.load(global_ext_name, dataset_name, "train")
            test_cls,  _ = mean_pooled_cache.load(global_ext_name, dataset_name, "test")

            # ----------------------------------------------------------------
            # Val CLS — extract and cache on demand
            # ----------------------------------------------------------------
            if mean_pooled_cache.exists(global_ext_name, dataset_name, "valid"):
                val_cls, _ = mean_pooled_cache.load(global_ext_name, dataset_name, "valid")
            else:
                print(f"      Extracting val CLS embeddings ({global_ext_name})...")
                global_extractor = EXTRACTORS[global_ext_name]()
                loader = get_dataloader(
                    args.data_root, dataset_name, "valid",
                    image_size=global_extractor.image_size,
                    patch_size=global_extractor.patch_size,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                val_cls = global_extractor.extract(loader, device=device)
                mean_pooled_cache.save(val_cls, val_labels, global_ext_name, dataset_name, "valid")

            # Normal val samples → z-score normalization reference
            normal_mask = val_labels == 0
            ref_patches = val_patches[normal_mask]
            ref_cls     = val_cls[normal_mask]

            # ----------------------------------------------------------------
            # Per-scorer fusion
            # ----------------------------------------------------------------
            for scorer_name, (scorer_cls, params) in MEAN_POOLED_SCORERS.items():
                print(f"      {scorer_name}...", end=" ", flush=True)
                try:
                    local_scorer  = scorer_cls(**params)
                    global_scorer = scorer_cls(**params)
                    local_scorer.fit(train_patches)
                    global_scorer.fit(train_cls)

                    # Normalization reference: normal val samples only
                    ref_local  = local_scorer.score_patches(train_patches, ref_patches).max(axis=1)
                    ref_global = global_scorer.score(ref_cls)

                    # Score val and test sets
                    val_local_raw   = local_scorer.score_patches(train_patches, val_patches).max(axis=1)
                    val_global_raw  = global_scorer.score(val_cls)
                    test_local_raw  = local_scorer.score_patches(train_patches, test_patches).max(axis=1)
                    test_global_raw = global_scorer.score(test_cls)

                    # Z-normalise
                    val_local_norm   = _z_normalize(ref_local,  val_local_raw)
                    val_global_norm  = _z_normalize(ref_global, val_global_raw)
                    test_local_norm  = _z_normalize(ref_local,  test_local_raw)
                    test_global_norm = _z_normalize(ref_global, test_global_raw)

                    # -- Baseline: element-wise max --
                    max_scores  = np.maximum(test_global_norm, test_local_norm)
                    max_metrics = evaluate(max_scores, test_labels)

                    # -- Weighted: alpha from val AUROC grid search --
                    best_alpha, _ = _find_best_alpha(val_global_norm, val_local_norm, val_labels)
                    val_weighted  = best_alpha * val_global_norm + (1.0 - best_alpha) * val_local_norm
                    threshold     = find_best_f1_threshold(val_weighted, val_labels)

                    weighted_scores  = best_alpha * test_global_norm + (1.0 - best_alpha) * test_local_norm
                    weighted_metrics = evaluate(weighted_scores, test_labels, threshold=threshold)
                    weighted_metrics["best_alpha"] = round(float(best_alpha), 2)

                    results[(dataset_name, ext_name, f"fusion_max/{scorer_name}")]      = {
                        "params": params, "metrics": max_metrics,
                    }
                    results[(dataset_name, ext_name, f"fusion_weighted/{scorer_name}")] = {
                        "params": params, "metrics": weighted_metrics,
                    }

                    print(
                        f"max_AUROC={max_metrics['AUROC']:.4f}  "
                        f"α*={best_alpha:.2f}  "
                        f"weighted_AUROC={weighted_metrics['AUROC']:.4f}"
                    )

                except Exception as exc:
                    print(f"FAILED: {exc}")
                    results[(dataset_name, ext_name, f"fusion_max/{scorer_name}")]      = {
                        "params": params, "metrics": {"error": str(exc)},
                    }
                    results[(dataset_name, ext_name, f"fusion_weighted/{scorer_name}")] = {
                        "params": params, "metrics": {"error": str(exc)},
                    }

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def summarize(results: dict) -> str:
    if not results:
        return "No results."

    rows = []
    for (dataset, extractor, label), entry in results.items():
        row = {
            "Dataset":   dataset,
            "Extractor": extractor,
            "Config":    label,
            **entry["metrics"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["Dataset", "Extractor", "AUROC"], ascending=[True, True, False])

    quality_cols = ["Dataset", "Extractor", "Config", "AUROC", "AUPRC", "F1", "Precision", "Recall", "best_alpha"]
    runtime_cols = ["fit_time_s", "score_time_s", "fit_peak_mb", "score_peak_mb"]
    df = df[[c for c in quality_cols + runtime_cols if c in df.columns]]

    return df.to_string(index=False)


def save_results(results: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {" | ".join(k): v for k, v in results.items()}
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out}")


def parse_args() -> argparse.Namespace:
    valid_experiments = list(EXPERIMENTS.keys())
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep + mean-pooled ablation for anomaly detection scorers"
    )
    parser.add_argument("--data-root", default="./Datasets")
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS, metavar="DATASET",
    )
    parser.add_argument(
        "--extractors", nargs="+",
        default=["dinov2", "dinov3"],   # dinov3_cls_reg is mean-pooled only
        choices=list(EXTRACTORS),
        metavar="EXTRACTOR",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=[],
        choices=valid_experiments, metavar="EXPERIMENT",
        help=f"Scorer hyperparam sweep to run (Part 1). Omit to skip. Choices: {valid_experiments}",
    )
    parser.add_argument(
        "--mean-pooled", action="store_true",
        help="Run global representation comparison (Experiment 2). Requires GPU for extraction.",
    )
    parser.add_argument(
        "--fusion", action="store_true",
        help="Run global+local score fusion (Experiment 3). BMAD datasets only. "
             "Requires --mean-pooled to have been run first.",
    )
    parser.add_argument("--cache-dir",  default="./cache")
    parser.add_argument("--output",     default="./results/hyperparam_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_results = {}

    if args.experiments:
        scorer_results = run_scorer_experiments(args)
        all_results.update(scorer_results)

    if args.mean_pooled:
        mean_pooled_results = run_mean_pooled_experiments(args)
        all_results.update(mean_pooled_results)

    if args.fusion:
        fusion_results = run_fusion_experiments(args)
        all_results.update(fusion_results)

    if all_results:
        print("\n" + summarize(all_results))
        save_results(all_results, args.output)
    else:
        print("No results produced.")