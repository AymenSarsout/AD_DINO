import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.datasets.loader import DATASETS, get_dataloader
from benchmark.evaluation.metrics import evaluate
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

EXPERIMENTS: dict[str, list[tuple]] = {
    "knn": [
        (KNNScorer, {"k": 1,  "aggregation": "mean"}),
        (KNNScorer, {"k": 3,  "aggregation": "mean"}),
        (KNNScorer, {"k": 5,  "aggregation": "mean"}),   # default
        (KNNScorer, {"k": 10, "aggregation": "mean"}),
        (KNNScorer, {"k": 20, "aggregation": "mean"}),
        (KNNScorer, {"k": 50, "aggregation": "mean"}),
        (KNNScorer, {"k": 5,  "aggregation": "max"}),
        (KNNScorer, {"k": 10, "aggregation": "max"}),
        (KNNScorer, {"k": 20, "aggregation": "max"}),
    ],
    "kmeans": [
        (KMeansScorer, {"n_clusters": 8}),
        (KMeansScorer, {"n_clusters": 16}),
        (KMeansScorer, {"n_clusters": 32}),   # default
        (KMeansScorer, {"n_clusters": 64}),
        (KMeansScorer, {"n_clusters": 128}),
    ],
    "mahalanobis": [
        (MahalanobisScorer, {"reg": 1e-6}),
        (MahalanobisScorer, {"reg": 1e-5}),   # default
        (MahalanobisScorer, {"reg": 1e-4}),
        (MahalanobisScorer, {"reg": 1e-3}),
        (MahalanobisScorer, {"reg": 1e-2}),
    ],
    "memory_bank": [
        # Sweep coreset_ratio with k=1
        (MemoryBankScorer, {"coreset_ratio": 0.01, "k": 1}),
        (MemoryBankScorer, {"coreset_ratio": 0.05, "k": 1}),
        (MemoryBankScorer, {"coreset_ratio": 0.10, "k": 1}),   # default
        (MemoryBankScorer, {"coreset_ratio": 0.25, "k": 1}),
        (MemoryBankScorer, {"coreset_ratio": 0.50, "k": 1}),
        # Sweep k with default coreset_ratio
        (MemoryBankScorer, {"coreset_ratio": 0.10, "k": 3}),
        (MemoryBankScorer, {"coreset_ratio": 0.10, "k": 5}),
    ],
    # normalization requires re-extraction — uses a separate cache namespace
    "normalization": [
        (KNNScorer,         {"k": 5, "aggregation": "mean"}),
        (MahalanobisScorer, {"reg": 1e-5}),
    ],
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


def _fit_and_score(
    scorer_cls,
    params: dict,
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Fit scorer, score test set, evaluate metrics + timing + memory."""
    scorer = scorer_cls(**params)

    tracemalloc.start()
    t0 = time.perf_counter()
    scorer.fit(train_emb)
    fit_time = time.perf_counter() - t0
    _, fit_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    scores = scorer.score(test_emb)
    score_time = time.perf_counter() - t0
    _, score_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = evaluate(scores, test_labels)
    metrics["fit_time_s"] = round(fit_time, 4)
    metrics["score_time_s"] = round(score_time, 4)
    metrics["fit_peak_mb"] = round(fit_peak / 1024 ** 2, 2)
    metrics["score_peak_mb"] = round(score_peak / 1024 ** 2, 2)
    return metrics


def run_scorer_experiments(args) -> dict:
    cache = EmbeddingCache(args.cache_dir)
    results = {}

    experiment_names = [e for e in args.experiments if e != "normalization"]
    if not experiment_names:
        return results

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")

        for ext_name in args.extractors:
            print(f"  Extractor: {ext_name}")

            if not cache.exists(ext_name, dataset_name, "train"):
                print(f"    [SKIP] No cached embeddings for {dataset_name}/{ext_name}. "
                      f"Run run_benchmark.py first.")
                continue

            train_emb, _ = cache.load(ext_name, dataset_name, "train")
            test_emb, test_labels = cache.load(ext_name, dataset_name, "test")

            for exp_name in experiment_names:
                for scorer_cls, params in EXPERIMENTS[exp_name]:
                    label = _experiment_label(scorer_cls, params)
                    key = (dataset_name, ext_name, label)

                    print(f"    {label}...", end=" ", flush=True)
                    metrics = _fit_and_score(scorer_cls, params, train_emb, test_emb, test_labels)
                    results[key] = {"params": params, "metrics": metrics}

                    print(
                        f"AUROC={metrics['AUROC']:.4f}  F1={metrics['F1']:.4f}"
                        f"  fit={metrics['fit_time_s']:.3f}s  score={metrics['score_time_s']:.3f}s"
                        f"  fit_mem={metrics['fit_peak_mb']:.1f}MB"
                    )

    return results



def run_normalization_experiments(args) -> dict:
    device = get_device()
    print(f"\nDevice: {device}")
    results = {}

    norm_cache = EmbeddingCache(Path(args.cache_dir) / "no_norm")

    print(f"\n{'='*60}")
    print("Normalization ablation: no ImageNet normalization")

    for dataset_name in args.datasets:
        for ext_name in args.extractors:
            extractor = EXTRACTORS[ext_name]()

            for split in ("train", "test"):
                if norm_cache.exists(ext_name, dataset_name, split):
                    print(f"  [{ext_name}] Loading cached {split} (no_norm)...")
                    continue
                print(f"  [{ext_name}] Extracting {split} (no_norm)...")
                loader = get_dataloader(
                    args.data_root, dataset_name, split,
                    image_size=extractor.image_size,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    normalize=False,
                )
                embeddings = extractor.extract(loader, device=device)
                labels = loader.dataset.label_array
                norm_cache.save(embeddings, labels, ext_name, dataset_name, split)

            train_emb, _ = norm_cache.load(ext_name, dataset_name, "train")
            test_emb, test_labels = norm_cache.load(ext_name, dataset_name, "test")

            for scorer_cls, params in EXPERIMENTS["normalization"]:
                label = _experiment_label(scorer_cls, params)
                key = (dataset_name, ext_name, f"no_norm/{label}")

                print(f"    no_norm {label}...", end=" ", flush=True)
                metrics = _fit_and_score(scorer_cls, params, train_emb, test_emb, test_labels)
                results[key] = {"normalize": False, "params": params, "metrics": metrics}

                print(f"AUROC={metrics['AUROC']:.4f}  F1={metrics['F1']:.4f}")

    return results


def summarize(results: dict) -> str:
    if not results:
        return "No results."

    rows = []
    for (dataset, extractor, label), entry in results.items():
        row = {
            "Dataset": dataset,
            "Extractor": extractor,
            "Config": label,
            **entry["metrics"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["Dataset", "Extractor", "AUROC"], ascending=[True, True, False])

    quality_cols = ["Dataset", "Extractor", "Config", "AUROC", "AUPRC", "F1", "Precision", "Recall"]
    runtime_cols = ["fit_time_s", "score_time_s", "fit_peak_mb", "score_peak_mb"]
    df = df[[c for c in quality_cols + runtime_cols if c in df.columns]]

    return df.to_string(index=False)


def save_results(results: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Convert tuple keys to strings for JSON serialization
    serializable = {
        " | ".join(k): v for k, v in results.items()
    }
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out}")


def parse_args() -> argparse.Namespace:
    valid_experiments = list(EXPERIMENTS.keys())
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for anomaly detection scorers"
    )
    parser.add_argument(
        "--data-root",
        default="./Datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS, metavar="DATASET",
    )
    parser.add_argument(
        "--extractors", nargs="+", default=list(EXTRACTORS), choices=list(EXTRACTORS),
        metavar="EXTRACTOR",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=[e for e in valid_experiments if e != "normalization"],
        choices=valid_experiments, metavar="EXPERIMENT",
        help=f"Which experiments to run. Choices: {valid_experiments}. "
             f"'normalization' requires GPU for re-extraction.",
    )
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output", default="./results/hyperparam_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_results = {}

    scorer_results = run_scorer_experiments(args)
    all_results.update(scorer_results)

    if "normalization" in args.experiments:
        norm_results = run_normalization_experiments(args)
        all_results.update(norm_results)

    if all_results:
        print("\n" + summarize(all_results))
        save_results(all_results, args.output)
    else:
        print("No results produced.")