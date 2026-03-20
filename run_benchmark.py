import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from benchmark.datasets.loader import DATASETS, get_dataloader
from benchmark.evaluation.metrics import evaluate, summarize_results
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


def extract_or_load(
    extractor,
    cache: EmbeddingCache,
    data_root: str,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists(extractor.name, dataset_name, split):
        print(f"    [{extractor.name}] Loading cached {split} embeddings...")
        return cache.load(extractor.name, dataset_name, split)

    print(f"    [{extractor.name}] Extracting {split} embeddings...")
    loader = get_dataloader(
        data_root, dataset_name, split,
        image_size=extractor.image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    embeddings = extractor.extract(loader, device=device)
    labels = loader.dataset.label_array
    cache.save(embeddings, labels, extractor.name, dataset_name, split)
    return embeddings, labels


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
            results[dataset_name][ext_name] = {}

            train_emb, _ = extract_or_load(
                extractor, cache, args.data_root, dataset_name, "train",
                args.batch_size, args.num_workers, device,
            )
            test_emb, test_labels = extract_or_load(
                extractor, cache, args.data_root, dataset_name, "test",
                args.batch_size, args.num_workers, device,
            )

            for scorer_name in args.scorers:
                print(f"    [{ext_name}] Scoring with {scorer_name}...", end=" ", flush=True)
                scorer = SCORERS[scorer_name]()

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

                results[dataset_name][ext_name][scorer_name] = metrics
                print(
                    f"AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  F1={metrics['F1']:.4f}"
                    f"  fit={metrics['fit_time_s']:.3f}s  score={metrics['score_time_s']:.3f}s"
                    f"  fit_mem={metrics['fit_peak_mb']:.1f}MB  score_mem={metrics['score_peak_mb']:.1f}MB"
                )

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run(args)
    print("\n" + summarize_results(results))
    save_results(results, args.output)
