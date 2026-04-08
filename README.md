# Medical Imaging Anomaly Detection Benchmark

Unsupervised anomaly detection benchmark for medical imaging.
Evaluates combinations of **feature extractors** × **anomaly scorers** and reports AUROC, AUPRC, F1, Precision, and Recall.

---

## Setup

```bash
pip install -r requirements.txt
```

Place datasets under `./Datasets/` following the BMAD layout:

```
Datasets/<name>/
├── train/good/
├── test/good/
└── test/Ungood/
    └── label/   # optional pixel masks
```

---

## Run the Benchmark

```bash
python run_benchmark.py                                # all datasets, extractors, scorers
python run_benchmark.py --datasets OCT2017 MLL23       # specific datasets
python run_benchmark.py --scorers knn mahalanobis      # specific scorers
python run_benchmark.py --data-root /path/to/Datasets  # custom data root
```

Results are saved to `./results/benchmark_results.json`.

| Argument | Default | Description |
|---|---|---|
| `--data-root` | `./Datasets` | Root directory for datasets |
| `--datasets` | all | Datasets to evaluate |
| `--extractors` | `dinov2 dinov3` | Feature extractors |
| `--scorers` | all | Anomaly scorers |
| `--cache-dir` | `./cache` | Cached embeddings directory |
| `--output` | `./results/benchmark_results.json` | Output path |
| `--batch-size` | `32` | DataLoader batch size |
| `--num-workers` | `4` | DataLoader workers |

---

## Run Experiments

`run_experiments.py` contains three supplementary experiments. Each is opt-in via a flag.

**Experiment 1 — Scorer hyperparameter sweep** (`--experiments`)  
Sweeps k, aggregation, number of clusters, etc. for each scorer over the cached patch embeddings. Run `run_benchmark.py` first to populate the cache.

```bash
python run_experiments.py --experiments knn kmeans mahalanobis memory_bank
```

**Experiment 2 — Global representation comparison** (`--mean-pooled`)  
Compares three global descriptors (CLS token, mean-pooled patches, CLS + register tokens) across all scorers with default hyperparameters. Requires GPU for embedding extraction.

```bash
python run_experiments.py --mean-pooled
```

**Experiment 3 — Global + local score fusion** (`--fusion`)  
Combines image-level (CLS) and patch-level anomaly scores via weighted fusion. Searches for the optimal fusion weight α on the validation split. BMAD datasets only (MLL23 excluded — no val split). Requires Experiment 2 embeddings to be cached first.

```bash
python run_experiments.py --mean-pooled --fusion
```

Results are saved to `./results/hyperparam_results.json`.

All three experiments share the same CLI options as `run_benchmark.py` (`--datasets`, `--extractors`, `--cache-dir`, etc.).

---

## Extending the Benchmark

**Add a dataset** — see `benchmark/datasets/loader.py`

**Add a feature extractor** — see `benchmark/feature_extraction/base.py`

**Add an anomaly scorer** — see `benchmark/scoring/base.py`

**Add a metric** — see `benchmark/evaluation/metrics.py`