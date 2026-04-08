import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_pixel_auroc(anomaly_maps: np.ndarray, masks: np.ndarray) -> float:
    scores = anomaly_maps.flatten().astype(np.float64)
    labels = masks.flatten().astype(np.int32)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    return round(float(roc_auc_score(labels, scores)), 4)


def find_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the score threshold that maximizes F1 on the given split."""
    best_f1, best_thresh = 0.0, float(np.unique(scores)[0])
    for t in np.unique(scores):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    return best_thresh


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Compute all metrics for a test split.

    To add a new metric:
    1. Compute it from `scores` / `labels` / `preds` below.
    2. Add it to the returned dict with a string key (e.g. "AUROC").
    3. Add the key to the `quality_cols` list in summarize_results() so it appears in the output table.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Labels must contain both normal (0) and anomalous (1) samples.")

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    if threshold is None:
        threshold = find_best_f1_threshold(scores, labels)

    preds = (scores >= threshold).astype(int)
    return {
        "AUROC":     round(float(auroc), 4),
        "AUPRC":     round(float(auprc), 4),
        "Threshold": round(float(threshold), 6),
        "F1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "Precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "Recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
    }


def summarize_results(results: dict) -> str:
    rows = []
    for dataset, extractors in results.items():
        for extractor, scorers in extractors.items():
            for scorer, metrics in scorers.items():
                rows.append({
                    "Dataset": dataset,
                    "Extractor": extractor,
                    "Scorer": scorer,
                    **metrics,
                })

    if not rows:
        return "No results."

    df = pd.DataFrame(rows)
    df = df.sort_values(["Dataset", "Extractor", "AUROC"], ascending=[True, True, False])

    quality_cols = ["Dataset", "Extractor", "Scorer", "AUROC", "AUPRC", "pixel_AUROC", "AUPRO", "F1", "Precision", "Recall"]
    runtime_cols = ["fit_time_s", "score_time_s", "fit_peak_mb", "score_peak_mb"]
    ordered = quality_cols + [c for c in runtime_cols if c in df.columns]
    df = df[[c for c in ordered if c in df.columns]]

    return df.to_string(index=False)
