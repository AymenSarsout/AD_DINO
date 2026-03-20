import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict:
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Labels must contain both normal (0) and anomalous (1) samples.")

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # Find threshold that maximises F1
    thresholds = np.percentile(scores, np.linspace(0, 100, 300))
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds = (scores >= best_thresh).astype(int)
    return {
        "AUROC": round(float(auroc), 4),
        "AUPRC": round(float(auprc), 4),
        "F1": round(float(best_f1), 4),
        "Precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
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

    quality_cols = ["Dataset", "Extractor", "Scorer", "AUROC", "AUPRC", "F1", "Precision", "Recall"]
    runtime_cols = ["fit_time_s", "score_time_s", "fit_peak_mb", "score_peak_mb"]
    ordered = quality_cols + [c for c in runtime_cols if c in df.columns]
    df = df[[c for c in ordered if c in df.columns]]

    return df.to_string(index=False)
