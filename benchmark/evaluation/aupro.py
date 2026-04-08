"""AUPRO metric and spatial anomaly-map utilities."""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label as sk_label

try:
    from scipy.integrate import trapezoid
except ImportError:  # scipy < 1.7
    from scipy.integrate import trapz as trapezoid  # type: ignore[no-redef]


def patch_scores_to_maps(
    patch_scores: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    N, n_patches = patch_scores.shape
    patch_grid = int(round(n_patches ** 0.5))
    maps_grid = patch_scores.reshape(N, patch_grid, patch_grid)
    return np.stack([get_anomaly_map(m, target_size=image_size) for m in maps_grid])


def get_anomaly_map(
    patch_scores: np.ndarray,
    target_size: int = 224,
) -> np.ndarray:
    tensor = torch.from_numpy(patch_scores.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        tensor, size=(target_size, target_size),
        mode="bilinear", align_corners=False
    )
    return upsampled.squeeze(0).squeeze(0).numpy()


def compute_aupro(
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
    fpr_limit: float = 0.3,
) -> float:
    assert anomaly_maps.shape == masks.shape, (
        f"Shape mismatch: anomaly_maps {anomaly_maps.shape} vs masks {masks.shape}"
    )
    N, H, W = anomaly_maps.shape

    comp_sorted_scores: list[np.ndarray] = []
    for i in range(N):
        labeled = sk_label(masks[i], connectivity=2)
        n_comps = int(labeled.max())
        for comp_id in range(1, n_comps + 1):
            comp_pixels = anomaly_maps[i][labeled == comp_id]
            comp_sorted_scores.append(np.sort(comp_pixels))

    if not comp_sorted_scores:
        return 0.0

    normal_mask = masks == 0
    n_normal = int(normal_mask.sum())
    if n_normal == 0:
        return 0.0
    normal_sorted = np.sort(anomaly_maps[normal_mask])  # ascending

    num_thresholds = 200
    positions = np.linspace(0, len(normal_sorted) - 1, num=num_thresholds, dtype=int)
    thresholds = np.unique(normal_sorted[positions])

    fprs = (n_normal - np.searchsorted(normal_sorted, thresholds)) / n_normal

    n_thresh = len(thresholds)
    all_overlaps = np.empty((len(comp_sorted_scores), n_thresh), dtype=np.float64)
    for c_idx, sorted_scores in enumerate(comp_sorted_scores):
        n_c = len(sorted_scores)
        all_overlaps[c_idx] = (n_c - np.searchsorted(sorted_scores, thresholds)) / n_c
    pros = all_overlaps.mean(axis=0)

    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pros = pros[sort_idx]

    _, unique_idx = np.unique(fprs, return_index=True)
    fprs = fprs[unique_idx]
    pros = pros[unique_idx]

    if fprs[0] > 0.0:
        fprs = np.concatenate([[0.0], fprs])
        pros = np.concatenate([[0.0], pros])
    within = fprs <= fpr_limit
    if not within.any():
        return 0.0

    fprs_clip = fprs[within]
    pros_clip = pros[within]

    if fprs_clip[-1] < fpr_limit:
        pro_at_limit = float(np.interp(fpr_limit, fprs, pros))
        fprs_clip = np.append(fprs_clip, fpr_limit)
        pros_clip = np.append(pros_clip, pro_at_limit)

    aupro = float(trapezoid(pros_clip, fprs_clip)) / fpr_limit
    return round(float(np.clip(aupro, 0.0, 1.0)), 4)