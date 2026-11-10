"""Selector collapse / interpretability diagnostics (docs/phase2a_design.md
section 10). Computed for every method, including Random-K and
Fixed-clinical-K as built-in null/negative controls.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.data.clinical_lead_subsets import LEAD_NAMES
from src.data.labels import SUPERCLASSES


def selection_frequency(masks: np.ndarray) -> dict:
    """masks: [N, 12] binary. Returns {lead_name: fraction_selected}."""
    freq = masks.mean(axis=0)
    return {name: float(f) for name, f in zip(LEAD_NAMES, freq)}


def selection_entropy(masks: np.ndarray) -> float:
    """Normalized Shannon entropy (0 = always the same leads, 1 = maximally
    uniform use of all 12 leads) of the per-lead selection-frequency
    distribution, treated as a distribution over leads by normalizing the
    12 frequencies to sum to 1.
    """
    freq = masks.mean(axis=0)
    total = freq.sum()
    if total <= 0:
        return float("nan")
    p = freq / total
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    max_entropy = np.log(len(freq))
    return float(entropy / max_entropy) if max_entropy > 0 else float("nan")


def pairwise_co_selection(masks: np.ndarray) -> list:
    """masks: [N, 12] -> [12, 12] fraction of samples where both leads are selected."""
    n = masks.shape[0]
    co = (masks.T @ masks) / max(n, 1)
    return co.tolist()


def seed_stability_jaccard(masks_a: np.ndarray, masks_b: np.ndarray) -> float:
    """Mean per-sample Jaccard overlap between selected-lead sets from two
    seeds on the same test records (sample-specific selectors, e.g. proposed).
    """
    inter = (masks_a * masks_b).sum(axis=1)
    union = ((masks_a + masks_b) > 0).sum(axis=1)
    union = np.clip(union, 1, None)
    return float(np.mean(inter / union))


def seed_stability_spearman(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Spearman rank correlation between two seeds' global score vectors
    (Global-learned-K: one 12-length ranking per seed).
    """
    rho, _ = spearmanr(scores_a, scores_b)
    return float(rho)


def selection_by_label(masks: np.ndarray, labels: np.ndarray) -> dict:
    """masks: [N, 12], labels: [N, 5] multi-hot. Returns
    {class_name: {lead_name: mean_selection_frequency_among_positives}}.
    """
    result = {}
    for i, cls in enumerate(SUPERCLASSES):
        pos_idx = labels[:, i] == 1
        if pos_idx.sum() == 0:
            result[cls] = {name: float("nan") for name in LEAD_NAMES}
        else:
            result[cls] = selection_frequency(masks[pos_idx])
    return result
