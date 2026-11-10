"""Multilabel evaluation metrics: macro/per-class AUROC and F1."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.data.labels import SUPERCLASSES


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """y_true, y_prob: [N, C]. Returns macro + per-class AUROC and F1.

    A class with no positive examples in y_true has undefined AUROC; it is
    reported as NaN rather than silently skipped or defaulted to 0.5, so the
    caller can see the data issue instead of a masked failure.
    """
    y_pred = (y_prob >= threshold).astype(int)

    per_class_auroc = {}
    per_class_f1 = {}
    valid_aurocs = []
    for i, cls in enumerate(SUPERCLASSES):
        y_t = y_true[:, i]
        if y_t.min() == y_t.max():
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y_t, y_prob[:, i])
            valid_aurocs.append(auroc)
        per_class_auroc[cls] = auroc
        per_class_f1[cls] = f1_score(y_t, y_pred[:, i], zero_division=0)

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "per_class_auroc": per_class_auroc,
        "per_class_f1": per_class_f1,
        "n_valid_auroc_classes": len(valid_aurocs),
    }


def select_per_class_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                                 candidates: np.ndarray | None = None) -> dict:
    """Sweeps candidate thresholds independently per class, selecting the one
    maximizing F1 on the given (validation) data. Never call this on test
    data -- see docs/phase2b_design.md section 6.
    """
    if candidates is None:
        candidates = np.arange(0.1, 0.91, 0.05)
    thresholds = {}
    for i, cls in enumerate(SUPERCLASSES):
        y_t = y_true[:, i]
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            y_pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_t, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[cls] = best_t
    return thresholds


def compute_metrics_with_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: dict) -> dict:
    """Like compute_metrics, but with a per-class threshold dict (from
    select_per_class_thresholds) instead of one scalar threshold for all classes.
    """
    per_class_auroc = {}
    per_class_f1 = {}
    valid_aurocs = []
    for i, cls in enumerate(SUPERCLASSES):
        y_t = y_true[:, i]
        if y_t.min() == y_t.max():
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y_t, y_prob[:, i])
            valid_aurocs.append(auroc)
        per_class_auroc[cls] = auroc
        y_pred = (y_prob[:, i] >= thresholds[cls]).astype(int)
        per_class_f1[cls] = f1_score(y_t, y_pred, zero_division=0)

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "per_class_auroc": per_class_auroc,
        "per_class_f1": per_class_f1,
        "n_valid_auroc_classes": len(valid_aurocs),
        "thresholds": thresholds,
    }
