"""Macro/per-class AUPRC and calibration metrics (Brier score, expected
calibration error, reliability curves) for the JBHI clinical-metrics
completion package (docs/jbhi_paper_plan.md section 5).

All of this is purely additive: nothing in `src/evaluation/metrics.py` is
modified, and every function here is DESCRIPTIVE ONLY -- computed once on
already-frozen predictions, never used to re-tune a threshold, re-select a
policy, or otherwise revise any decision made before these predictions were
produced (docs/manuscript_claims_lock.md's existing discipline, extended to
these new metrics).

**Calibration binning rule, pre-specified before any real predictions were
computed with this module** (fixed here, not chosen after looking at
results): Expected Calibration Error and reliability curves both use
`n_bins=10`, EQUAL-WIDTH bins over [0, 1] (the standard ECE convention, e.g.
Guo et al. 2017), never adaptive/quantile binning. This choice is not
revised anywhere downstream of this file.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.labels import SUPERCLASSES

DEFAULT_N_BINS = 10


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_true, y_prob: [N, C]. Macro + per-class average precision (AUPRC).

    Mirrors src.evaluation.metrics.compute_metrics' convention: a class with
    no positive examples has undefined AUPRC and is reported as NaN, not
    silently skipped or defaulted.
    """
    per_class_auprc = {}
    valid = []
    for i, cls in enumerate(SUPERCLASSES):
        y_t = y_true[:, i]
        if y_t.min() == y_t.max():
            auprc = float("nan")
        else:
            auprc = float(average_precision_score(y_t, y_prob[:, i]))
            valid.append(auprc)
        per_class_auprc[cls] = auprc
    macro_auprc = float(np.mean(valid)) if valid else float("nan")
    return {
        "macro_auprc": macro_auprc,
        "per_class_auprc": per_class_auprc,
        "n_valid_auprc_classes": len(valid),
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_true, y_prob: [N, C]. Standard multiclass/multilabel Brier score,
    computed independently per class as mean((p - y)^2) -- well-defined for
    every class regardless of class balance (unlike AUROC/AUPRC, it needs no
    positive-and-negative-example guard).
    """
    per_class = {}
    for i, cls in enumerate(SUPERCLASSES):
        per_class[cls] = float(np.mean((y_prob[:, i] - y_true[:, i]) ** 2))
    macro = float(np.mean(list(per_class.values())))
    return {"macro_brier": macro, "per_class_brier": per_class}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = DEFAULT_N_BINS) -> dict:
    """y_true, y_prob: [N, C]. Standard ECE: |mean predicted prob - empirical
    positive fraction| within each of `n_bins` EQUAL-WIDTH bins over [0,1],
    weighted by bin occupancy, computed independently per class then
    macro-averaged. Fixed n_bins=10 per this module's pre-specified binning
    rule (top of file) -- never adapted per class or per condition.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    per_class = {}
    for i, cls in enumerate(SUPERCLASSES):
        p = y_prob[:, i]
        y = y_true[:, i]
        n = len(p)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bins - 1:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            conf = float(p[mask].mean())
            acc = float(y[mask].mean())
            ece += (cnt / n) * abs(acc - conf)
        per_class[cls] = float(ece)
    macro = float(np.mean(list(per_class.values())))
    return {"macro_ece": macro, "per_class_ece": per_class, "n_bins": n_bins}


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray,
                       n_bins: int = DEFAULT_N_BINS) -> dict:
    """y_true, y_prob: [N, C]. Per-class AND pooled-across-classes reliability
    curve data (bin edges, mean predicted probability, empirical positive
    fraction, and count, per bin) for plotting reliability diagrams. Pooling
    treats every (record, class) pair as one point -- since prevalence
    differs sharply by class (NORM ~44% vs HYP ~12%), the pooled curve is a
    descriptive overall-calibration view, not a substitute for the per-class
    curves, both are returned. Empty bins report count=0 and NaN for mean
    predicted/empirical fraction rather than being silently dropped.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    def _curve(p: np.ndarray, y: np.ndarray) -> dict:
        mean_pred, emp_frac, counts = [], [], []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p <= hi) if b == n_bins - 1 else (p >= lo) & (p < hi)
            cnt = int(mask.sum())
            counts.append(cnt)
            if cnt == 0:
                mean_pred.append(float("nan"))
                emp_frac.append(float("nan"))
            else:
                mean_pred.append(float(p[mask].mean()))
                emp_frac.append(float(y[mask].mean()))
        return {"mean_predicted": mean_pred, "empirical_fraction": emp_frac, "count": counts}

    per_class = {}
    for i, cls in enumerate(SUPERCLASSES):
        per_class[cls] = _curve(y_prob[:, i], y_true[:, i])

    pooled = _curve(y_prob.reshape(-1), y_true.reshape(-1))

    return {
        "n_bins": n_bins,
        "bin_edges": edges.tolist(),
        "bin_centers": bin_centers,
        "per_class": per_class,
        "pooled": pooled,
    }


def compute_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = DEFAULT_N_BINS) -> dict:
    """Convenience bundle: AUPRC + Brier + ECE (+ reliability curve data) in
    one call, used by the post-hoc reporting scripts so every caller applies
    the identical, pre-specified protocol.
    """
    out = {}
    out.update(compute_auprc(y_true, y_prob))
    out.update(brier_score(y_true, y_prob))
    out.update(expected_calibration_error(y_true, y_prob, n_bins=n_bins))
    out["reliability_curve"] = reliability_curve(y_true, y_prob, n_bins=n_bins)
    return out
