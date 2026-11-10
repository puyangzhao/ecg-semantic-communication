"""Shared statistics for the revision's formal evaluation.

* auroc / auprc: tie-aware (average-rank Mann-Whitney AUROC; sklearn-identical step-wise average
  precision with tied scores grouped), verified against sklearn in __main__.
* patient_draws: ONE set of patient-level bootstrap resamples (all records of a drawn patient kept)
  shared by every method, seed and comparison that uses the same cohort.
* paired_boot: per resample, the metric difference is computed per seed and then averaged over seeds;
  percentile 95% interval; two-sided bootstrap p = min(1, 2*min(P(d<=0), P(d>=0))). Intervals are
  conditional on the trained models (no retraining inside the bootstrap).
* holm: Holm-Bonferroni adjusted p-values.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y, s):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # last index of each tied group
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def class_metrics(y, p):
    """y [n, C], p [n, C] -> dict of arrays [C] auroc, auprc"""
    return {"auroc": np.array([auroc(y[:, c], p[:, c]) for c in range(y.shape[1])]),
            "auprc": np.array([auprc(y[:, c], p[:, c]) for c in range(y.shape[1])])}


def patient_draws(pid, B=2000, seed=20261110):
    rng = np.random.default_rng(seed)
    pats = np.unique(pid)
    groups = {p: np.flatnonzero(pid == p) for p in pats}
    return [np.concatenate([groups[p] for p in rng.choice(pats, size=len(pats), replace=True)]) for _ in range(B)]


def record_draws(n, B=2000, seed=20261111):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def boot_p(d):
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


if __name__ == "__main__":
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = rng.integers(20, 400)
        y = rng.integers(0, 2, n)
        s = rng.integers(0, rng.integers(2, 30), n).astype(float) if trial % 2 else rng.random(n)
        if y.min() == y.max():
            continue
        assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12
        assert abs(auprc(y, s) - average_precision_score(y, s)) < 1e-12, (auprc(y, s), average_precision_score(y, s))
    print("tie-aware auroc/auprc identical to sklearn on 200 random (tied and untied) cases")
