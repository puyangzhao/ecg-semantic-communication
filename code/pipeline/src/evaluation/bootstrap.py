"""Paired, patient-level bootstrap CI for the macro-AUROC difference between
two methods (docs/phase2a_design.md section 8).

Resamples patient IDs (not individual records) with replacement, so patients
with more than one PTB-XL record are not treated as independent samples.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.calibration import compute_auprc
from src.evaluation.hierarchical_bootstrap import _vectorized_macro_auprc_batch
from src.evaluation.metrics import compute_metrics


def _rank_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Fast, pure-numpy macro-AUROC via the Mann-Whitney rank-sum identity,
    used only inside bootstrap loops (thousands of calls) to avoid sklearn's
    per-call overhead. Same NaN-skip-on-constant-class convention as
    `compute_metrics`, and verified to match it exactly on real data in
    tests/test_bootstrap.py (this is the same validated rank trick already
    used in src/evaluation/hierarchical_bootstrap.py's
    `_vectorized_macro_auroc_batch` for Phase 2B).
    """
    n, c = y_true.shape
    aurocs = []
    for cls in range(c):
        y_c = y_true[:, cls]
        if y_c.min() == y_c.max():
            continue
        ranks = np.argsort(np.argsort(y_prob[:, cls])) + 1
        pos_mask = y_c == 1
        n_pos = int(pos_mask.sum())
        n_neg = n - n_pos
        rank_sum_pos = ranks[pos_mask].sum()
        aurocs.append((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def paired_bootstrap_auroc_diff(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                                 patient_ids: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """y_true, prob_a, prob_b: [N, C], aligned to the same records.
    prob_a is "method A" (e.g. proposed), prob_b is "method B" (e.g. strongest
    baseline); mean_diff = mean(macro_auroc_a - macro_auroc_b) over resamples.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_to_idx[p] for p in sampled])
        auroc_a = compute_metrics(y_true[idx], prob_a[idx])["macro_auroc"]
        auroc_b = compute_metrics(y_true[idx], prob_b[idx])["macro_auroc"]
        if not (np.isnan(auroc_a) or np.isnan(auroc_b)):
            diffs.append(auroc_a - auroc_b)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"n_boot_valid": 0, "n_boot_requested": n_boot, "mean_diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "n_boot_valid": int(len(diffs)),
        "n_boot_requested": n_boot,
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def patient_level_paired_bootstrap(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                                    patient_ids: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Corrected single-seed, single-level patient bootstrap (see
    docs/phase2a_estimand_audit.md). Purely additive -- does not modify or
    replace `paired_bootstrap_auroc_diff`, which is preserved unchanged as
    the historical Phase 2A record.

    Fixes two issues found in `paired_bootstrap_auroc_diff`'s use in
    `scripts/aggregate_phase2a.py` (never in this function's own logic):
    (1) the caller averaged y_prob across the 3 independently-trained seeds
    before calling it, collapsing the seed dimension into one array
    (estimand B: AUROC-of-averaged-predictions) instead of reporting each
    seed separately; (2) the returned "mean_diff" is the MEAN OF THE
    BOOTSTRAP RESAMPLE DISTRIBUTION, not the observed statistic computed
    once on the real, unresampled data -- these are close but not
    identical (verified empirically in the audit), so the point estimate
    and the CI it is reported alongside were not guaranteed to target the
    same quantity.

    This function is meant to be called ONCE PER SEED (y_true/prob_a/prob_b/
    patient_ids all from that seed's own predictions only -- no seed
    averaging), matching the corrected-estimand discipline already
    established in `src/evaluation/hierarchical_bootstrap.py` for Phase 2B:
    the "observed_statistic" field is computed directly on the real data
    (not derived from the bootstrap distribution), and is exactly what the
    CI is built around.

    No channel-realization dimension exists in Phase 2A (unlike Phase 2B's
    Monte Carlo channel draws), so only ONE level of resampling (patients)
    is used here -- deliberately not a hierarchical/two-level bootstrap.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    auroc_a_observed = compute_metrics(y_true, prob_a)["macro_auroc"]
    auroc_b_observed = compute_metrics(y_true, prob_b)["macro_auroc"]
    observed_statistic = auroc_a_observed - auroc_b_observed

    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_to_idx[p] for p in sampled])
        auroc_a = _rank_macro_auroc(y_true[idx], prob_a[idx])
        auroc_b = _rank_macro_auroc(y_true[idx], prob_b[idx])
        if not (np.isnan(auroc_a) or np.isnan(auroc_b)):
            diffs.append(auroc_a - auroc_b)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"n_boot_valid": 0, "n_boot_requested": n_boot, "observed_statistic": observed_statistic,
                "auroc_a_observed": auroc_a_observed, "auroc_b_observed": auroc_b_observed,
                "percentile_ci": [float("nan"), float("nan")]}
    return {
        "n_boot_valid": int(len(diffs)),
        "n_boot_requested": n_boot,
        "observed_statistic": observed_statistic,
        "auroc_a_observed": auroc_a_observed,
        "auroc_b_observed": auroc_b_observed,
        "bootstrap_mean": float(np.mean(diffs)),
        "percentile_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def _rank_macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Fast, pure-numpy macro-AUPRC (average precision), used only inside
    bootstrap loops (thousands of calls) to avoid sklearn's per-call
    overhead. Thin single-array wrapper around
    src.evaluation.hierarchical_bootstrap._vectorized_macro_auprc_batch
    (batch size 1) rather than a separate re-derivation, specifically so the
    TIE-GROUP-AWARE correctness fix documented there (patient-level
    bootstrap resampling routinely creates exact score ties via duplicated
    rows, which a naive per-example formula handles incorrectly relative to
    sklearn) is not duplicated -- and cannot silently drift out of sync --
    across the two bootstrap modules. Verified to match
    sklearn.metrics.average_precision_score, including under deliberately
    tie-heavy resampled data, in tests/test_bootstrap.py.
    """
    return float(_vectorized_macro_auprc_batch(y_true, y_prob[None, :, :])[0])


def patient_level_paired_bootstrap_auprc(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                                          patient_ids: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUPRC counterpart of `patient_level_paired_bootstrap`, identical
    patient-level resampling discipline and identical observed-statistic/CI
    consistency guarantee (the point estimate is computed directly on the
    real data, not derived from the bootstrap distribution). Purely
    additive -- does not modify or replace the AUROC version. Called once per
    seed, same as `patient_level_paired_bootstrap`; no cross-seed pooling.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    auprc_a_observed = compute_auprc(y_true, prob_a)["macro_auprc"]
    auprc_b_observed = compute_auprc(y_true, prob_b)["macro_auprc"]
    observed_statistic = auprc_a_observed - auprc_b_observed

    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_to_idx[p] for p in sampled])
        auprc_a = _rank_macro_auprc(y_true[idx], prob_a[idx])
        auprc_b = _rank_macro_auprc(y_true[idx], prob_b[idx])
        if not (np.isnan(auprc_a) or np.isnan(auprc_b)):
            diffs.append(auprc_a - auprc_b)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"n_boot_valid": 0, "n_boot_requested": n_boot, "observed_statistic": observed_statistic,
                "auprc_a_observed": auprc_a_observed, "auprc_b_observed": auprc_b_observed,
                "percentile_ci": [float("nan"), float("nan")]}
    return {
        "n_boot_valid": int(len(diffs)),
        "n_boot_requested": n_boot,
        "observed_statistic": observed_statistic,
        "auprc_a_observed": auprc_a_observed,
        "auprc_b_observed": auprc_b_observed,
        "bootstrap_mean": float(np.mean(diffs)),
        "percentile_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }
