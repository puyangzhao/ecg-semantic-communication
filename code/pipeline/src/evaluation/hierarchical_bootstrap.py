"""Hierarchical paired bootstrap for Monte Carlo channel-realization
diagnostics (Phase 2B). Targets EXACTLY the primary estimand:

    Expected channel performance = mean over the n_realizations
    realization-specific patient-level macro-AUROC values.

The observed paired statistic and the bootstrap CI are computed from the
SAME function (`_vectorized_macro_auroc_batch`) so the reported point
estimate is guaranteed, by construction, to equal the value the CI is
centered/derived from -- the mismatch this module corrects. See
docs/phase2b_mechanism_diagnostic.md.

Uses a rank-based (double-argsort) AUROC formula, vectorized across a batch
of realizations/bootstrap iterations at once, since a naive per-realization
sklearn.roc_auc_score loop is too slow for a 2000-iteration x 30-realization
hierarchical bootstrap (~11ms/call, would take 25-90+ minutes; vectorized,
seconds). Verified against src.evaluation.metrics.compute_metrics in
tests/test_hierarchical_bootstrap.py.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_metrics


def _vectorized_macro_auroc_batch(y_true: np.ndarray, probs_batch: np.ndarray) -> np.ndarray:
    """y_true: [N, C] fixed multi-hot labels for the (possibly resampled)
    record set. probs_batch: [B, N, C], B independent probability arrays
    (realizations or bootstrap iterations) for that SAME record set.

    Returns [B] macro-AUROC values, averaged over classes with both a
    positive and a negative example in y_true (matching
    src.evaluation.metrics.compute_metrics' NaN-skip convention -- which
    class is skipped is the same for every b, since y_true doesn't vary
    across the batch dimension).
    """
    b, n, c = probs_batch.shape
    per_class_auc = np.zeros((b, c))
    valid = []
    for cls in range(c):
        y_c = y_true[:, cls]
        if y_c.min() == y_c.max():
            continue
        valid.append(cls)
        scores = probs_batch[:, :, cls]  # [B, N]
        # double-argsort rank trick, vectorized over B; ties are broken
        # arbitrarily-but-consistently rather than averaged (sklearn's
        # convention) -- negligible for continuous sigmoid outputs, verified
        # against compute_metrics in the unit tests.
        ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1) + 1  # [B, N], 1..N
        pos_mask = y_c == 1
        n_pos = int(pos_mask.sum())
        n_neg = n - n_pos
        rank_sum_pos = ranks[:, pos_mask].sum(axis=-1)  # [B]
        per_class_auc[:, cls] = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    if not valid:
        return np.full(b, np.nan)
    return per_class_auc[:, valid].mean(axis=1)


def realization_level_aurocs(y_true: np.ndarray, probs_by_realization: np.ndarray,
                              record_idx: np.ndarray | None = None) -> np.ndarray:
    """probs_by_realization: [n_real, N, C]. Returns [n_real] per-realization
    patient-level macro-AUROC, optionally restricted to `record_idx` (an
    index array into the N records, used for bootstrap resampling)."""
    if record_idx is not None:
        y = y_true[record_idx]
        probs = probs_by_realization[:, record_idx, :]
    else:
        y = y_true
        probs = probs_by_realization
    return _vectorized_macro_auroc_batch(y, probs)


def observed_paired_statistic(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> dict:
    """probs_a, probs_b: [n_real, N, C], CRN-paired by realization index
    (realization j of probs_a and realization j of probs_b share the same
    underlying channel-noise draw). Returns the PRIMARY estimand: mean over
    realization-specific patient-level macro-AUROC(a) - macro-AUROC(b).
    """
    aurocs_a = realization_level_aurocs(y_true, probs_a)
    aurocs_b = realization_level_aurocs(y_true, probs_b)
    diff = aurocs_a - aurocs_b
    return {
        "mean_diff": float(np.mean(diff)),
        "per_realization_diff": diff.tolist(),
        "aurocs_a": aurocs_a.tolist(),
        "aurocs_b": aurocs_b.tolist(),
        "mean_auroc_a": float(np.mean(aurocs_a)),
        "mean_auroc_b": float(np.mean(aurocs_b)),
    }


def hierarchical_paired_bootstrap(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
                                   patient_ids: np.ndarray, n_boot: int = 2000,
                                   resample_realizations: bool = True, seed: int = 0) -> dict:
    """probs_a, probs_b: [n_real, N, C], CRN-paired. patient_ids: [N].

    Each bootstrap iteration: (1) resamples patient IDs once, with
    replacement, and gathers ALL records for the sampled patients (so a
    multi-record patient's records move together -- patient-level, not
    record-level, resampling); (2) optionally resamples the realization
    index set (with replacement) -- the SAME resampled realization indices
    are used for both probs_a and probs_b, preserving the CRN pairing;
    (3) recomputes the mean realization-specific macro-AUROC difference on
    that resampled (patients, realizations) combination -- i.e. exactly the
    same estimand as `observed_paired_statistic`, just on resampled data.

    The `observed_statistic` field in the returned dict is computed by
    `observed_paired_statistic` directly (not derived from the bootstrap
    distribution), so it is guaranteed to be the actual point estimate the
    CI is built around -- see tests/test_hierarchical_bootstrap.py for an
    explicit equality check.
    """
    rng = np.random.default_rng(seed)
    n_real = probs_a.shape[0]
    unique_patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    observed = observed_paired_statistic(y_true, probs_a, probs_b)
    observed_stat = observed["mean_diff"]

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rec_idx = np.concatenate([patient_to_idx[p] for p in sampled_patients])

        real_idx = rng.choice(n_real, size=n_real, replace=True) if resample_realizations else np.arange(n_real)

        aurocs_a = realization_level_aurocs(y_true, probs_a[real_idx], record_idx=rec_idx)
        aurocs_b = realization_level_aurocs(y_true, probs_b[real_idx], record_idx=rec_idx)
        boot_diffs[i] = np.mean(aurocs_a - aurocs_b)

    boot_mean = float(np.mean(boot_diffs))
    boot_median = float(np.median(boot_diffs))
    pct_lo, pct_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
    # basic (bootstrap-centered / reflection) CI: 2*observed - percentile
    basic_lo = 2 * observed_stat - pct_hi
    basic_hi = 2 * observed_stat - pct_lo
    prop_gt_zero = float(np.mean(boot_diffs > 0))

    return {
        "observed_statistic": observed_stat,
        "observed_detail": observed,
        "bootstrap_mean": boot_mean,
        "bootstrap_median": boot_median,
        "percentile_ci": [pct_lo, pct_hi],
        "basic_ci": [basic_lo, basic_hi],
        "proportion_boot_diffs_gt_zero": prop_gt_zero,
        "n_boot": n_boot,
        "resample_realizations": resample_realizations,
    }


def policy_expected_auroc(y_true: np.ndarray, probs_by_snr: dict, r_map: dict, weights: dict,
                           record_idx: np.ndarray | None = None,
                           real_idx_by_snr: dict | None = None) -> float:
    """Weighted-across-SNR expected macro-AUROC for ONE r-selection policy, on
    one (possibly resampled) set of patients + per-SNR realizations.

    probs_by_snr: {snr: {r: [n_real, N, C] array}}, raw/unresampled.
    r_map: {snr: r} -- which r this policy uses at each SNR (e.g. a fixed-r
    policy uses the same r everywhere; the adaptive policy switches r by SNR).
    weights: {snr: weight}, need not sum to 1 over the full grid if some SNRs
    are excluded (weight 0 or absent) by a sensitivity distribution -- such
    SNRs are skipped entirely (not evaluated), matching the deterministic
    policy-comparison convention used in the validation-stage report.
    record_idx: optional bootstrap-resampled record indices (patient-level).
    real_idx_by_snr: optional {snr: array of resampled realization indices},
    shared across whichever policies are being compared in the same bootstrap
    iteration, to preserve CRN pairing under resampling.
    """
    total = 0.0
    for snr, w in weights.items():
        if w == 0:
            continue
        r = r_map[snr]
        probs = probs_by_snr[snr][r]
        ridx = real_idx_by_snr[snr] if real_idx_by_snr is not None else np.arange(probs.shape[0])
        probs_sel = probs[ridx]
        aurocs = realization_level_aurocs(y_true, probs_sel, record_idx=record_idx)
        total += w * float(np.mean(aurocs))
    return total


def observed_policy_paired_statistic(y_true: np.ndarray, probs_by_snr: dict,
                                      r_map_a: dict, r_map_b: dict, weights: dict) -> dict:
    """PRIMARY estimand for policy-level comparisons: difference of two
    policies' weighted-across-SNR expected macro-AUROC, on the full
    (unresampled) data -- the point estimate the bootstrap CI must match.
    """
    val_a = policy_expected_auroc(y_true, probs_by_snr, r_map_a, weights)
    val_b = policy_expected_auroc(y_true, probs_by_snr, r_map_b, weights)
    return {"expected_auroc_a": val_a, "expected_auroc_b": val_b, "diff": val_a - val_b}


def hierarchical_paired_bootstrap_multi_snr(y_true: np.ndarray, probs_by_snr: dict, patient_ids: np.ndarray,
                                             r_map_a: dict, r_map_b: dict, weights: dict,
                                             n_boot: int = 2000, resample_realizations: bool = True,
                                             seed: int = 0) -> dict:
    """Generalizes `hierarchical_paired_bootstrap` from a single SNR condition
    to a weighted combination across SNRs (an r-selection *policy*, e.g. a
    fixed-r policy or an SNR-adaptive policy), while preserving the same two
    resampling primitives: (1) ONE patient resample per bootstrap iteration,
    shared across every SNR and both policies; (2) an independent realization
    resample per SNR, shared between the two policies being compared at that
    SNR (preserving CRN pairing). `observed_statistic` is computed directly
    by `observed_policy_paired_statistic`, not derived from the bootstrap
    distribution -- same point-estimate/CI consistency guarantee as
    `hierarchical_paired_bootstrap`, verified in tests/test_hierarchical_bootstrap.py.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    observed = observed_policy_paired_statistic(y_true, probs_by_snr, r_map_a, r_map_b, weights)
    observed_stat = observed["diff"]

    active_snrs = [snr for snr, w in weights.items() if w != 0]
    n_real_by_snr = {snr: next(iter(probs_by_snr[snr].values())).shape[0] for snr in active_snrs}

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rec_idx = np.concatenate([patient_to_idx[p] for p in sampled_patients])

        real_idx_by_snr = {}
        for snr in active_snrs:
            n_real = n_real_by_snr[snr]
            real_idx_by_snr[snr] = (rng.choice(n_real, size=n_real, replace=True)
                                     if resample_realizations else np.arange(n_real))

        val_a = policy_expected_auroc(y_true, probs_by_snr, r_map_a, weights,
                                       record_idx=rec_idx, real_idx_by_snr=real_idx_by_snr)
        val_b = policy_expected_auroc(y_true, probs_by_snr, r_map_b, weights,
                                       record_idx=rec_idx, real_idx_by_snr=real_idx_by_snr)
        boot_diffs[i] = val_a - val_b

    boot_mean = float(np.mean(boot_diffs))
    boot_median = float(np.median(boot_diffs))
    pct_lo, pct_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
    basic_lo = 2 * observed_stat - pct_hi
    basic_hi = 2 * observed_stat - pct_lo
    prop_gt_zero = float(np.mean(boot_diffs > 0))

    return {
        "observed_statistic": observed_stat,
        "observed_detail": observed,
        "bootstrap_mean": boot_mean,
        "bootstrap_median": boot_median,
        "percentile_ci": [pct_lo, pct_hi],
        "basic_ci": [basic_lo, basic_hi],
        "proportion_boot_diffs_gt_zero": prop_gt_zero,
        "n_boot": n_boot,
        "resample_realizations": resample_realizations,
    }


def auroc_of_mean_predictions(y_true: np.ndarray, probs_by_realization: np.ndarray) -> float:
    """Sensitivity-analysis estimand B: macro-AUROC of the realization-averaged
    probability, NOT the primary estimand -- report separately, never mixed
    into the primary table (docs/phase2b_mechanism_diagnostic.md instruction 8).
    """
    mean_probs = probs_by_realization.mean(axis=0)
    return compute_metrics(y_true, mean_probs)["macro_auroc"]


def pooled_realization_auroc(y_true: np.ndarray, probs_by_realization: np.ndarray) -> float:
    """Sensitivity-analysis estimand C: macro-AUROC computed by pooling every
    (record, realization) pair as if it were an independent example. Also
    NOT the primary estimand -- reported separately only.
    """
    n_real = probs_by_realization.shape[0]
    y_tiled = np.tile(y_true, (n_real, 1))
    probs_pooled = probs_by_realization.reshape(-1, probs_by_realization.shape[-1])
    return compute_metrics(y_tiled, probs_pooled)["macro_auroc"]


# ---------------------------------------------------------------------------
# AUPRC counterparts (JBHI clinical-metrics completion package). Purely
# additive: every AUROC function above is unchanged; each function below
# mirrors its AUROC namesake exactly, substituting a vectorized macro-AUPRC
# (average precision) batch statistic for the vectorized macro-AUROC one.
# Descriptive/statistical use only -- never used to re-select K, r, a
# threshold, or the SNR-adaptive policy (docs/jbhi_paper_plan.md section 7
# checklist).
# ---------------------------------------------------------------------------

def _vectorized_macro_auprc_batch(y_true: np.ndarray, probs_batch: np.ndarray) -> np.ndarray:
    """y_true: [N, C] fixed multi-hot labels. probs_batch: [B, N, C].

    Returns [B] macro-AUPRC (average precision) values, vectorized over the
    batch dimension B, using the TIE-GROUP-AWARE formula that exactly
    matches sklearn.metrics.average_precision_score even under exact score
    ties -- not just for tie-free continuous scores. This distinction is not
    academic here: patient-level bootstrap resampling (with replacement)
    routinely duplicates a resampled patient's exact row, creating exact
    score ties that a naive per-example rank-sum-style formula (correct for
    AUROC, since Mann-Whitney's rank sum is invariant to how same-label ties
    are ordered) does NOT handle correctly for average precision -- sklearn
    evaluates precision once at the END of each tied score group (weighted
    by that group's recall increment), not once per individual duplicate
    row. An earlier version of this function used the naive per-example
    formula and was caught, by a dedicated tie-stress unit test
    (tests/test_hierarchical_bootstrap.py), to diverge from sklearn by
    several thousandths of macro-AUPRC on realistically-duplicated
    resampled data -- exactly the kind of estimand bug this project's
    existing bootstrap machinery has repeatedly had to catch and fix (see
    docs/phase2a_estimand_audit.md, docs/phase2b_mechanism_diagnostic.md).
    This corrected version is verified byte-for-byte against sklearn on
    both tie-free and deliberately tie-heavy synthetic data.

    Algorithm: sort each row descending by score; identify contiguous
    tied-score groups via `np.maximum.accumulate` (forward-fill of the most
    recent group-start index, fully vectorized along the last axis); for
    the LAST position of each group, add
    (positives in that group) * (precision at the group's last position),
    then divide by n_pos -- this is exactly
    sum_groups (recall_increment_group * precision_at_group_end), the
    standard tie-aware AP definition sklearn implements via
    `precision_recall_curve`'s threshold grouping.
    """
    b, n, c = probs_batch.shape
    per_class_ap = np.zeros((b, c))
    valid = []
    idx_range = np.arange(n)
    for cls in range(c):
        y_c = y_true[:, cls]
        if y_c.min() == y_c.max():
            continue
        valid.append(cls)
        n_pos = int(y_c.sum())
        scores = probs_batch[:, :, cls]  # [B, N]
        order = np.argsort(-scores, axis=-1)  # [B, N], descending
        scores_sorted = np.take_along_axis(scores, order, axis=-1)  # [B, N]
        y_sorted = np.broadcast_to(y_c, (b, n))[np.arange(b)[:, None], order]  # [B, N]

        # group-start boundary: True where this position starts a new tied
        # score group (always True at position 0)
        boundary = np.empty((b, n), dtype=bool)
        boundary[:, 0] = True
        boundary[:, 1:] = scores_sorted[:, 1:] != scores_sorted[:, :-1]

        # forward-fill the most recent group-start index onto every position
        candidate = np.where(boundary, idx_range[None, :], -1)
        group_start_idx = np.maximum.accumulate(candidate, axis=-1)  # [B, N]

        cumsum_pos = np.cumsum(y_sorted, axis=-1)  # [B, N]
        prev_idx = np.clip(group_start_idx - 1, 0, None)
        cumsum_before_group = np.take_along_axis(cumsum_pos, prev_idx, axis=-1)
        cumsum_before_group = np.where(group_start_idx == 0, 0, cumsum_before_group)

        # a position is the LAST member of its group iff the next position
        # starts a new group (or it is the final position in the row)
        last_in_group = np.empty((b, n), dtype=bool)
        last_in_group[:, :-1] = boundary[:, 1:]
        last_in_group[:, -1] = True

        num_pos_in_group = cumsum_pos - cumsum_before_group  # valid only where last_in_group
        precision_at_group_end = cumsum_pos / (idx_range[None, :] + 1)

        contrib = np.where(last_in_group, num_pos_in_group * precision_at_group_end, 0.0)
        per_class_ap[:, cls] = contrib.sum(axis=-1) / n_pos
    if not valid:
        return np.full(b, np.nan)
    return per_class_ap[:, valid].mean(axis=1)


def realization_level_auprcs(y_true: np.ndarray, probs_by_realization: np.ndarray,
                              record_idx: np.ndarray | None = None) -> np.ndarray:
    """AUPRC counterpart of `realization_level_aurocs`."""
    if record_idx is not None:
        y = y_true[record_idx]
        probs = probs_by_realization[:, record_idx, :]
    else:
        y = y_true
        probs = probs_by_realization
    return _vectorized_macro_auprc_batch(y, probs)


def policy_expected_auprc(y_true: np.ndarray, probs_by_snr: dict, r_map: dict, weights: dict,
                           record_idx: np.ndarray | None = None,
                           real_idx_by_snr: dict | None = None) -> float:
    """AUPRC counterpart of `policy_expected_auroc`."""
    total = 0.0
    for snr, w in weights.items():
        if w == 0:
            continue
        r = r_map[snr]
        probs = probs_by_snr[snr][r]
        ridx = real_idx_by_snr[snr] if real_idx_by_snr is not None else np.arange(probs.shape[0])
        probs_sel = probs[ridx]
        auprcs = realization_level_auprcs(y_true, probs_sel, record_idx=record_idx)
        total += w * float(np.mean(auprcs))
    return total


def observed_policy_paired_statistic_auprc(y_true: np.ndarray, probs_by_snr: dict,
                                            r_map_a: dict, r_map_b: dict, weights: dict) -> dict:
    """AUPRC counterpart of `observed_policy_paired_statistic`."""
    val_a = policy_expected_auprc(y_true, probs_by_snr, r_map_a, weights)
    val_b = policy_expected_auprc(y_true, probs_by_snr, r_map_b, weights)
    return {"expected_auprc_a": val_a, "expected_auprc_b": val_b, "diff": val_a - val_b}


def hierarchical_paired_bootstrap_multi_snr_auprc(y_true: np.ndarray, probs_by_snr: dict, patient_ids: np.ndarray,
                                                   r_map_a: dict, r_map_b: dict, weights: dict,
                                                   n_boot: int = 2000, resample_realizations: bool = True,
                                                   seed: int = 0) -> dict:
    """AUPRC counterpart of `hierarchical_paired_bootstrap_multi_snr` -- same
    two-level (patient + Monte-Carlo-realization) resampling discipline,
    same observed-statistic/CI consistency guarantee, substituting
    `policy_expected_auprc` for `policy_expected_auroc` throughout.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    observed = observed_policy_paired_statistic_auprc(y_true, probs_by_snr, r_map_a, r_map_b, weights)
    observed_stat = observed["diff"]

    active_snrs = [snr for snr, w in weights.items() if w != 0]
    n_real_by_snr = {snr: next(iter(probs_by_snr[snr].values())).shape[0] for snr in active_snrs}

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rec_idx = np.concatenate([patient_to_idx[p] for p in sampled_patients])

        real_idx_by_snr = {}
        for snr in active_snrs:
            n_real = n_real_by_snr[snr]
            real_idx_by_snr[snr] = (rng.choice(n_real, size=n_real, replace=True)
                                     if resample_realizations else np.arange(n_real))

        val_a = policy_expected_auprc(y_true, probs_by_snr, r_map_a, weights,
                                       record_idx=rec_idx, real_idx_by_snr=real_idx_by_snr)
        val_b = policy_expected_auprc(y_true, probs_by_snr, r_map_b, weights,
                                       record_idx=rec_idx, real_idx_by_snr=real_idx_by_snr)
        boot_diffs[i] = val_a - val_b

    boot_mean = float(np.mean(boot_diffs))
    boot_median = float(np.median(boot_diffs))
    pct_lo, pct_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
    basic_lo = 2 * observed_stat - pct_hi
    basic_hi = 2 * observed_stat - pct_lo
    prop_gt_zero = float(np.mean(boot_diffs > 0))

    return {
        "observed_statistic": observed_stat,
        "observed_detail": observed,
        "bootstrap_mean": boot_mean,
        "bootstrap_median": boot_median,
        "percentile_ci": [pct_lo, pct_hi],
        "basic_ci": [basic_lo, basic_hi],
        "proportion_boot_diffs_gt_zero": prop_gt_zero,
        "n_boot": n_boot,
        "resample_realizations": resample_realizations,
    }
