"""Demographic subgroup evaluation for Phase 2A (secondary analysis).

See docs/phase2a_design.md section 11. Age recoding, insufficient-data
guards, worst-group AUROC, and max-min subgroup gap are all predefined
here, before any Phase 2A result exists.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.labels import SUPERCLASSES

AGE_BIN_LABELS = ["<40", "40-59", "60-74", "75-89", "90+"]
MIN_POS = 10
MIN_NEG = 10


def recode_age_group(age: np.ndarray) -> np.ndarray:
    """PTB-XL encodes ages over 89 as the literal value 300 (de-identification
    convention, confirmed: 293/21799 records, none between 90 and 299). This
    is recoded to a "90+" bucket -- 300 is never used as a literal age in any
    downstream computation.
    """
    age = np.asarray(age)
    out = np.empty(age.shape, dtype=object)
    out[age < 40] = AGE_BIN_LABELS[0]
    out[(age >= 40) & (age < 60)] = AGE_BIN_LABELS[1]
    out[(age >= 60) & (age < 75)] = AGE_BIN_LABELS[2]
    out[(age >= 75) & (age < 90)] = AGE_BIN_LABELS[3]
    out[age >= 90] = AGE_BIN_LABELS[4]  # includes the age==300 sentinel
    return out


def _class_auroc_with_guard(y_t: np.ndarray, y_p: np.ndarray) -> dict:
    n_pos = int(y_t.sum())
    n_neg = int(len(y_t) - n_pos)
    if n_pos < MIN_POS:
        return {"auroc": float("nan"), "n_pos": n_pos, "n_neg": n_neg, "reason": "insufficient_positive"}
    if n_neg < MIN_NEG:
        return {"auroc": float("nan"), "n_pos": n_pos, "n_neg": n_neg, "reason": "insufficient_negative"}
    return {"auroc": float(roc_auc_score(y_t, y_p)), "n_pos": n_pos, "n_neg": n_neg, "reason": None}


def subgroup_auroc_table(y_true: np.ndarray, y_prob: np.ndarray, subgroup_labels: np.ndarray) -> dict:
    """y_true, y_prob: [N, C]. subgroup_labels: [N] categorical.

    Returns {subgroup_name: {"n": int, "per_class": {cls: {...}}, "macro_auroc": float|nan}}.
    """
    result = {}
    for g in sorted(set(subgroup_labels)):
        idx = subgroup_labels == g
        n = int(idx.sum())
        per_class = {}
        valid = []
        for i, cls in enumerate(SUPERCLASSES):
            cell = _class_auroc_with_guard(y_true[idx, i], y_prob[idx, i])
            per_class[cls] = cell
            if not np.isnan(cell["auroc"]):
                valid.append(cell["auroc"])
        macro = float(np.mean(valid)) if valid else float("nan")
        result[str(g)] = {"n": n, "per_class": per_class, "macro_auroc": macro,
                           "n_valid_classes": len(valid)}
    return result


def worst_group_and_gap(subgroup_table: dict) -> dict:
    """subgroup_table: output of subgroup_auroc_table. Computes worst-group
    AUROC and max-min gap only over subgroups with a defined macro_auroc; if
    fewer than 2 are defined, reports NaN with an explicit reason rather than
    a spurious number from an underpowered comparison.
    """
    defined = {g: v["macro_auroc"] for g, v in subgroup_table.items() if not np.isnan(v["macro_auroc"])}
    if len(defined) < 2:
        return {
            "worst_group": None, "worst_group_auroc": float("nan"),
            "best_group": None, "best_group_auroc": float("nan"),
            "max_min_gap": float("nan"),
            "reason": "fewer_than_2_defined_subgroups",
        }
    worst_g = min(defined, key=defined.get)
    best_g = max(defined, key=defined.get)
    return {
        "worst_group": worst_g, "worst_group_auroc": defined[worst_g],
        "best_group": best_g, "best_group_auroc": defined[best_g],
        "max_min_gap": defined[best_g] - defined[worst_g],
        "reason": None,
    }
