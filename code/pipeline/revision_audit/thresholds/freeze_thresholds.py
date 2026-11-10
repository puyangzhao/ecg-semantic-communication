#!/usr/bin/env python
"""Development-set operating points, one rule for every model and diagnosis (frozen before fold 10).

Rule (pre-specified): per (model, seed, class), on fold 9, candidate thresholds are the
midpoints between consecutive distinct predicted probabilities (plus 0 and 1); choose the one
maximizing Youden's J = sensitivity + specificity - 1; ties -> the LOWEST threshold among the
maximizers (higher sensitivity); the numeric probability threshold is frozen. These are
"development-set operating points", not clinically validated cut-offs.

Models (K=1): orig_A (proposed selector + its head), orig_B (proposed head, V5 forced),
orig_C (matched fixed-V5 head), and, when available, new_A / new_B / new_C on the revision backbone.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def youden_threshold(y, p):
    u = np.unique(p)
    cand = np.concatenate([[0.0], (u[1:] + u[:-1]) / 2, [1.0]])
    pos, neg = y == 1, y == 0
    # vectorized over candidates via sorting
    ps = np.sort(p[pos]), np.sort(p[neg])
    sens = 1 - np.searchsorted(ps[0], cand, side="left") / pos.sum()   # P(p >= t | pos)
    spec = np.searchsorted(ps[1], cand, side="left") / neg.sum()       # P(p < t | neg)
    J = sens + spec - 1
    best = np.flatnonzero(np.isclose(J, J.max(), rtol=0, atol=1e-12))
    i = best[0]  # lowest threshold among maximizers
    return float(cand[i]), float(sens[i]), float(spec[i]), float(J[i])


def val_logits():
    out = {}
    for s in (0, 1, 2):
        d = np.load(os.path.join(ROOT, f"revision_audit/k1_cd_sttc_path/results/val_logits_ABC_seed{s}.npz"))
        for k in "ABC":
            out[(f"orig_{k}", s)] = (d["y_true"], d[f"{k}_logits"])
    new_path = os.path.join(HERE, "new_backbone_val_logits.npz")
    if os.path.exists(new_path):
        d = np.load(new_path)
        for s in (0, 1, 2):
            for k in "ABC":
                if f"new_{k}_seed{s}" in d:
                    out[(f"new_{k}", s)] = (d["y_true"], d[f"new_{k}_seed{s}"])
    return out


def main():
    res = {"rule": __doc__.split("Rule (pre-specified): ")[1].split("Models")[0].strip(), "thresholds": {}}
    for (name, s), (y, lg) in sorted(val_logits().items()):
        p = 1 / (1 + np.exp(-lg.astype(np.float64)))
        res["thresholds"].setdefault(name, {})[str(s)] = {
            c: dict(zip(("threshold", "val_sens", "val_spec", "val_J"), youden_threshold(y[:, i], p[:, i])))
            for i, c in enumerate(CLASSES)}
    json.dump(res, open(os.path.join(HERE, "frozen_thresholds.json"), "w"), indent=2)
    for name, v in res["thresholds"].items():
        print(name, {c: round(np.mean([v[s][c]["threshold"] for s in v]), 3) for c in CLASSES})


if __name__ == "__main__":
    main()
