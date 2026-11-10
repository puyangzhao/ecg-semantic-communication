#!/usr/bin/env python
"""External evaluation on PhysioNet ecg-arrhythmia 1.0.0 (post-lock only; nothing fitted externally).

Inputs: adapter output cache/ecgarr/x_mv.npy (frozen to_training_format), frozen label
file results/external_labels.csv built from label_map_frozen.json (hash in the lock manifest).
Primary   : per seed 0-2, orig_A (proposed selector + head) vs orig_C (matched fixed-V5 head), K=1,
            for CD and STTC-partial, AUROC and AUPRC; record-level bootstrap (no patient identifier),
            B=2000 shared draws, per-seed difference then mean.
Secondary : orig_A - orig_B (same-head ablation); orig_B - orig_C.
Gate      : not evaluated externally (decided before any external model output): its training target
            is the mean BCE gain over all five PTB-XL classes, not interpretable with only CD and
            STTC-partial mapped.
Reported  : prevalence, absolute metrics, selection frequency. Raw AUPRC is not compared with PTB-XL.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
for p in ("revision_audit/common", "revision_audit/external_validation_plan", "revision_audit/gate_lock",
          "revision_audit/k1_gate_pilot"):
    sys.path.insert(0, os.path.join(ROOT, p))
from external_pipeline import FrozenK1Pipeline  # noqa: E402
from stats import auprc, auroc, boot_p, record_draws  # noqa: E402

OUTC = {"CD": 3, "STTC_partial": 2}  # PTB-XL head output index used for each external outcome
CACHE = "cache/ecgarr"


def main():
    assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json")), "post-lock only"
    torch.set_num_threads(16)
    lab = pd.read_csv(os.path.join(HERE, "results/external_labels.csv"))
    idx = pd.read_csv(os.path.join(CACHE, "x_index.csv"))
    pos = pd.Series(np.arange(len(idx)), index=idx["record"])
    sel = pos.loc[lab["record"]].to_numpy()
    X = np.load(os.path.join(CACHE, "x_mv.npy"), mmap_mode="r")
    pipe = FrozenK1Pipeline("cpu", seeds=(0, 1, 2))
    Z = torch.cat([pipe.embed(np.ascontiguousarray(X[sel[i:i + 2048]])) for i in range(0, len(sel), 2048)])
    torch.save({"z": Z, "record": list(lab["record"])}, os.path.join(CACHE, "z_orig_encoder.pt"))
    pr = pipe.predict(Z)
    sig = lambda v: 1 / (1 + np.exp(-v.astype(np.float64)))
    draws = record_draws(len(lab), 2000)
    out = {"n_eligible": int(len(lab)), "outcomes": {}, "selection_frequency_A": {}}
    for s in (0, 1, 2):
        out["selection_frequency_A"][s] = np.bincount(pr[s]["A_lead_idx"], minlength=12).tolist()
    for oc, ci in OUTC.items():
        yv = lab[oc].to_numpy()
        m = yv >= 0
        y = yv[m]
        res = {"n": int(m.sum()), "positives": int(y.sum()), "prevalence": float(y.mean()), "absolute": {}, "paired": {}}
        P = {(k, s): sig(pr[s][f"{k}_logits"][m, ci]) for k in "ABC" for s in (0, 1, 2)}
        for k in "ABC":
            res["absolute"][k] = {met: {"per_seed": [f(y, P[(k, s)]) for s in (0, 1, 2)]} for met, f in (("auroc", auroc), ("auprc", auprc))}
            for met in ("auroc", "auprc"):
                res["absolute"][k][met]["mean"] = float(np.mean(res["absolute"][k][met]["per_seed"]))
        mi = np.flatnonzero(m)
        pos_in_m = -np.ones(len(lab), int)
        pos_in_m[mi] = np.arange(len(mi))
        for a_, b_ in (("A", "C"), ("A", "B"), ("B", "C")):
            for met, f in (("auroc", auroc), ("auprc", auprc)):
                obs = [f(y, P[(a_, s)]) - f(y, P[(b_, s)]) for s in (0, 1, 2)]
                bd = []
                for d in draws:
                    ii = pos_in_m[d]
                    ii = ii[ii >= 0]
                    bd.append(np.mean([f(y[ii], P[(a_, s)][ii]) - f(y[ii], P[(b_, s)][ii]) for s in (0, 1, 2)]))
                res["paired"][f"{a_}-{b_}_{met}"] = {"per_seed": list(map(float, obs)), "mean": float(np.mean(obs)),
                                                    "ci95": list(map(float, np.nanpercentile(bd, [2.5, 97.5]))), "p_boot": boot_p(bd)}
        out["outcomes"][oc] = res
        print(oc, res["n"], res["positives"], {k: (round(v["mean"], 4), [round(q, 4) for q in v["ci95"]]) for k, v in res["paired"].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results/external_eval.json"), "w"), indent=2)
    np.savez(os.path.join(HERE, "results/external_logits.npz"), record=lab["record"].to_numpy(),
             **{f"{k}_seed{s}": pr[s][f"{k}_logits"] for k in "ABC" for s in (0, 1, 2)},
             **{f"A_lead_seed{s}": pr[s]["A_lead_idx"] for s in (0, 1, 2)})


if __name__ == "__main__":
    main()
