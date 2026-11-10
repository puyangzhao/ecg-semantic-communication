#!/usr/bin/env python
"""Six-seed training-stability analysis of the primary K=1 comparison (post hoc; see PLAN.md).

orig_A (selector) - orig_C (matched fixed V5), original frozen backbone, test fold 10.
Seeds 0-2: logits reused from revision_audit/formal_eval/results/k1_logits_test.npz.
Seeds 3-5: computed once from the checkpoints listed in PLAN.md, same inference path as k1_eval.py.

    python revision_audit/stability6/stability6_eval.py      # -> results/stability6_test.json, .npz
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revision_audit/common"))
from scripts.train_phase2a import build_selector  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from stats import auprc, auroc, patient_draws  # noqa: E402

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = (0, 1, 2, 3, 4, 5)
V5 = LEAD_NAMES.index("V5")
SEL = {s: f"outputs/runs/phase2a_proposed_seed{s}/checkpoints/best.pt" for s in (0, 1, 2)}
SEL.update({s: f"revision_audit/k1_gate_pilot/phase2a_extra_seeds/runs/phase2a_proposed_seed{s}/checkpoints/best.pt"
            for s in (3, 4, 5)})
FIX = {s: f"revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed{s}/checkpoints/best.pt" for s in SEEDS}


def sig(x):
    return 1 / (1 + np.exp(-x.astype(np.float64)))


def ld(path, method):
    m = Phase2ASelectionModel(build_selector(method, 32), latent_dim=32, num_classes=5)
    m.load_state_dict(torch.load(os.path.join(ROOT, path), map_location="cpu")["model_state_dict"])
    return m.eval()


def metrics(y, p):
    out = {}
    for j, c in enumerate(CLASSES):
        out[f"{c}_auroc"] = auroc(y[:, j], p[:, j])
        out[f"{c}_auprc"] = auprc(y[:, j], p[:, j])
    out["macro_auroc"] = float(np.mean([out[f"{c}_auroc"] for c in CLASSES]))
    out["macro_auprc"] = float(np.mean([out[f"{c}_auprc"] for c in CLASSES]))
    return out


KEYS = ["macro_auroc", "macro_auprc", "CD_auroc", "CD_auprc", "STTC_auroc", "STTC_auprc"]


def main():
    saved = np.load(os.path.join(ROOT, "revision_audit/formal_eval/results/k1_logits_test.npz"))
    d = torch.load(os.path.join(ROOT, "outputs/phase2a/embeddings/test.pt"))
    Z, N = d["z"], d["z"].shape[0]
    y = d["y"].numpy().astype(int)
    pid = d["patient_id"].numpy()
    assert np.array_equal(saved["ecg_id"], d["ecg_id"].numpy()) and np.array_equal(saved["y"], y)
    v5 = torch.zeros(N, 12)
    v5[:, V5] = 1

    P, check = {}, {}
    with torch.no_grad():
        for s in SEEDS:
            pa, pc = ld(SEL[s], "proposed"), ld(FIX[s], "fixed_clinical_k")
            la, lc = pa(Z, 1)[0].numpy(), pc.classifier(pc.fusion(Z, v5)).numpy()
            if s in (0, 1, 2):  # reuse locked predictions; recomputation is only a consistency check
                check[f"seed{s}_A_max_abs_logit_diff"] = float(np.abs(la - saved[f"orig_A_seed{s}"]).max())
                check[f"seed{s}_C_max_abs_logit_diff"] = float(np.abs(lc - saved[f"orig_C_seed{s}"]).max())
                la, lc = saved[f"orig_A_seed{s}"], saved[f"orig_C_seed{s}"]
            P[("A", s)], P[("C", s)] = sig(la), sig(lc)

    per_seed = {}
    for s in SEEDS:
        ma, mc = metrics(y, P[("A", s)]), metrics(y, P[("C", s)])
        per_seed[s] = {"selector": {k: ma[k] for k in KEYS}, "fixed_V5": {k: mc[k] for k in KEYS},
                       "diff": {k: ma[k] - mc[k] for k in KEYS}}

    locked = json.load(open(os.path.join(ROOT, "revision_audit/formal_eval/results/k1_eval_test.json")))
    check["locked_primary_per_seed"] = locked["primary"]["per_seed"]
    check["recomputed_primary_seed0_2"] = [per_seed[s]["diff"]["macro_auroc"] for s in (0, 1, 2)]
    check["max_abs_vs_locked"] = float(np.max(np.abs(np.array(check["locked_primary_per_seed"])
                                                     - np.array(check["recomputed_primary_seed0_2"]))))

    across = {}
    for k in KEYS:
        v = np.array([per_seed[s]["diff"][k] for s in SEEDS])
        across[k] = {"mean": float(v.mean()), "sd_between_seeds": float(v.std(ddof=1)),
                     "min": float(v.min()), "max": float(v.max()), "n_positive": int((v > 0).sum())}

    draws = patient_draws(pid, 2000)
    boot = {k: [] for k in KEYS}
    for idx in draws:
        yb = y[idx]
        ds = {k: [] for k in KEYS}
        for s in SEEDS:
            ma, mc = metrics(yb, P[("A", s)][idx]), metrics(yb, P[("C", s)][idx])
            for k in KEYS:
                ds[k].append(ma[k] - mc[k])
        for k in KEYS:
            boot[k].append(np.mean(ds[k]))
    for k in KEYS:
        b = np.array(boot[k])
        across[k]["patient_bootstrap_ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    out = {"status": "post hoc six-seed training-stability analysis, added after the locked fold-10 results; see PLAN.md",
           "comparison": "orig_A (selector) - orig_C (matched fixed V5), K=1, original backbone, PTB-XL fold 10",
           "seeds": list(SEEDS), "n_records": int(N), "n_patients": int(len(np.unique(pid))),
           "sd_meaning": "between-training-run SD of the per-seed difference (ddof=1)",
           "ci_meaning": "2000 patient-level paired resamples (rng 20261110); per resample per-seed difference then mean "
                         "over the six seeds; patient-sampling uncertainty given these six trained models",
           "checkpoints": {"selector": SEL, "fixed_V5": FIX},
           "per_seed": {str(s): v for s, v in per_seed.items()}, "across_seeds": across, "checks": check}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results/stability6_test.json"), "w"), indent=1)
    np.savez(os.path.join(HERE, "results/stability6_probs_test.npz"), ecg_id=d["ecg_id"].numpy(), y=y,
             patient_id=pid, **{f"{a}_seed{s}": P[(a, s)] for a in "AC" for s in SEEDS})
    print(json.dumps({"across": across, "checks": check}, indent=1))


if __name__ == "__main__":
    main()
