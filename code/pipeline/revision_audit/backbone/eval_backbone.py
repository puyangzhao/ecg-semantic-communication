#!/usr/bin/env python
"""Backbone comparison on full 12-lead input (and fixed random lead subsets for the lead-aware model).

Models: submitted encoder+head (outputs/runs/baseline_run1, seed 42, GPU-trained), the same
architecture retrained on CPU with seeds 0-2 (orig_seed*), and the revision backbone (new_seed*).
Reported: macro / per-class AUROC and AUPRC, parameter counts, best epoch, training time; paired
patient-bootstrap of (new - orig retrained) 12-lead macro-AUROC/AUPRC, per seed then mean.
Usage: python revision_audit/backbone/eval_backbone.py --split val|test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "revision_audit/common"))
from stats import auprc, auroc, boot_p, patient_draws  # noqa: E402
from train_backbone import build, predict  # noqa: E402
from src.models.full_model import ECGBaselineModel  # noqa: E402

CACHE = "cache/ptbxl100"


def mv(y, p, idx=None):
    if idx is not None:
        y, p = y[idx], p[idx]
    au = [auroc(y[:, c], p[:, c]) for c in range(5)]
    ap = [auprc(y[:, c], p[:, c]) for c in range(5)]
    return np.array([np.mean(au), np.mean(ap)] + au + ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    if a.split == "test":
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json"))
    torch.set_num_threads(16)
    d = np.load(os.path.join(CACHE, f"{a.split}.npz"))
    x, y, pid = d["x"], d["y"], d["patient_id"]
    ones = np.ones((len(y), 12), np.float32)
    P, info = {}, {}
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/baseline.yaml")))
    sub = ECGBaselineModel(num_leads=12, num_classes=5, latent_dim=32, fusion="mean", base_channels=16, num_blocks=3)
    sub.load_state_dict(torch.load(os.path.join(ROOT, "outputs/runs/baseline_run1/checkpoints/best.pt"), map_location="cpu")["model_state_dict"])
    P["submitted_seed42"] = predict(sub, "orig", x, ones)
    info["submitted_seed42"] = {"params": sum(p.numel() for p in sub.parameters()), "note": "GPU-trained, baseline.yaml", "cfg": cfg["train"]}
    for arch in ("orig", "new"):
        for s in (0, 1, 2):
            m = build(arch)
            m.load_state_dict(torch.load(os.path.join(HERE, f"runs/{arch}_seed{s}/best.pt"))["model_state_dict"])
            P[f"{arch}_seed{s}"] = predict(m, arch, x, ones)
            tr = json.load(open(os.path.join(HERE, f"runs/{arch}_seed{s}/train_result.json")))
            c = json.load(open(os.path.join(HERE, f"runs/{arch}_seed{s}/config.json")))
            info[f"{arch}_seed{s}"] = {"params_total": c["params_total"], "params_encoder": c["params_encoder"],
                                       "best_epoch": tr["best_epoch"], "runtime_min": tr["runtime_min"]}
            if arch == "new":
                rng = np.random.default_rng(4242)
                for k in (1, 3, 6):
                    mk = np.zeros((len(y), 12), np.float32)
                    np.put_along_axis(mk, np.argsort(rng.random((len(y), 12)), 1)[:, :k], 1, 1)
                    P[f"new_seed{s}_randomK{k}"] = predict(m, "new", x, mk)
    names = ["macro_auroc", "macro_auprc"] + [f"auroc_{c}" for c in ("NORM", "MI", "STTC", "CD", "HYP")] + \
            [f"auprc_{c}" for c in ("NORM", "MI", "STTC", "CD", "HYP")]
    obs = {k: mv(y, v) for k, v in P.items()}
    draws = patient_draws(pid, 1000, seed=20261130)
    bd = []
    for idx in draws:
        bd.append([np.mean([mv(y, P[f"new_seed{s}"], idx)[i] - mv(y, P[f"orig_seed{s}"], idx)[i] for s in (0, 1, 2)]) for i in (0, 1)])
    bd = np.array(bd)
    out = {"split": a.split, "absolute": {k: dict(zip(names, map(float, v))) for k, v in obs.items()}, "info": info,
           "new_minus_orig_retrained_12lead": {
               n: {"per_seed": [float(obs[f"new_seed{s}"][i] - obs[f"orig_seed{s}"][i]) for s in (0, 1, 2)],
                   "mean": float(np.mean([obs[f"new_seed{s}"][i] - obs[f"orig_seed{s}"][i] for s in (0, 1, 2)])),
                   "ci95": list(map(float, np.percentile(bd[:, i], [2.5, 97.5]))), "p_boot": boot_p(bd[:, i])}
               for i, n in enumerate(("macro_auroc", "macro_auprc"))}}
    json.dump(out, open(os.path.join(HERE, "results", f"backbone_{a.split}.json"), "w"), indent=2)
    for k, v in obs.items():
        print(k, round(v[0], 4), round(v[1], 4))
    print(json.dumps(out["new_minus_orig_retrained_12lead"]))


if __name__ == "__main__":
    main()
