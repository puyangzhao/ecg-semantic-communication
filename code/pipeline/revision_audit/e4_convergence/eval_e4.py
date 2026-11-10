#!/usr/bin/env python
"""E4 evaluation: extended-training (E4, max 120 epochs) vs original 40-epoch lead-aware backbone.

Models per seed s in {0,1,2} (paired by seed; nothing re-selected here):
  e4_12 / o40_12  backbone + its own receiver, full 12-lead input
  e4_A   proposed K=1 selector + head on the E4 encoder        e4_B  same head, V5 forced
  e4_C   matched fixed-V5 head on the E4 encoder               e4_BEST fixed head, E4 fold-9 best lead
  e4_GL  population-level learned ranking (K=1)
  o40_A / o40_C / o40_BEST  the same heads on the 40-epoch encoder (k1_fair runs; best lead V6)
Statistics reuse formal_eval/k1_eval.py (metric_vec, bootstrap_all) and common/stats.py: one shared set of
2000 patient-level resamples (rng 20261110); per resample, per-seed difference then mean over seeds;
percentile 95% CI; bootstrap p. Exploratory, post hoc, unadjusted. Between-seed SD reported separately.
--split test requires plan_hashes/EVAL_FROZEN.sha256 and refuses to overwrite an existing test result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("E4_REPO_ROOT", os.path.dirname(os.path.dirname(HERE)))  # override only for the scratch dry run
for p in ("", "revision_audit/common", "revision_audit/backbone", "revision_audit/k1_fair", "revision_audit/formal_eval"):
    sys.path.insert(0, os.path.join(ROOT, p))
from k1_eval import MNAMES, bootstrap_all, metric_vec, sig  # noqa: E402
from lead_resnet import LeadAwareModel  # noqa: E402
from stats import boot_p, patient_draws  # noqa: E402
from train_backbone import CACHE, NEW_KW  # noqa: E402
from train_heads import build as build_head  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402

SEEDS = (0, 1, 2)
V5 = LEAD_NAMES.index("V5")


@torch.no_grad()
def logits_12(ckpt, x):
    m = LeadAwareModel(**NEW_KW)
    m.load_state_dict(torch.load(ckpt)["model_state_dict"])
    m.eval()
    out = []
    for i in range(0, len(x), 256):
        xb = torch.from_numpy(x[i:i + 256])
        out.append(m(xb, torch.ones(len(xb), 12)))
    return torch.cat(out).numpy()


@torch.no_grad()
def heads(emb_path, run_dir, prefix, s, best_lead, tag):
    Z = torch.load(emb_path)["z"]
    v5 = torch.zeros(len(Z), 12)
    v5[:, V5] = 1

    def ld(method, lead=None):
        m = build_head(method, prefix, lead)
        name = f"{prefix}_{method}{'_' + lead if lead else ''}_seed{s}"
        m.load_state_dict(torch.load(os.path.join(run_dir, name, "best.pt"))["model_state_dict"])
        return m.eval()
    pa = ld("proposed")
    out = {(f"{tag}_A", s): pa(Z, 1)[0].numpy(), (f"{tag}_B", s): pa.logits_with_mask(Z, v5).numpy(),
           (f"{tag}_C", s): ld("fixed_lead", "V5")(Z, 1)[0].numpy(),
           (f"{tag}_BEST", s): ld("fixed_lead", best_lead)(Z, 1)[0].numpy()}
    if tag == "e4":
        out[(f"{tag}_GL", s)] = ld("global_learned_k")(Z, 1)[0].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    res_path = os.path.join(HERE, "results", f"e4_eval_{a.split}.json")
    if a.split == "test":
        assert os.path.exists(os.path.join(HERE, "plan_hashes/EVAL_FROZEN.sha256")), "freeze before fold 10"
        assert not os.path.exists(res_path), "E4 fold-10 evaluation already run once"
    torch.set_num_threads(16)
    d = np.load(os.path.join(CACHE, f"{a.split}.npz"))
    x, y = d["x"], d["y"]
    e4_best = json.load(open(os.path.join(HERE, "results/e4_best_fixed_lead.json")))["locked_best_lead"]
    o40_best = json.load(open(os.path.join(ROOT, "revision_audit/k1_fair/results/new_backbone_best_fixed_lead.json")))["locked_best_lead"]
    P = {}
    for s in SEEDS:
        P[("e4_12", s)] = logits_12(os.path.join(HERE, f"runs/new_seed{s}_e120/best.pt"), x)
        P[("o40_12", s)] = logits_12(os.path.join(ROOT, f"revision_audit/backbone/runs/new_seed{s}/best.pt"), x)
        P.update(heads(os.path.join(HERE, f"emb/seed{s}/{a.split}.pt"), os.path.join(HERE, "runs"), f"e4_s{s}", s, e4_best, "e4"))
        P.update(heads(os.path.join(ROOT, f"revision_audit/backbone/emb/new_seed{s}/{a.split}.pt"),
                       os.path.join(ROOT, "revision_audit/k1_fair/runs"), f"new_s{s}", s, o40_best, "o40"))
    emb = torch.load(os.path.join(HERE, f"emb/seed0/{a.split}.pt"))
    assert np.array_equal(emb["ecg_id"].numpy(), d["ecg_id"]) and np.array_equal(emb["y"].numpy(), y)
    pid = emb["patient_id"].numpy()
    probs = {k: sig(v) for k, v in P.items()}
    obs = {k: metric_vec(y, v) for k, v in probs.items()}
    boot = bootstrap_all(y, probs, patient_draws(pid, a.B))
    models = sorted({k[0] for k in probs})
    out = {"split": a.split, "n_records": int(len(y)), "n_patients": int(len(np.unique(pid))), "B": a.B,
           "bootstrap": "patient-level, rng 20261110, shared by all models/comparisons; conditional on trained models",
           "e4_best_fixed_lead": e4_best, "o40_best_fixed_lead": o40_best, "absolute": {}, "paired": {}}
    for m in models:
        per = np.array([obs[(m, s)] for s in SEEDS])
        bm = np.mean([boot[(m, s)] for s in SEEDS], 0)
        out["absolute"][m] = {n: {"per_seed": per[:, i].tolist(), "mean": float(per[:, i].mean()),
                                  "sd_seeds": float(per[:, i].std(ddof=1)),
                                  "ci95_of_seed_mean": [float(np.nanpercentile(bm[:, i], q)) for q in (2.5, 97.5)]}
                              for i, n in enumerate(MNAMES)}
    comps = [("e4_12", "o40_12"), ("e4_A", "e4_C"), ("e4_A", "e4_BEST"), ("e4_A", "e4_B"), ("e4_B", "e4_C"),
             ("e4_A", "e4_GL"), ("e4_A", "o40_A"), ("e4_C", "o40_C"), ("o40_A", "o40_C"), ("o40_A", "o40_BEST")]
    for u, v in comps:
        res = {}
        for i, n in enumerate(MNAMES):
            ps = [float(obs[(u, s)][i] - obs[(v, s)][i]) for s in SEEDS]
            bd = np.mean([boot[(u, s)][:, i] - boot[(v, s)][:, i] for s in SEEDS], 0)
            res[n] = {"per_seed": ps, "mean": float(np.mean(ps)), "sd_seeds": float(np.std(ps, ddof=1)),
                      "ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))], "p_boot": boot_p(bd)}
        out["paired"][f"{u}-{v}"] = res
    # addendum 1: descriptive protocol-alignment sensitivity (records with >=1 superclass label only)
    lab = y.sum(1) > 0
    sub_draws = [idx[lab[idx]] for idx in patient_draws(pid, 500)]
    out["labelled_only_12lead"] = {"n_records": int(lab.sum()), "n_excluded_all_negative": int((~lab).sum())}
    for m in ("e4_12", "o40_12"):
        per = np.array([metric_vec(y[lab], probs[(m, s)][lab]) for s in SEEDS])
        bm = np.array([np.mean([metric_vec(y[i], probs[(m, s)][i]) for s in SEEDS], 0) for i in sub_draws])
        out["labelled_only_12lead"][m] = {n: {"per_seed": per[:, j].tolist(), "mean": float(per[:, j].mean()),
                                              "sd_seeds": float(per[:, j].std(ddof=1)),
                                              "ci95_of_seed_mean_B500": [float(np.nanpercentile(bm[:, j], q)) for q in (2.5, 97.5)]}
                                          for j, n in enumerate(MNAMES) if n in ("macro_auroc", "macro_auprc")}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(res_path, "w"), indent=2)
    np.savez(os.path.join(HERE, "results", f"e4_logits_{a.split}.npz"), y=y, patient_id=pid, ecg_id=d["ecg_id"],
             **{f"{m}_seed{s}": v for (m, s), v in P.items()})
    for m in ("e4_12", "o40_12", "e4_A", "e4_C", "e4_BEST"):
        r = out["absolute"][m]
        print(m, {n: round(r[n]["mean"], 4) for n in ("macro_auroc", "macro_auprc")}, [round(q, 4) for q in r["macro_auroc"]["per_seed"]])
    for c in ("e4_12-o40_12", "e4_A-e4_C", "e4_A-e4_BEST", "o40_A-o40_C"):
        r = out["paired"][c]["macro_auroc"]
        print(c, round(r["mean"], 4), [round(q, 4) for q in r["ci95"]])


if __name__ == "__main__":
    main()
