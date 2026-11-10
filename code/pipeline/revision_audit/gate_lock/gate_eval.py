#!/usr/bin/env python
"""Apply the locked K=1 gate (and the margin / random comparators) to a cohort, unchanged.

For each requested seed: original proposed selector+head checkpoint of that seed -> A (selector lead)
and B (V5 forced) logits, selector candidate lead and top1-top2 margin; gate = ridge refitted exactly
as in gate_pilot.py / lock_gate_seeds.py (deterministic; asserted identical to the locked parameters
for seeds 0, 3, 5); scores, rank-matched budgets (25/50/75%), frozen TRAINING-quantile thresholds and
200 random routings per budget are produced in exactly gate_pilot.py's order and RNG, so that
--split val reproduces every saved fold-9 row (implementation check).

Output per seed: <out>/seed{s}/per_record.npz + gate_eval_results.json, the format read by
revision_audit/k1_gate_pilot/multiseed_bootstrap.py --root <out>.
The 50% rank-matched policy is a QUOTA comparison (it ranks the whole evaluated batch); the
threshold policies apply the frozen training thresholds record by record.

Usage: python revision_audit/gate_lock/gate_eval.py --split test --seeds 0,1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GP = os.path.join(ROOT, "revision_audit/k1_gate_pilot")
sys.path.insert(0, ROOT)
sys.path.insert(0, GP)
import gate_pilot as gp  # noqa: E402
from oof_inner_fold import k1_outputs  # noqa: E402
from scripts.train_phase2a import build_selector  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402

V5 = gp.V5


def refit_gate(s):
    oof_name = (lambda k: f"oof_heldout_fold{k}.npz") if s == 0 else (lambda k: f"oof_seed{s}_heldout_fold{k}.npz")
    oof = [dict(np.load(os.path.join(GP, "results/oof", oof_name(k)))) for k in range(1, 9)]
    tr = {k: np.concatenate([o[k] for o in oof]) for k in ("z", "y_true", "A_logits", "B_logits", "cand_lead",
                                                          "margin_top1_top2")}
    sw = tr["cand_lead"] != V5
    X = gp.features(tr["z"], tr["cand_lead"], tr["margin_top1_top2"])[sw]
    T = (gp.bce(tr["B_logits"], tr["y_true"]) - gp.bce(tr["A_logits"], tr["y_true"]))[sw]
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    g = Ridge(alpha=gp.CFG["gate_model"]["alpha"], fit_intercept=True).fit((X - mu) / sd, T)
    lock = os.path.join(HERE, f"gate_seed{s}.npz")
    if os.path.exists(lock):
        L = np.load(lock)
        assert np.allclose(L["coef"], g.coef_, rtol=0, atol=1e-12) and np.allclose(L["mu"], mu) and np.allclose(L["sd"], sd)
    return g, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--seeds", default="0,3,5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.split == "test":
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json"))
    out_root = a.out or os.path.join(HERE, "results", a.split)
    torch.set_num_threads(8)
    d = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{a.split}.pt"))
    y = d["y"].numpy()
    checks_all = {}
    for s in map(int, a.seeds.split(",")):
        g, mu, sd = refit_gate(s)
        ck = (f"outputs/runs/phase2a_proposed_seed{s}/checkpoints/best.pt" if s <= 2 else
              f"revision_audit/k1_gate_pilot/phase2a_extra_seeds/runs/phase2a_proposed_seed{s}/checkpoints/best.pt")
        model = Phase2ASelectionModel(build_selector("proposed", 32), latent_dim=32, num_classes=5)
        model.load_state_dict(torch.load(os.path.join(ROOT, ck), map_location="cpu")["model_state_dict"])
        with torch.no_grad():
            o = k1_outputs(model, d["z"])
        A, B, cand, margin = o["A_logits"], o["B_logits"], o["cand_lead"], o["margin_top1_top2"]
        sw = cand != V5
        n_sw = int(sw.sum())
        X = gp.features(d["z"].numpy(), cand, margin)
        pred = np.zeros((len(y), 5))
        pred[sw] = g.predict((X[sw] - mu) / sd)
        scores = {name: pred @ w for name, w in gp.SCORE_W.items()}
        scores["margin"] = margin
        saved_res = json.load(open(os.path.join(GP, "results" if s == 0 else f"results/seed{s}", "gate_pilot_results.json")))
        thresholds = saved_res["thresholds_from_training"]

        def policy_logits(switch):
            assert not (switch & ~sw).any()
            return np.where(switch[:, None], A, B)

        def rank_switch(score, q):
            n = int(round(q * n_sw))
            idx = np.where(sw)[0]
            s_ = np.zeros(len(y), bool)
            s_[idx[np.argsort(-score[idx], kind="stable")[:n]]] = True
            return s_

        rows, masks = [], {}

        def add(name, budget, switch, extra=None):
            rows.append({"policy": name, "budget": budget, "actual_switch_rate": float(switch[sw].mean()) if n_sw else float("nan"),
                         "n_switched": int(switch.sum()), **gp.metrics(y, policy_logits(switch)), **(extra or {})})
            masks[f"{name}@{budget}"] = switch

        add("B (0%, all V5)", "0%", np.zeros(len(y), bool))
        add("A (100%, selector lead)", "100%", sw.copy())
        rng = np.random.default_rng(gp.RANDOM_SEED)
        for q in gp.CFG["budgets_fraction_of_switchable"]:
            tag = f"{int(q * 100)}%"
            for s_name in ("equal", "sttc_weighted", "margin"):
                label = {"equal": "gate equal-weight", "sttc_weighted": "gate STTC x2", "margin": "selector margin"}[s_name]
                add(f"{label} (rank-matched)", tag, rank_switch(scores[s_name], q))
                thr = thresholds[f"{s_name}@{tag}"]
                add(f"{label} (train threshold)", tag, sw & (scores[s_name] > thr), {"threshold": thr})
            n = int(round(q * n_sw))
            idx = np.where(sw)[0]
            reps = []
            for _ in range(gp.RANDOM_REPS):
                m = np.zeros(len(y), bool)
                m[rng.choice(idx, size=n, replace=False)] = True
                reps.append(gp.metrics(y, policy_logits(m)))
            rows.append({"policy": f"random (mean of {gp.RANDOM_REPS})", "budget": tag, "actual_switch_rate": n / n_sw,
                         "n_switched": n, **{k: float(np.mean([r[k] for r in reps])) for k in reps[0]}})
        checks = {"n_records": int(len(y)), "n_switchable": n_sw, "split": a.split, "checkpoint": ck}
        if a.split == "val":  # implementation check against the saved fold-9 rows
            saved = {(r["policy"], r["budget"]): r for r in saved_res["rows"]}
            dev = max(abs(r[k] - saved[(r["policy"], r["budget"])][k]) for r in rows for k in ("macro_auroc", "macro_auprc",
                                                                                                "CD_auprc", "STTC_auprc")
                      if (r["policy"], r["budget"]) in saved)
            checks["val_rows_vs_saved_fold9_max_abs"] = float(dev)
        od = os.path.join(out_root, f"seed{s}")
        os.makedirs(od, exist_ok=True)
        json.dump({"checks": checks, "thresholds_from_training": thresholds, "rows": rows}, open(os.path.join(od, "gate_eval_results.json"), "w"), indent=2)
        np.savez(os.path.join(od, "per_record.npz"), ecg_id=d["ecg_id"].numpy(), patient_id=d["patient_id"].numpy(), y_true=y,
                 A_logits=A, B_logits=B, cand_lead=cand, margin=margin, switchable=sw, pred_gain=pred,
                 **{f"score_{k}": v for k, v in scores.items()},
                 **{"switch__" + k.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct"): v for k, v in masks.items()})
        checks_all[s] = checks
        print(s, checks, flush=True)
    json.dump(checks_all, open(os.path.join(out_root, "checks.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
