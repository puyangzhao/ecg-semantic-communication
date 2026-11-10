#!/usr/bin/env python
"""Stage 2: K=1 two-way gate (A vs B) pilot, seed 0, CPU, clean Phase 2A.

Implements configs/gate_prespec.yaml exactly (settings frozen before any
gate result was computed). Training data = out-of-fold A/B predictions on
PTB-XL folds 1-8 from stage 1 (selector/head OOF, conditional on the original
frozen encoder). Evaluation = fold 9 (development), original seed-0
checkpoints. test.pt and external data are never loaded.

Usage:
    python revision_audit/k1_gate_pilot/gate_pilot.py [--seed {0,1,2}]
    (seed 0 writes to results/, seeds 1-2 to results/seed{s}/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from oof_inner_fold import k1_outputs  # noqa: E402
from scripts.train_phase2a import build_selector  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402
from src.data.labels import SUPERCLASSES  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(HERE, "configs/gate_prespec.yaml")))
V5 = LEAD_NAMES.index("V5")
RES = os.path.join(HERE, "results")
ROUND3 = os.path.join(ROOT, "revision_audit/k1_cd_sttc_path/results/val_logits_ABC_seed0.npz")
SCORE_W = {"equal": np.ones(5) / 5, "sttc_weighted": np.array([1, 1, 2, 1, 1]) / 6}  # NORM, MI, STTC, CD, HYP
RANDOM_REPS, RANDOM_SEED = 200, 20261104


def bce(logits, y):
    return np.logaddexp(0.0, logits) - y * logits  # elementwise BCE-with-logits


def features(z, cand, margin):
    n = len(cand)
    onehot = np.zeros((n, 12))
    onehot[np.arange(n), cand] = 1.0
    return np.concatenate([z[:, V5, :], z[np.arange(n), cand, :], onehot, margin[:, None]], axis=1)


def metrics(y, logits) -> dict:
    p = torch.sigmoid(torch.from_numpy(logits.astype(np.float32))).numpy()
    out = {}
    for i, c in enumerate(SUPERCLASSES):
        out[f"{c}_auroc"] = float(roc_auc_score(y[:, i], p[:, i]))
        out[f"{c}_auprc"] = float(average_precision_score(y[:, i], p[:, i]))
    out["macro_auroc"] = float(np.mean([out[f"{c}_auroc"] for c in SUPERCLASSES]))
    out["macro_auprc"] = float(np.mean([out[f"{c}_auprc"] for c in SUPERCLASSES]))
    return out


def main():
    global RES, ROUND3, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, choices=(0, 1, 2, 3, 4, 5))
    SEED = ap.parse_args().seed
    if SEED != 0:
        RES = os.path.join(HERE, "results", f"seed{SEED}")
        ROUND3 = ROUND3.replace("seed0.npz", f"seed{SEED}.npz")
    oof_name = (lambda k: f"oof_heldout_fold{k}.npz") if SEED == 0 else (lambda k: f"oof_seed{SEED}_heldout_fold{k}.npz")
    torch.set_num_threads(8)
    # ---------------- training data: OOF folds 1-8 ----------------
    oof = [dict(np.load(os.path.join(HERE, "results", "oof", oof_name(k)))) for k in range(1, 9)]
    tr = {key: np.concatenate([o[key] for o in oof]) for key in ("z", "y_true", "A_logits", "B_logits", "cand_lead",
                                                                   "margin_top1_top2", "ecg_id", "patient_id")}
    tr_sw = tr["cand_lead"] != V5
    X_tr = features(tr["z"], tr["cand_lead"], tr["margin_top1_top2"])[tr_sw]
    T_tr = (bce(tr["B_logits"], tr["y_true"]) - bce(tr["A_logits"], tr["y_true"]))[tr_sw]  # >0: switching helps
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd < 1e-12] = 1.0
    gate = Ridge(alpha=CFG["gate_model"]["alpha"], fit_intercept=True).fit((X_tr - mu) / sd, T_tr)
    pred_tr = gate.predict((X_tr - mu) / sd)

    # ---------------- fold 9: original seed-0 checkpoints ----------------
    val = torch.load(os.path.join(ROOT, "outputs/phase2a/embeddings/val.pt"))
    y = val["y"].numpy()
    model = Phase2ASelectionModel(build_selector("proposed", 32), latent_dim=32, num_classes=5)
    ckpt = (f"outputs/runs/phase2a_proposed_seed{SEED}/checkpoints/best.pt" if SEED <= 2 else
            f"revision_audit/k1_gate_pilot/phase2a_extra_seeds/runs/phase2a_proposed_seed{SEED}/checkpoints/best.pt")
    model.load_state_dict(torch.load(os.path.join(ROOT, ckpt), map_location="cpu")["model_state_dict"])
    v9 = k1_outputs(model, val["z"])
    if SEED >= 3:
        # Extra seeds: no round-3 saved logits and no matched fixed-V5 head. A/B come directly from the new
        # checkpoint (CPU); the independent C baseline is not reported for these seeds.
        r3 = {"ecg_id": val["ecg_id"].numpy(), "y_true": y, "proposed_lead_idx": v9["cand_lead"],
              "A_logits": v9["A_logits"], "B_logits": v9["B_logits"], "C_logits": None}
    else:
        r3 = np.load(ROUND3)
    checks = {"fold9_AB_source": "round-3 saved logits" if SEED <= 2 else f"new checkpoint {ckpt} (CPU); no C",
        "fold9_ids_labels_match_round3": bool(np.array_equal(r3["ecg_id"], val["ecg_id"].numpy()) and np.array_equal(r3["y_true"], y)),
        "fold9_cand_lead_matches_round3": bool(np.array_equal(v9["cand_lead"], r3["proposed_lead_idx"])),
        "fold9_A_cpu_vs_round3_max_abs": float(np.abs(v9["A_logits"] - r3["A_logits"]).max()),
        "fold9_B_cpu_vs_round3_max_abs": float(np.abs(v9["B_logits"] - r3["B_logits"]).max()),
    }
    # Use the round-3 saved logits as THE fold-9 A/B/C so the endpoints reproduce them exactly.
    A, B, C = r3["A_logits"], r3["B_logits"], r3["C_logits"]
    cand, margin = v9["cand_lead"], v9["margin_top1_top2"]
    sw = cand != V5
    n_sw = int(sw.sum())
    X9 = features(val["z"].numpy(), cand, margin)
    pred9 = np.zeros((len(y), 5))
    pred9[sw] = gate.predict((X9[sw] - mu) / sd)
    actual9 = bce(B, y) - bce(A, y)

    scores9 = {name: pred9 @ w for name, w in SCORE_W.items()}
    scores_tr = {name: pred_tr @ w for name, w in SCORE_W.items()}
    scores9["margin"], scores_tr["margin"] = margin, tr["margin_top1_top2"][tr_sw]

    def policy_logits(switch):
        assert not (switch & ~sw).any()
        return np.where(switch[:, None], A, B)

    def rank_switch(score, q):
        n = int(round(q * n_sw))
        idx = np.where(sw)[0]
        chosen = idx[np.argsort(-score[idx], kind="stable")[:n]]
        s = np.zeros(len(y), bool)
        s[chosen] = True
        return s

    rows, switch_masks = [], {}

    def add(name, budget, switch, extra=None):
        m = metrics(y, policy_logits(switch))
        rows.append({"policy": name, "budget": budget, "actual_switch_rate": float(switch[sw].mean()),
                     "n_switched": int(switch.sum()), **m, **(extra or {})})
        switch_masks[f"{name}@{budget}"] = switch

    none, all_ = np.zeros(len(y), bool), sw.copy()
    add("B (0%, all V5)", "0%", none)
    add("A (100%, selector lead)", "100%", all_)
    if C is not None:
        rows.append({"policy": "C (fixed-V5 head, independent)", "budget": "-", "actual_switch_rate": float("nan"),
                 "n_switched": 0, **metrics(y, C)})
    thresholds = {}
    rng = np.random.default_rng(RANDOM_SEED)
    for q in CFG["budgets_fraction_of_switchable"]:
        tag = f"{int(q * 100)}%"
        for s_name in ("equal", "sttc_weighted", "margin"):
            label = {"equal": "gate equal-weight", "sttc_weighted": "gate STTC x2", "margin": "selector margin"}[s_name]
            add(f"{label} (rank-matched)", tag, rank_switch(scores9[s_name], q))
            thr = float(np.quantile(scores_tr[s_name], 1 - q))
            thresholds[f"{s_name}@{tag}"] = thr
            add(f"{label} (train threshold)", tag, sw & (scores9[s_name] > thr), {"threshold": thr})
        n = int(round(q * n_sw))
        idx = np.where(sw)[0]
        reps = []
        for _ in range(RANDOM_REPS):
            s = np.zeros(len(y), bool)
            s[rng.choice(idx, size=n, replace=False)] = True
            reps.append(metrics(y, policy_logits(s)))
        mean = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        sdv = {f"{k}_sd_over_reps": float(np.std([r[k] for r in reps])) for k in ("CD_auprc", "STTC_auprc", "macro_auprc",
                                                                                  "macro_auroc")}
        rows.append({"policy": f"random (mean of {RANDOM_REPS})", "budget": tag, "actual_switch_rate": n / n_sw,
                     "n_switched": n, **mean, **sdv})

    # endpoint checks against round-3 saved predictions
    checks["endpoint_0pct_logits_vs_round3_B_max_abs"] = float(np.abs(policy_logits(none) - B).max())
    checks["endpoint_100pct_logits_vs_round3_A_max_abs"] = float(np.abs(policy_logits(all_) - A).max())
    checks["n_records"], checks["n_switchable"] = int(len(y)), n_sw
    checks["n_candidate_is_V5"] = int((~sw).sum())
    checks["A_vs_B_on_candidate_V5_records_max_abs"] = float(np.abs(A[~sw] - B[~sw]).max())
    checks["train_n_records"], checks["train_n_switchable"] = int(len(tr_sw)), int(tr_sw.sum())
    checks["train_patients_disjoint_from_fold9"] = bool(not (set(tr["patient_id"]) & set(val["patient_id"].numpy())))
    # descriptive: how well the gate ranks true fold-9 gains (evaluation only)
    gate_diag = {c: float(np.corrcoef(pred9[sw, i], actual9[sw, i])[0, 1]) for i, c in enumerate(SUPERCLASSES)}
    gate_diag["equal_score_vs_actual_mean_gain"] = float(np.corrcoef(scores9["equal"][sw], actual9[sw].mean(1))[0, 1])
    gate_diag["train_in_sample_r2"] = float(gate.score((X_tr - mu) / sd, T_tr))

    os.makedirs(RES, exist_ok=True)
    json.dump({"prespec": CFG, "checks": checks, "thresholds_from_training": thresholds,
               "gate_fold9_pred_vs_actual_gain_pearson": gate_diag, "rows": rows,
               "encoder_caveat": CFG["oof_training"]["encoder_caveat"]},
              open(os.path.join(RES, "gate_pilot_results.json"), "w"), indent=2)
    np.savez(os.path.join(RES, "fold9_per_record.npz"), ecg_id=val["ecg_id"].numpy(), patient_id=val["patient_id"].numpy(),
             y_true=y, A_logits=A, B_logits=B, **({"C_logits": C} if C is not None else {}), cand_lead=cand, margin=margin, switchable=sw,
             pred_gain=pred9, actual_gain=actual9, **{f"score_{k}": v for k, v in scores9.items()},
             **{"switch__" + k.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct"): v
                for k, v in switch_masks.items()})
    write_tables(rows, checks, thresholds, gate_diag)
    plot(rows)


def write_tables(rows, checks, thresholds, gate_diag):
    f = lambda v: "–" if v != v else f"{v:.4f}"
    L = ["## Main table (fold 9, n=2183; switch rate = share of switchable records)", "",
         "| Budget | Policy | Switch rate | CD AUPRC | CD AUROC | STTC AUPRC | STTC AUROC | Macro AUPRC | Macro AUROC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['budget']} | {r['policy']} | {f(r['actual_switch_rate'])} | {f(r['CD_auprc'])} | {f(r['CD_auroc'])} | "
                 f"{f(r['STTC_auprc'])} | {f(r['STTC_auroc'])} | {f(r['macro_auprc'])} | {f(r['macro_auroc'])} |")
    L += ["", "## Other classes", "", "| Budget | Policy | NORM AUPRC | NORM AUROC | MI AUPRC | MI AUROC | HYP AUPRC | HYP AUROC |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['budget']} | {r['policy']} | " + " | ".join(f(r[f"{c}_{m}"]) for c in ("NORM", "MI", "HYP")
                                                                       for m in ("auprc", "auroc")) + " |")
    L += ["", "## Checks", "", "```", json.dumps(checks, indent=1), "```", "",
          "## Thresholds fixed on training (OOF folds 1-8) scores", "", "```", json.dumps(thresholds, indent=1), "```", "",
          "## Gate diagnostics (fold 9, descriptive): Pearson r of predicted vs actual BCE gain", "", "```",
          json.dumps(gate_diag, indent=1), "```"]
    open(os.path.join(RES, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    get = lambda pol, b: next(r for r in rows if r["policy"] == pol and r["budget"] == b)
    B0, A1 = get("B (0%, all V5)", "0%"), get("A (100%, selector lead)", "100%")
    Cr = next((r for r in rows if r["policy"].startswith("C ")), None)
    budgets = ["25%", "50%", "75%"]
    series = [("gate equal-weight (rank-matched)", "Gate, equal weights", "#2a78d6", "o"),
              ("gate STTC x2 (rank-matched)", "Gate, STTC ×2", "#eb6834", "s"),
              ("selector margin (rank-matched)", "Selector margin", "#1baf7a", "D"),
              (f"random (mean of {RANDOM_REPS})", "Random (mean of 200)", "#eda100", "^")]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for pol, lab, col, mk in series:
        pts = [B0] + [get(pol, b) for b in budgets] + [A1]
        ax.plot([p["STTC_auprc"] for p in pts], [p["CD_auprc"] for p in pts], color=col, marker=mk, lw=2, ms=7,
                label=lab, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        mid = get(pol, "50%")
        ax.plot(mid["STTC_auprc"], mid["CD_auprc"], marker=mk, ms=12, color=col, markeredgecolor="#1a1a1a",
                markeredgewidth=1.2, zorder=4)
    for pol, col, mk in (("gate equal-weight (train threshold)", "#2a78d6", "o"), ("gate STTC x2 (train threshold)", "#eb6834", "s")):
        p = get(pol, "50%")
        ax.plot(p["STTC_auprc"], p["CD_auprc"], marker=mk, ms=10, markerfacecolor="white", markeredgecolor=col,
                markeredgewidth=2, ls="none", zorder=5)
    if Cr is not None:
        ax.plot(Cr["STTC_auprc"], Cr["CD_auprc"], marker="X", ms=11, color="#555555", ls="none", zorder=5,
            label="C: matched fixed-V5 head")
    for p, t, dx in [(B0, "B: all V5 (0%)", 6), (A1, "A: selector lead (100%)", 6)] + ([(Cr, "C", 8)] if Cr else []):
        ax.annotate(t, (p["STTC_auprc"], p["CD_auprc"]), xytext=(dx, -12), textcoords="offset points", fontsize=8,
                    color="#333333")
    ax.plot([], [], marker="o", ms=10, markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=2, ls="none",
            label="Gate at 50%, training threshold")
    ax.plot([], [], marker="o", ms=12, color="#bbbbbb", markeredgecolor="#1a1a1a", ls="none", label="50% budget (primary)")
    ax.set_xlabel("STTC AUPRC (fold 9)")
    ax.set_ylabel("CD AUPRC (fold 9)")
    ax.set_title(f"K=1 gate pilot, seed {SEED}: B → 25/50/75% switched → A\n(development, fold 9; rank-matched budgets)",
                 fontsize=9.5, loc="left")
    ax.grid(color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES, f"cd_vs_sttc_auprc.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
