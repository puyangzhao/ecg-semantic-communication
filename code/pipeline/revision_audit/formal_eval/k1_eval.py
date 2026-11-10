#!/usr/bin/env python
"""Formal K=1 clinical evaluation (fold 10 after the lock; --split val is the pre-lock dry run).

Models per seed s in {0,1,2} (pre-fixed seeds; nothing re-selected here):
  original backbone (frozen submitted encoder):
    orig_A  proposed selector + its own head            outputs/runs/phase2a_proposed_seed{s}
    orig_B  same proposed head, V5 forced                (same-head ablation)
    orig_C  matched fixed-V5 head                        revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed{s}
    orig_II submitted fixed Lead II head                 outputs/runs/phase2a_fixed_clinical_k_seed{s}
    orig_GL submitted population-level learned ranking   outputs/runs/phase2a_global_learned_k_seed{s}
  revision backbone (frozen encoder new_seed{s}):
    new_A / new_B / new_C / new_II / new_GL / new_BEST (fold-9 best separately-trained fixed lead)
Statistics: one shared set of 2000 patient-level resamples (rng 20261110) for every comparison on the
split; per resample, per-seed difference then mean over seeds; percentile 95% CI; bootstrap p.
Primary: orig_A - orig_C macro-AUROC. Key secondary (Holm-adjusted together): orig_A - orig_C for
macro-AUPRC, CD AUROC, CD AUPRC, STTC AUROC, STTC AUPRC. Everything else exploratory (unadjusted CI).
Also: frozen development-set thresholds (sens/spec/TP/FP/TN/FN), temperature scaling fitted on fold 9,
descriptive subgroups (sex, age bins) with intervals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
for p in ("revision_audit/common", "revision_audit/backbone", "revision_audit/k1_fair"):
    sys.path.insert(0, os.path.join(ROOT, p))
from scripts.train_phase2a import build_selector  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402
from src.evaluation.subgroup_metrics import recode_age_group  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from stats import auprc, auroc, boot_p, holm, patient_draws  # noqa: E402
from train_heads import build as build_head  # noqa: E402

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = (0, 1, 2)
V5, II = LEAD_NAMES.index("V5"), LEAD_NAMES.index("II")
G = {}


def sig(x):
    return 1 / (1 + np.exp(-x.astype(np.float64)))


def load_orig(split):
    d = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{split}.pt"))
    Z, N = d["z"], d["z"].shape[0]
    out = {}
    v5 = torch.zeros(N, 12)
    v5[:, V5] = 1
    for s in SEEDS:
        def ld(path, method):
            m = Phase2ASelectionModel(build_selector(method, 32), latent_dim=32, num_classes=5)
            m.load_state_dict(torch.load(os.path.join(ROOT, path), map_location="cpu")["model_state_dict"])
            return m.eval()
        with torch.no_grad():
            pa = ld(f"outputs/runs/phase2a_proposed_seed{s}/checkpoints/best.pt", "proposed")
            out[("orig_A", s)] = pa(Z, 1)[0].numpy()
            out[("orig_B", s)] = pa.classifier(pa.fusion(Z, v5)).numpy()
            pc = ld(f"revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed{s}/checkpoints/best.pt", "fixed_clinical_k")
            out[("orig_C", s)] = pc.classifier(pc.fusion(Z, v5)).numpy()
            out[("orig_II", s)] = ld(f"outputs/runs/phase2a_fixed_clinical_k_seed{s}/checkpoints/best.pt", "fixed_clinical_k")(Z, 1)[0].numpy()
            out[("orig_GL", s)] = ld(f"outputs/runs/phase2a_global_learned_k_seed{s}/checkpoints/best.pt", "global_learned_k")(Z, 1)[0].numpy()
            out[("orig_A_lead", s)] = pa.selector.raw_scores(Z).argmax(1).numpy()
    return d, out


def load_new(split, best_lead):
    out = {}
    for s in SEEDS:
        p = os.path.join(ROOT, f"revision_audit/backbone/emb/new_seed{s}/{split}.pt")
        if not os.path.exists(p):
            return None
        Z = torch.load(p)["z"]
        N = Z.shape[0]
        def ld(method, lead=None):
            m = build_head(method, f"new_s{s}", lead)
            tag = f"new_s{s}_{method}{'_' + lead if lead else ''}_seed{s}"
            m.load_state_dict(torch.load(os.path.join(ROOT, f"revision_audit/k1_fair/runs/{tag}/best.pt"))["model_state_dict"])
            return m.eval()
        v5 = torch.zeros(N, 12)
        v5[:, V5] = 1
        with torch.no_grad():
            pa = ld("proposed")
            out[("new_A", s)] = pa(Z, 1)[0].numpy()
            out[("new_B", s)] = pa.logits_with_mask(Z, v5).numpy()
            out[("new_C", s)] = ld("fixed_lead", "V5")(Z, 1)[0].numpy()
            out[("new_II", s)] = ld("fixed_lead", "II")(Z, 1)[0].numpy()
            out[("new_GL", s)] = ld("global_learned_k")(Z, 1)[0].numpy()
            out[("new_BEST", s)] = ld("fixed_lead", best_lead)(Z, 1)[0].numpy()
    return out


def metric_vec(y, p, idx=None):
    if idx is not None:
        y, p = y[idx], p[idx]
    au = [auroc(y[:, c], p[:, c]) for c in range(5)]
    ap = [auprc(y[:, c], p[:, c]) for c in range(5)]
    return np.array(au + ap + [np.nanmean(au), np.nanmean(ap)])


MNAMES = [f"auroc_{c}" for c in CLASSES] + [f"auprc_{c}" for c in CLASSES] + ["macro_auroc", "macro_auprc"]


def _boot_worker(args):
    keys, draw_slice = args
    y, P = G["y"], G["P"]
    return [{k: metric_vec(y, P[k], idx) for k in keys} for idx in G["draws"][draw_slice]]


def bootstrap_all(y, P, draws, workers=24):
    G.update(y=y, P=P, draws=draws)
    keys = list(P)
    B = len(draws)
    chunks = [slice(i, min(B, i + (B + workers - 1) // workers)) for i in range(0, B, (B + workers - 1) // workers)]
    with Pool(workers) as pool:
        parts = pool.map(_boot_worker, [(keys, c) for c in chunks])
    res = [r for part in parts for r in part]
    return {k: np.stack([r[k] for r in res]) for k in keys}  # [B, 12]


def youden_apply(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return tp, fp, tn, fn


def fit_temperature(logits, y):
    lg, yy = torch.tensor(logits, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / logT.exp(), yy)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def ece_brier(y, p, bins=10):
    e, b = [], []
    edges = np.linspace(0, 1, bins + 1)
    for c in range(y.shape[1]):
        idx = np.clip(np.digitize(p[:, c], edges[1:-1]), 0, bins - 1)
        tot = 0.0
        for k in range(bins):
            m = idx == k
            if m.any():
                tot += m.mean() * abs(p[m, c].mean() - y[m, c].mean())
        e.append(tot)
        b.append(np.mean((p[:, c] - y[:, c]) ** 2))
    return float(np.mean(e)), float(np.mean(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    if a.split == "test":
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json")), "fold 10 only after the lock"
    torch.set_num_threads(8)
    d, P = load_orig(a.split)
    y, pid = d["y"].numpy(), d["patient_id"].numpy()
    lead_file = os.path.join(ROOT, "revision_audit/k1_fair/results/new_backbone_best_fixed_lead.json")
    best_lead = json.load(open(lead_file))["locked_best_lead"] if os.path.exists(lead_file) else None
    newP = load_new(a.split, best_lead) if best_lead else None
    if newP:
        P.update(newP)
    leads = {s: P.pop(("orig_A_lead", s)) for s in SEEDS}
    out = {"split": a.split, "n_records": int(len(y)), "n_patients": int(len(np.unique(pid))),
           "prevalence": dict(zip(CLASSES, map(float, y.mean(0)))), "new_backbone_included": bool(newP),
           "new_backbone_best_fixed_lead": best_lead}
    # crosscheck against the submission's saved fold-10 predictions
    if a.split == "test":
        chk = {}
        for s in SEEDS:
            for tag, run in (("orig_A", "proposed"), ("orig_II", "fixed_clinical_k"), ("orig_GL", "global_learned_k")):
                sv = np.load(os.path.join(ROOT, f"outputs/runs/phase2a_{run}_seed{s}/test_raw_K1.npz"))
                assert np.array_equal(sv["ecg_id"], d["ecg_id"].numpy())
                chk[f"{tag}_seed{s}"] = float(np.abs(sv["y_prob"] - sig(P[(tag, s)])).max())
        out["crosscheck_vs_saved_submission_probs_max_abs"] = chk
    probs = {k: sig(v) for k, v in P.items()}
    obs = {k: metric_vec(y, v) for k, v in probs.items()}
    draws = patient_draws(pid, a.B)
    boot = bootstrap_all(y, probs, draws)
    models = sorted({k[0] for k in probs})
    out["absolute"] = {m: {"per_seed": {s: dict(zip(MNAMES, map(float, obs[(m, s)]))) for s in SEEDS},
                           "mean": dict(zip(MNAMES, map(float, np.mean([obs[(m, s)] for s in SEEDS], 0)))),
                           "ci95_of_seed_mean": {n: [float(np.nanpercentile(np.mean([boot[(m, s)][:, i] for s in SEEDS], 0), q))
                                                     for q in (2.5, 97.5)] for i, n in enumerate(MNAMES)}}
                       for m in models}
    comps = [("orig_A", "orig_C"), ("orig_A", "orig_B"), ("orig_B", "orig_C"), ("orig_A", "orig_II"), ("orig_A", "orig_GL")]
    if newP:
        comps += [("new_A", "new_C"), ("new_A", "new_B"), ("new_A", "new_BEST"), ("new_A", "new_II"), ("new_A", "new_GL"),
                  ("new_A", "orig_A"), ("new_C", "orig_C")]
    out["paired"] = {}
    for x, z in comps:
        res = {}
        for i, n in enumerate(MNAMES):
            per_seed = [float(obs[(x, s)][i] - obs[(z, s)][i]) for s in SEEDS]
            bd = np.mean([boot[(x, s)][:, i] - boot[(z, s)][:, i] for s in SEEDS], 0)
            res[n] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                      "ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))], "p_boot": boot_p(bd)}
        out["paired"][f"{x}-{z}"] = res
    prim = out["paired"]["orig_A-orig_C"]
    fam = {n: prim[n]["p_boot"] for n in ("macro_auprc", "auroc_CD", "auprc_CD", "auroc_STTC", "auprc_STTC")}
    out["primary"] = {"endpoint": "orig_A - orig_C macro_auroc", **prim["macro_auroc"]}
    out["key_secondary_holm"] = {n: {**prim[n], "p_holm": p} for n, p in holm(fam).items()}
    # ---- frozen thresholds ----
    thr = json.load(open(os.path.join(ROOT, "revision_audit/thresholds/frozen_thresholds.json")))["thresholds"]
    out["thresholds"] = {}
    for m in [m for m in ("orig_A", "orig_B", "orig_C", "new_A", "new_B", "new_C") if m in thr and m in models]:
        cls = {}
        for ci, c in enumerate(CLASSES):
            rows = []
            for s in SEEDS:
                t = thr[m][str(s)][c]["threshold"]
                tp, fp, tn, fn = youden_apply(y[:, ci], probs[(m, s)][:, ci], t)
                rows.append({"seed": s, "threshold": t, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                             "sens": tp / (tp + fn), "spec": tn / (tn + fp)})
            bs = np.array([[np.mean([(lambda q: (q[0] / (q[0] + q[3]), q[2] / (q[2] + q[1])))(
                youden_apply(y[idx, ci], probs[(m, s)][idx, ci], thr[m][str(s)][c]["threshold"]))[k] for s in SEEDS])
                for k in (0, 1)] for idx in draws[:1000]])
            cls[c] = {"per_seed": rows, "sens_mean": float(np.mean([r["sens"] for r in rows])),
                      "spec_mean": float(np.mean([r["spec"] for r in rows])),
                      "sens_ci95": list(map(float, np.percentile(bs[:, 0], [2.5, 97.5]))),
                      "spec_ci95": list(map(float, np.percentile(bs[:, 1], [2.5, 97.5])))}
        out["thresholds"][m] = cls
    # ---- temperature scaling (fit on fold 9) ----
    dval, Pval = load_orig("val")
    out["calibration"] = {}
    for m in ("orig_A", "orig_C"):
        rows = []
        for s in SEEDS:
            T = fit_temperature(Pval[(m, s)], dval["y"].numpy())
            pre, post = sig(P[(m, s)]), sig(P[(m, s)] / T)
            e0, b0 = ece_brier(y, pre)
            e1, b1 = ece_brier(y, post)
            bs = np.array([[ece_brier(y[i], pre[i])[0], ece_brier(y[i], post[i])[0]] for i in draws[:500]])
            # exploratory: per-class Platt scaling (slope + intercept) fitted on fold 9; class-weighted training
            # shifts each class's intercept, which a single temperature cannot correct
            from sklearn.linear_model import LogisticRegression
            platt = np.zeros_like(pre)
            for c in range(5):
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(Pval[(m, s)][:, c:c + 1], dval["y"].numpy()[:, c])
                platt[:, c] = lr.predict_proba(P[(m, s)][:, c:c + 1])[:, 1]
            e2, b2 = ece_brier(y, platt)
            rows.append({"seed": s, "T": T, "ece_before": e0, "ece_after": e1, "brier_before": b0, "brier_after": b1,
                         "ece_before_ci95": list(map(float, np.percentile(bs[:, 0], [2.5, 97.5]))),
                         "ece_after_ci95": list(map(float, np.percentile(bs[:, 1], [2.5, 97.5]))),
                         "ece_platt_exploratory": e2, "brier_platt_exploratory": b2})
        out["calibration"][m] = rows
    # ---- descriptive subgroups (orig_A vs orig_C) ----
    age = recode_age_group(d["age"].numpy())
    sex = d["sex"].numpy().astype(int).astype(str)
    out["subgroups"] = {}
    for gname, gv in (("sex", sex), ("age", age)):
        res = {}
        for g in sorted(set(gv)):
            msk = gv == g
            cell = {"n": int(msk.sum()), "n_patients": int(len(np.unique(pid[msk]))),
                    "positives": dict(zip(CLASSES, map(int, y[msk].sum(0))))}
            for m in ("orig_A", "orig_C"):
                vals = []
                for s in SEEDS:
                    au = [auroc(y[msk, c], probs[(m, s)][msk, c]) if min(y[msk, c].sum(), (1 - y[msk, c]).sum()) >= 10 else np.nan
                          for c in range(5)]
                    vals.append(au)
                cell[f"{m}_macro_auroc_valid_classes"] = float(np.nanmean(np.nanmean(vals, 0)))
                cell[f"{m}_per_class_auroc"] = dict(zip(CLASSES, map(float, np.nanmean(vals, 0))))
            bs = []
            for idx in draws[:500]:
                ii = idx[msk[idx]]
                vv = []
                for m in ("orig_A", "orig_C"):
                    vv.append(np.nanmean([np.nanmean([auroc(y[ii, c], probs[(m, s)][ii, c]) if min(y[ii, c].sum(), (1 - y[ii, c]).sum()) >= 10 else np.nan
                                                      for c in range(5)]) for s in SEEDS]))
                bs.append(vv)
            bs = np.array(bs)
            cell["orig_A_ci95"] = list(map(float, np.nanpercentile(bs[:, 0], [2.5, 97.5])))
            cell["orig_C_ci95"] = list(map(float, np.nanpercentile(bs[:, 1], [2.5, 97.5])))
            cell["descriptive_only"] = bool(min(cell["positives"].values()) < 10)
            res[g] = cell
        out["subgroups"][gname] = res
    out["selection_frequency_orig_A"] = {s: dict(zip(LEAD_NAMES, map(float, np.bincount(leads[s], minlength=12) / len(y))))
                                         for s in SEEDS}
    tag = a.split
    json.dump(out, open(os.path.join(HERE, "results", f"k1_eval_{tag}.json"), "w"), indent=2, default=float)
    np.savez(os.path.join(HERE, "results", f"k1_logits_{tag}.npz"), y=y, patient_id=pid, ecg_id=d["ecg_id"].numpy(),
             **{f"{m}_seed{s}": v for (m, s), v in P.items()})
    print(json.dumps({"primary": out["primary"], "key_secondary_holm": out["key_secondary_holm"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
