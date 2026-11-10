#!/usr/bin/env python
"""Stage 3: multi-seed summary + patient-level paired bootstrap for the K=1
gate pilot (fold 9, development). Reads each seed's stage-2 outputs; no
training, no fold 10, no external data.

Primary comparison (pre-specified 50% of switchable records):
  equal-weight gate (rank-matched) vs (1) random routing with the same number
  of switches, 200 pre-specified routings, and (2) selector top1-top2 margin.
Separate table: training-threshold execution, where each policy uses its own
training-quantile threshold record by record; the random comparator there
switches the same number of records the threshold gate actually switched
(200 new routings, rng 20261105 + seed).

Random baselines: the 200 routings of stage 2 are replayed from the
pre-specified RNG (20261104, same draw order), each routing's metric is
computed separately and then averaged -- predictions are never averaged. The
replay is verified against the stage-2 saved random means.

Bootstrap: patients resampled with replacement (all records kept); ONE draw
per resample shared by every policy and all seeds; per-seed
differences first, then mean over seeds. Interval is conditional on these
trained models and the routings as implemented.

Usage:
    python revision_audit/k1_gate_pilot/multiseed_bootstrap.py                                   # seeds 0-2 -> results/multiseed
    python revision_audit/k1_gate_pilot/multiseed_bootstrap.py --seeds 0,1,2,3,4,5 --out multiseed6
    # rebuild tables.md from an existing multiseed_bootstrap.json (no bootstrap rerun)
    python revision_audit/k1_gate_pilot/multiseed_bootstrap.py --seeds 0,1,2,3,4,5 --out multiseed6 --tables-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from src.data.labels import SUPERCLASSES  # noqa: E402
from src.evaluation.hierarchical_bootstrap import (  # noqa: E402
    _vectorized_macro_auprc_batch,
    _vectorized_macro_auroc_batch,
)

SEEDS = (0, 1, 2)
BUDGETS = (0.25, 0.50, 0.75)
RANDOM_REPS, RANDOM_SEED, THR_RANDOM_SEED_BASE = 200, 20261104, 20261105
N_BOOT, BOOT_SEED, N_WORKERS = 2000, 20261106, 24
OUT = os.path.join(HERE, "results", "multiseed")
METRICS = ("CD_auprc", "STTC_auprc", "macro_auprc", "macro_auroc", "CD_auroc")
CD, STTC = SUPERCLASSES.index("CD"), SUPERCLASSES.index("STTC")
FIXED = ("gate_eq", "gate_sttc", "margin", "gate_eq_thr", "gate_sttc_thr", "margin_thr")
COMPARISONS = {  # name: (a, b) where b may be a random block
    "gate_eq_vs_random@50": ("gate_eq", "RANDOM"),
    "gate_eq_vs_margin@50": ("gate_eq", "margin"),
    "gate_sttc_vs_random@50": ("gate_sttc", "RANDOM"),
    "gate_sttc_vs_margin@50": ("gate_sttc", "margin"),
    "thr_gate_eq_vs_matched_random": ("gate_eq_thr", "RANDOM_THR"),
    "thr_gate_eq_vs_margin_thr": ("gate_eq_thr", "margin_thr"),
}


ROOT_OVERRIDE = None  # --root: read <root>/seed{s}/per_record.npz + gate_eval_results.json (revision fold-10 eval)
PER_RECORD, RESULTS_JSON = "fold9_per_record.npz", "gate_pilot_results.json"


def seed_dir(s):
    if ROOT_OVERRIDE:
        return os.path.join(ROOT_OVERRIDE, f"seed{s}")
    return os.path.join(HERE, "results") if s == 0 else os.path.join(HERE, "results", f"seed{s}")


def key(policy, budget):
    k = f"{policy}@{budget}"
    return "switch__" + k.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")


def probs_of(A, B, switch):
    return torch.sigmoid(torch.from_numpy(np.where(switch[:, None], A, B).astype(np.float32))).numpy()


def sk_metrics(y, p):
    au = [roc_auc_score(y[:, i], p[:, i]) for i in range(5)]
    ap = [average_precision_score(y[:, i], p[:, i]) for i in range(5)]
    return np.array([ap[CD], ap[STTC], np.mean(ap), np.mean(au), au[CD]])


def _tie_aware_auroc_batch(y, S):
    """y [n] binary, S [R, n] -> [R] AUROC with average ranks for ties (sklearn-identical).
    Added 2026-11-04 after the fold-10 implementation check failed (1.2e-6 > 1e-6) on exactly tied scores
    with the arbitrary-tie helper; failure occurred before any fold-10 bootstrap output existed."""
    from scipy.stats import rankdata
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    r = rankdata(S, axis=1)
    return (r[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def vec_metrics(y, P):
    """P: [R, n, 5] -> [R, len(METRICS)]"""
    au = np.stack([_tie_aware_auroc_batch(y[:, c], P[:, :, c]) for c in range(5)], axis=1)
    ap = np.stack([_vectorized_macro_auprc_batch(y[:, [c]], P[:, :, [c]]) for c in range(5)], axis=1)
    return np.stack([ap[:, CD], ap[:, STTC], ap.mean(1), au.mean(1), au[:, CD]], axis=1)


def load_seed(s):
    d = np.load(os.path.join(seed_dir(s), PER_RECORD))
    res = json.load(open(os.path.join(seed_dir(s), RESULTS_JSON)))
    y, A, B, sw = d["y_true"], d["A_logits"], d["B_logits"], d["switchable"]
    masks = {"gate_eq": d[key("gate equal-weight (rank-matched)", "50%")],
             "gate_sttc": d[key("gate STTC x2 (rank-matched)", "50%")],
             "margin": d[key("selector margin (rank-matched)", "50%")],
             "gate_eq_thr": d[key("gate equal-weight (train threshold)", "50%")],
             "gate_sttc_thr": d[key("gate STTC x2 (train threshold)", "50%")],
             "margin_thr": d[key("selector margin (train threshold)", "50%")]}
    idx, n_sw = np.where(sw)[0], int(sw.sum())
    rng = np.random.default_rng(RANDOM_SEED)  # replay stage-2 draw order exactly
    rand = {}
    for q in BUDGETS:
        n = int(round(q * n_sw))
        draws = []
        for _ in range(RANDOM_REPS):
            m = np.zeros(len(y), bool)
            m[rng.choice(idx, size=n, replace=False)] = True
            draws.append(m)
        rand[q] = draws
    n_thr = int(masks["gate_eq_thr"].sum())
    rng_t = np.random.default_rng(THR_RANDOM_SEED_BASE + s)
    rand_thr = []
    for _ in range(RANDOM_REPS):
        m = np.zeros(len(y), bool)
        m[rng_t.choice(idx, size=n_thr, replace=False)] = True
        rand_thr.append(m)
    stack = [masks[k] for k in FIXED] + rand[0.5] + rand_thr
    P = np.stack([probs_of(A, B, m) for m in stack])  # [6 + 400, n, 5]
    return {"y": y, "pid": d["patient_id"], "ecg": d["ecg_id"], "P": P, "masks": masks, "rand50": rand[0.5],
            "n_sw": n_sw, "n_thr": n_thr, "rows": res["rows"], "checks": res["checks"]}


def per_seed_stat(M):
    """M: [406, len(METRICS)] metrics of the stack -> dict comparison -> [len(METRICS)] difference."""
    pol = {k: M[i] for i, k in enumerate(FIXED)}
    pol["RANDOM"] = M[len(FIXED):len(FIXED) + RANDOM_REPS].mean(0)
    pol["RANDOM_THR"] = M[len(FIXED) + RANDOM_REPS:].mean(0)
    return {c: pol[a] - pol[b] for c, (a, b) in COMPARISONS.items()}, pol


_G = {}


def _init(seeds_data, seeds, root=None):
    global SEEDS, ROOT_OVERRIDE
    _G["data"] = seeds_data
    SEEDS = seeds  # workers may be spawned without main()'s globals
    ROOT_OVERRIDE = root


def _boot_chunk(draws):
    out = []
    for idx in draws:
        per = []
        for s in SEEDS:
            d = _G["data"][s]
            diff, _ = per_seed_stat(vec_metrics(d["y"][idx], d["P"][:, idx, :]))
            per.append(np.stack([diff[c] for c in COMPARISONS]))
        out.append(np.mean(per, axis=0))  # [n_comp, n_metric]
    return out


def main():
    global SEEDS, OUT, ROOT_OVERRIDE, PER_RECORD, RESULTS_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="alternative cohort root (default: fold-9 pilot results)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="multiseed")
    ap.add_argument("--tables-only", action="store_true",
                    help="rewrite tables.md from the saved multiseed_bootstrap.json; no bootstrap")
    a = ap.parse_args()
    SEEDS = tuple(int(x) for x in a.seeds.split(","))
    OUT = os.path.join(HERE, "results", a.out)
    if a.root:
        ROOT_OVERRIDE, PER_RECORD, RESULTS_JSON = a.root, "per_record.npz", "gate_eval_results.json"
        OUT = os.path.join(a.root, a.out)
    os.makedirs(OUT, exist_ok=True)
    data = {s: load_seed(s) for s in SEEDS}
    if a.tables_only:
        rep = json.load(open(os.path.join(OUT, "multiseed_bootstrap.json")))
        saved_seeds = tuple(int(k) for k in rep["observed"])
        assert saved_seeds == SEEDS, (saved_seeds, SEEDS)
        observed = {int(k): v for k, v in rep["observed"].items()}
        write_tables(rep["summary"], observed, rep["checks"], data)
        return
    for s in SEEDS[1:]:
        assert np.array_equal(data[s]["pid"], data[0]["pid"]) and np.array_equal(data[s]["ecg"], data[0]["ecg"])
        assert np.array_equal(data[s]["y"], data[0]["y"])
    y, pid = data[0]["y"], data[0]["pid"]

    # ---- observed (sklearn) + replay verification ----
    observed, checks = {}, {}
    for s in SEEDS:
        d = data[s]
        M = np.stack([sk_metrics(y, p) for p in d["P"]])
        Mv = vec_metrics(y, d["P"])
        checks[f"seed{s}_vectorized_vs_sklearn_max_abs"] = float(np.abs(M - Mv).max())
        diff, pol = per_seed_stat(M)
        saved_rand = next(r for r in d["rows"] if r["policy"].startswith("random") and r["budget"] == "50%")
        saved_gate = next(r for r in d["rows"] if r["policy"] == "gate equal-weight (rank-matched)" and r["budget"] == "50%")
        checks[f"seed{s}_random_replay_vs_saved_max_abs"] = float(max(abs(pol["RANDOM"][i] - saved_rand[m])
                                                                      for i, m in enumerate(METRICS)))
        checks[f"seed{s}_gate_eq_vs_saved_max_abs"] = float(max(abs(pol["gate_eq"][i] - saved_gate[m])
                                                               for i, m in enumerate(METRICS)))
        checks[f"seed{s}_n_switchable"], checks[f"seed{s}_n_switched_gate_eq@50"] = d["n_sw"], int(d["masks"]["gate_eq"].sum())
        checks[f"seed{s}_threshold_actual_switch_rate"] = {k: float(d["masks"][k].sum() / d["n_sw"])
                                                           for k in ("gate_eq_thr", "gate_sttc_thr", "margin_thr")}
        observed[s] = {"diff": {c: dict(zip(METRICS, map(float, v))) for c, v in diff.items()},
                       "policy": {k: dict(zip(METRICS, map(float, v))) for k, v in pol.items()}}
    assert all(checks[f"seed{s}_random_replay_vs_saved_max_abs"] < 1e-12 for s in SEEDS), checks
    assert all(checks[f"seed{s}_vectorized_vs_sklearn_max_abs"] < 1e-6 for s in SEEDS), checks

    # ---- bootstrap: one patient draw shared by all policies and seeds ----
    rng = np.random.default_rng(BOOT_SEED)
    patients = np.unique(pid)
    p2i = {p: np.where(pid == p)[0] for p in patients}
    draws = [np.concatenate([p2i[p] for p in rng.choice(patients, size=len(patients), replace=True)])
             for _ in range(N_BOOT)]
    chunks = [draws[i::N_WORKERS] for i in range(N_WORKERS)]
    torch.set_num_threads(1)
    with Pool(N_WORKERS, initializer=_init, initargs=(data, SEEDS, ROOT_OVERRIDE)) as pool:
        boot = np.stack([x for part in pool.map(_boot_chunk, chunks) for x in part])  # [N_BOOT, n_comp, n_metric]

    summary = {}
    for ci, c in enumerate(COMPARISONS):
        summary[c] = {}
        for mi, m in enumerate(METRICS):
            per_seed = [observed[s]["diff"][c][m] for s in SEEDS]
            summary[c][m] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                             "ci95": [float(np.percentile(boot[:, ci, mi], 2.5)), float(np.percentile(boot[:, ci, mi], 97.5))],
                             "all_seeds_positive": bool(all(v > 0 for v in per_seed))}
    report = {"status": (f"cohort root {ROOT_OVERRIDE}; {len(SEEDS)} trained seeds {SEEDS}" if ROOT_OVERRIDE else
                         f"development, fold 9; {len(SEEDS)} trained seeds {SEEDS}; fold 10 not loaded; training-stability analysis on the same validation fold"),
              "interval_meaning": "patient-sampling uncertainty conditional on these trained models and the routings as implemented",
              "bootstrap": {"n_boot": N_BOOT, "rng": BOOT_SEED, "shared_draw": "all policies and all seeds",
                            "aggregation": "per-seed difference, then mean over seeds",
                            "random": "200 routings per seed, metric per routing, then mean (predictions never averaged)"},
              "checks": checks, "observed": observed, "summary": summary}
    json.dump(report, open(os.path.join(OUT, "multiseed_bootstrap.json"), "w"), indent=2)
    write_tables(summary, observed, checks, data)


def write_tables(summary, observed, checks, data):
    f = lambda v: f"{v:+.4f}"
    names = {"gate_eq_vs_random@50": "Equal gate − random", "gate_eq_vs_margin@50": "Equal gate − margin",
             "gate_sttc_vs_random@50": "STTC×2 gate − random", "gate_sttc_vs_margin@50": "STTC×2 gate − margin",
             "thr_gate_eq_vs_matched_random": "Equal gate (thr) − random (same count)",
             "thr_gate_eq_vs_margin_thr": "Equal gate (thr) − margin (thr)"}

    def block(comps):
        L = ["| Comparison | Metric | " + " | ".join(f"Seed {s}" for s in SEEDS) + f" | Mean [95% patient-bootstrap] | All {len(SEEDS)} > 0 |",
             "|---|---|" + "---|" * len(SEEDS) + "---|---|"]
        for c in comps:
            for m in METRICS:
                r = summary[c][m]
                L.append(f"| {names[c]} | {m} | " + " | ".join(f(v) for v in r["per_seed"]) +
                         f" | **{f(r['mean'])}** [{f(r['ci95'][0])}, {f(r['ci95'][1])}] | {'yes' if r['all_seeds_positive'] else 'no'} |")
        return L

    n = len(SEEDS)
    title = f"cohort {os.path.basename(ROOT_OVERRIDE.rstrip('/'))}" if ROOT_OVERRIDE else "fold 9, development"
    L = [f"# {n}-seed K=1 gate pilot ({title}; seeds {', '.join(map(str, SEEDS))})", "",
         f"Intervals: patient-level paired bootstrap of the {n}-seed mean difference, conditional on these {n} trained models "
         "and the implemented routings. Random = mean over 200 routings of each routing's metric.", "",
         "## Primary: 50% of switchable records, rank-matched", ""] + block(["gate_eq_vs_random@50", "gate_eq_vs_margin@50"])
    L += ["", "## Secondary: STTC×2 gate at 50%", ""] + block(["gate_sttc_vs_random@50", "gate_sttc_vs_margin@50"])
    L += ["", "## Training-threshold execution (record by record), 50% target", "",
          "| Seed | Equal gate actual switch rate | STTC×2 gate | Margin |", "|---|---|---|---|"]
    for s in SEEDS:
        r = checks[f"seed{s}_threshold_actual_switch_rate"]
        L.append(f"| {s} | {r['gate_eq_thr']:.3f} | {r['gate_sttc_thr']:.3f} | {r['margin_thr']:.3f} |")
    L += [""] + block(["thr_gate_eq_vs_matched_random", "thr_gate_eq_vs_margin_thr"])
    L += ["", "## Per-seed absolute values, all pre-specified rows (A/B/C endpoints, 25/50/75%, STTC×2, thresholds, random)", "",
          "| Seed | Budget | Policy | Switch rate | CD AUPRC | STTC AUPRC | Macro AUPRC | Macro AUROC | CD AUROC |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        for r in data[s]["rows"]:
            sr = "–" if r["actual_switch_rate"] != r["actual_switch_rate"] else f"{r['actual_switch_rate']:.3f}"
            L.append(f"| {s} | {r['budget']} | {r['policy']} | {sr} | {r['CD_auprc']:.4f} | {r['STTC_auprc']:.4f} | "
                     f"{r['macro_auprc']:.4f} | {r['macro_auroc']:.4f} | {r['CD_auroc']:.4f} |")
    L += ["", "## Checks", "", "```", json.dumps(checks, indent=1), "```"]
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
