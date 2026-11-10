"""Shared helpers for revision2 (paths, metrics, paired patient bootstrap identical to formal_eval/k1_eval.py)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

R2 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(R2)
RA = os.path.join(ROOT, "revision_audit")
for p in (ROOT, os.path.join(RA, "comm"), os.path.join(RA, "common")):
    if p not in sys.path:
        sys.path.insert(0, p)
from stats import auprc, auroc, boot_p, patient_draws  # noqa: E402

PY = "python"
CACHE = "cache/revision2"
OLD_CACHE = "cache/comm_eval"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = (0, 1, 2)
MNAMES = [f"auroc_{c}" for c in CLASSES] + [f"auprc_{c}" for c in CLASSES] + ["macro_auroc", "macro_auprc"]
G = {}


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def metric_vec(y, p, idx=None):
    """identical to formal_eval/k1_eval.metric_vec"""
    if idx is not None:
        y, p = y[idx], p[idx]
    au = [auroc(y[:, c], p[:, c]) for c in range(5)]
    ap = [auprc(y[:, c], p[:, c]) for c in range(5)]
    return np.array(au + ap + [np.nanmean(au), np.nanmean(ap)])


def _worker(args):
    keys, sl = args
    y, P = G["y"], G["P"]
    return [{k: metric_vec(y, P[k], idx) for k in keys} for idx in G["draws"][sl]]


def bootstrap(y, P, draws, workers=32):
    """P: {key: probs [n, 5]} -> {key: [B, 12]} (same chunking as k1_eval.bootstrap_all, draw order kept)"""
    G.update(y=y, P=P, draws=draws)
    keys, B = list(P), len(draws)
    step = (B + workers - 1) // workers
    with Pool(workers) as pool:
        parts = pool.map(_worker, [(keys, slice(i, min(B, i + step))) for i in range(0, B, step)])
    res = [r for part in parts for r in part]
    return {k: np.stack([r[k] for r in res]) for k in keys}


def paired(obs, boot, x, z, seeds=SEEDS):
    """per-seed difference, then mean over seeds; percentile CI of the seed-mean difference"""
    out = {}
    for i, n in enumerate(MNAMES):
        per_seed = [float(obs[(x, s)][i] - obs[(z, s)][i]) for s in seeds]
        bd = np.mean([boot[(x, s)][:, i] - boot[(z, s)][:, i] for s in seeds], 0)
        out[n] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "sd_between_seeds": float(np.std(per_seed, ddof=1)),
                  "ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))], "p_boot": boot_p(bd)}
    return out


def absolute(obs, boot, m, seeds=SEEDS):
    return {"per_seed": {str(s): dict(zip(MNAMES, map(float, obs[(m, s)]))) for s in seeds},
            "mean": dict(zip(MNAMES, map(float, np.mean([obs[(m, s)] for s in seeds], 0)))),
            "sd_between_seeds": dict(zip(MNAMES, map(float, np.std([obs[(m, s)] for s in seeds], 0, ddof=1)))),
            "ci95_of_seed_mean": {n: [float(np.nanpercentile(np.mean([boot[(m, s)][:, i] for s in seeds], 0), q))
                                      for q in (2.5, 97.5)] for i, n in enumerate(MNAMES)}}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def require_frozen():
    """fold-10 runs refuse to start unless every frozen hash matches"""
    p = os.path.join(R2, "plan_hashes", "FROZEN.sha256")
    assert os.path.exists(p), "fold 10 refused: plan_hashes/FROZEN.sha256 missing"
    r = subprocess.run(["sha256sum", "-c", "--quiet", p], cwd=R2, capture_output=True, text=True)
    assert r.returncode == 0, "fold 10 refused: frozen hash mismatch\n" + r.stdout + r.stderr
