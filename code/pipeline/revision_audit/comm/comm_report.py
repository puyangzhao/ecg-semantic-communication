#!/usr/bin/env python
"""Policy-level and per-SNR contrasts between transmission methods, with a shared patient bootstrap.

Reads the CRN-paired logits written by eval_comm.py and the policies locked on fold 9
(locked_comm_policies.json; the submitted rep_orig policies AWGN tau=-10 / Rayleigh tau=0 are used
unchanged; Rayleigh LS-2 uses the newly locked tau). Estimand per (method, seed, SNR): mean over
the J channel realizations of the macro-AUROC (and macro-AUPRC); policy-level = mean over the
locked SNR distribution (uniform over -15..20 dB). Contrasts: per seed, then mean over seeds 0-2.
Bootstrap: B patient-level resamples shared by every method, seed, SNR and channel; to bound cost
the interval uses the first J_BOOT realizations (point estimates use all J).

Contrasts
  controlled (same lead, same embedding):  jscc_ctrl - rep_ctrl[policy];  sscc[lookup] - rep_ctrl[policy]
  end-to-end:                               jscc_e2e - rep_orig[submitted policy];  jscc_e2e - rep_ctrl[policy]
  adaptive vs best fixed (same family):     rep_orig[policy] - rep_orig[best fixed]; rep_ctrl[policy] - rep_ctrl[best fixed];
                                            sscc[lookup] - sscc[best fixed]
  lead/model effect within the scheme:      rep_ctrl[policy] - rep_orig[policy]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "revision_audit/common"))
from stats import auprc, boot_p, patient_draws  # noqa: E402

CACHE = "cache/comm_eval"
POLICY_SNRS = [-15, -10, -5, 0, 5, 10, 20]
CONDS = ["awgn_perfect", "rayleigh_perfect", "rayleigh_ls2"]
SUBMITTED_TAU = {"awgn_perfect": -10, "rayleigh_perfect": 0}
J_BOOT = 10
G = {}


def policy_map(cond, pol):
    """returns dict name -> {snr: method_key}"""
    c = pol["conditions"][cond]
    tau_orig = SUBMITTED_TAU.get(cond, c["rep_orig"]["threshold_tau_selected"])
    tau_ctrl = c["rep_ctrl"]["threshold_tau_selected"]
    thr = lambda pre, t: {s: f"{pre}_r{4 if (t is not None and s <= float(t)) else 2}" for s in POLICY_SNRS}
    return {"rep_orig_policy": thr("rep_orig", tau_orig), "rep_orig_bestfixed": {s: c["rep_orig"]["best_fixed"] for s in POLICY_SNRS},
            "rep_ctrl_policy": thr("rep_ctrl", tau_ctrl), "rep_ctrl_bestfixed": {s: c["rep_ctrl"]["best_fixed"] for s in POLICY_SNRS},
            "sscc_lookup": {s: c["sscc"]["lookup"][str(s)] for s in POLICY_SNRS},
            "sscc_bestfixed": {s: c["sscc"]["best_fixed"] for s in POLICY_SNRS},
            "jscc_ctrl": {s: "jscc_ctrl" for s in POLICY_SNRS}, "jscc_e2e": {s: "jscc_e2e" for s in POLICY_SNRS},
            "rep_orig_r1": {s: "rep_orig_r1" for s in POLICY_SNRS}}


CONTRASTS = [("jscc_ctrl", "rep_ctrl_policy"), ("sscc_lookup", "rep_ctrl_policy"), ("jscc_e2e", "rep_orig_policy"),
             ("jscc_e2e", "rep_ctrl_policy"), ("rep_orig_policy", "rep_orig_bestfixed"), ("rep_ctrl_policy", "rep_ctrl_bestfixed"),
             ("sscc_lookup", "sscc_bestfixed"), ("rep_ctrl_policy", "rep_orig_policy")]


def macro_auroc_batch(y, P):
    """y [n,5], P [J,n,5] -> [J] tie-aware macro AUROC"""
    out = np.zeros(P.shape[0])
    for c in range(5):
        pos = y[:, c] == 1
        n1, n0 = pos.sum(), (~pos).sum()
        r = rankdata(P[:, :, c], axis=1)
        out += (r[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out / 5


def macro_auprc_batch(y, P):
    return np.array([np.mean([auprc(y[:, c], P[j, :, c]) for c in range(5)]) for j in range(P.shape[0])])


def load(split, seed, cond, method, snr):
    kind, csi = cond.split("_")
    return np.load(os.path.join(CACHE, split, f"seed{seed}", f"{kind}_{csi}", f"{method}__{snr}.npy"))


def _worker(idxs):
    y, L = G["y"], G["L"]
    res = []
    for idx in idxs:
        yy = y[idx]
        vals = {k: macro_auroc_batch(yy, v[:, idx, :]).mean() for k, v in L.items()}
        res.append(vals)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    pol = json.load(open(os.path.join(HERE, "locked_comm_policies.json")))
    meta = np.load(os.path.join(CACHE, a.split, "seed0", "meta.npz"))
    y, pid = meta["y"], meta["patient_id"]
    out = {"split": a.split, "J_boot": J_BOOT, "B": a.B, "conditions": {}}
    # point estimates (all J) for every policy, SNR, seed
    L, point = {}, {}
    for cond in CONDS:
        pm = policy_map(cond, pol)
        for name, mp in pm.items():
            for snr in POLICY_SNRS:
                for s in (0, 1, 2):
                    lg = load(a.split, s, cond, mp[snr], snr)
                    p = expit(lg.astype(np.float64))
                    point[(cond, name, snr, s)] = (macro_auroc_batch(y, p).mean(), macro_auprc_batch(y, p).mean())
                    L[(cond, name, snr, s)] = p[:J_BOOT].astype(np.float32)
    G.update(y=y, L=L)
    draws = patient_draws(pid, a.B, seed=20261120)
    chunks = [draws[i::a.workers] for i in range(a.workers)]
    with Pool(a.workers) as pool:
        boot = [r for part in pool.map(_worker, chunks) for r in part]
    for cond in CONDS:
        pm = policy_map(cond, pol)
        res = {"policies": {n: {str(k): v for k, v in m.items()} for n, m in pm.items()}, "absolute": {}, "contrasts": {}}
        for name in pm:
            res["absolute"][name] = {
                "per_snr_auroc": {snr: float(np.mean([point[(cond, name, snr, s)][0] for s in (0, 1, 2)])) for snr in POLICY_SNRS},
                "per_snr_auprc": {snr: float(np.mean([point[(cond, name, snr, s)][1] for s in (0, 1, 2)])) for snr in POLICY_SNRS},
                "policy_auroc": float(np.mean([point[(cond, name, snr, s)][0] for snr in POLICY_SNRS for s in (0, 1, 2)])),
                "policy_auprc": float(np.mean([point[(cond, name, snr, s)][1] for snr in POLICY_SNRS for s in (0, 1, 2)]))}
        for x, z in CONTRASTS:
            per_snr = {}
            for snr in POLICY_SNRS + ["policy"]:
                snrs = POLICY_SNRS if snr == "policy" else [snr]
                obs_seed = [np.mean([point[(cond, x, q, s)][0] - point[(cond, z, q, s)][0] for q in snrs]) for s in (0, 1, 2)]
                bd = np.array([np.mean([np.mean([b[(cond, x, q, s)] - b[(cond, z, q, s)] for q in snrs]) for s in (0, 1, 2)])
                               for b in boot])
                per_snr[str(snr)] = {"per_seed": list(map(float, obs_seed)), "mean": float(np.mean(obs_seed)),
                                     "ci95": list(map(float, np.percentile(bd, [2.5, 97.5]))), "p_boot": boot_p(bd)}
            res["contrasts"][f"{x}-{z}"] = per_snr
        out["conditions"][cond] = res
        print(cond, {k: (round(v["policy"]["mean"], 4), [round(q, 4) for q in v["policy"]["ci95"]]) for k, v in res["contrasts"].items()},
              flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"comm_report_{a.split}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
