"""Policy locking and paired reporting for the revision2 transmission analyses.

lock(): the rule body of revision_audit/comm/lock_policies.main(), unchanged, parametrized by condition names and
option families (checked to reproduce comm/locked_comm_policies.json, PLAN §3).
report(): the estimand and bootstrap of revision_audit/comm/comm_report.py (realization mean per method/seed/SNR;
equal-weight mean over the seven policy SNRs; per-seed contrast then mean over seeds; patient_draws(pid, B,
seed=20261120) shared by everything; first J_BOOT=10 realizations inside the bootstrap), applied to macro-AUROC and
macro-AUPRC, with contrasts that may compare two conditions (e.g. rho = 0.95 vs rho = 1).
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
from scipy.special import expit

import r2common as C  # noqa: F401  (sets sys.path)
import comm_report as CR
import lock_policies as LP
from stats import boot_p, patient_draws

POLICY_SNRS = CR.POLICY_SNRS
SEEDS = (0, 1, 2)
G = {}


def lock(S, conds, families, submitted, rules_note=""):
    TAUS = LP.TAUS
    m = lambda cond, snr, opt, key="macro_auroc": float(np.mean([x["conditions"][f"{cond}_{snr}"][opt][key] for x in S]))
    out = {"rules": LP.__doc__ + rules_note, "policy_snrs": POLICY_SNRS, "conditions": {}}
    for cond in conds:
        oc = {}
        for fam, opts in families(cond).items():
            score = {o: (np.mean([m(cond, s, o) for s in POLICY_SNRS]), np.mean([m(cond, s, o, "macro_auprc") for s in POLICY_SNRS]))
                     for o in opts}
            best_fixed = sorted(opts, key=lambda o: (-score[o][0], -score[o][1], opts.index(o)))[0]
            lookup = {}
            for snr in POLICY_SNRS:
                lookup[snr] = sorted(opts, key=lambda o: (-m(cond, snr, o), -m(cond, snr, o, "macro_auprc"), opts.index(o)))[0]
            fam_res = {"best_fixed": best_fixed, "best_fixed_target": score[best_fixed][0], "lookup": lookup,
                       "lookup_target": float(np.mean([m(cond, s, lookup[s]) for s in POLICY_SNRS]))}
            if fam.startswith("rep"):
                pre = fam + "_r"
                tau_score = {}
                for t in TAUS:
                    pick = {s: f"{pre}{4 if s <= t else 2}" for s in POLICY_SNRS}
                    tau_score[t] = (np.mean([m(cond, s, pick[s]) for s in POLICY_SNRS]),
                                    np.mean([m(cond, s, pick[s], "macro_auprc") for s in POLICY_SNRS]))
                tau = sorted(TAUS, key=lambda t: (-tau_score[t][0], -tau_score[t][1], t))[0]
                fam_res["threshold_tau_selected"] = None if tau == -np.inf else tau
                fam_res["threshold_target"] = tau_score[tau][0]
                if fam == "rep_orig" and cond in submitted:
                    fam_res["submitted_tau"] = submitted[cond]
                    fam_res["submitted_target_on_this_val_harness"] = tau_score[submitted[cond]][0]
            oc[fam] = fam_res
        oc["jscc_ctrl_target"] = float(np.mean([m(cond, s, "jscc_ctrl") for s in POLICY_SNRS]))
        oc["jscc_e2e_target"] = float(np.mean([m(cond, s, "jscc_e2e") for s in POLICY_SNRS]))
        out["conditions"][cond] = oc
    return json.loads(json.dumps(out, default=str))


def _worker(idxs):
    y, L = G["y"], G["L"]
    return [{k: (CR.macro_auroc_batch(y[idx], v[:, idx, :]).mean(), CR.macro_auprc_batch(y[idx], v[:, idx, :]).mean())
             for k, v in L.items()} for idx in idxs]


def report(y, pid, conds, pmap, load, contrasts, B=1000, workers=48, snrs=POLICY_SNRS):
    """conds: list of condition names; pmap(cond) -> {name: {snr: file_key}}; load(cond, file_key, snr, seed) -> logits.
    contrasts: list of (label, (cond_x, name_x), (cond_z, name_z))."""
    L, point = {}, {}
    maps = {c: pmap(c) for c in conds}
    for cond in conds:
        for name, mp in maps[cond].items():
            for snr in snrs:
                for s in SEEDS:
                    key = (cond, mp[snr], snr, s)
                    if key in point:
                        continue
                    p = expit(load(cond, mp[snr], snr, s).astype(np.float64))
                    point[key] = (CR.macro_auroc_batch(y, p).mean(), CR.macro_auprc_batch(y, p).mean())
                    L[key] = p[:CR.J_BOOT].astype(np.float32)
    G.update(y=y, L=L)
    draws = patient_draws(pid, B, seed=20261120)
    with Pool(workers) as pool:
        parts = pool.map(_worker, [draws[i::workers] for i in range(workers)])
    boot = [None] * len(draws)
    for i, part in enumerate(parts):
        for j, r in enumerate(part):
            boot[i + j * workers] = r
    P = lambda c, n, q, s, i: point[(c, maps[c][n][q], q, s)][i]
    Bv = lambda b, c, n, q, s, i: b[(c, maps[c][n][q], q, s)][i]
    out = {"J_boot": CR.J_BOOT, "B": B, "bootstrap_seed": 20261120, "policy_snrs": snrs, "absolute": {}, "policies": {}, "contrasts": {}}
    for cond in conds:
        out["policies"][cond] = {n: {str(k): v for k, v in mm.items()} for n, mm in maps[cond].items()}
        out["absolute"][cond] = {}
        for name in maps[cond]:
            ab = {}
            for i, met in ((0, "auroc"), (1, "auprc")):
                ab[f"per_snr_{met}"] = {str(q): float(np.mean([P(cond, name, q, s, i) for s in SEEDS])) for q in snrs}
                ab[f"per_snr_{met}_per_seed"] = {str(q): [float(P(cond, name, q, s, i)) for s in SEEDS] for q in snrs}
                ps = [float(np.mean([P(cond, name, q, s, i) for q in snrs])) for s in SEEDS]
                ab[f"policy_{met}_per_seed"] = ps
                ab[f"policy_{met}"] = float(np.mean(ps))
                ab[f"policy_{met}_sd_between_seeds"] = float(np.std(ps, ddof=1))
            out["absolute"][cond][name] = ab
    for label, (cx, x), (cz, z) in contrasts:
        per = {}
        for snr in list(snrs) + ["policy"]:
            qs = snrs if snr == "policy" else [snr]
            e = {}
            for i, met in ((0, "auroc"), (1, "auprc")):
                obs = [np.mean([P(cx, x, q, s, i) - P(cz, z, q, s, i) for q in qs]) for s in SEEDS]
                bd = np.array([np.mean([np.mean([Bv(b, cx, x, q, s, i) - Bv(b, cz, z, q, s, i) for q in qs]) for s in SEEDS])
                               for b in boot])
                e[met] = {"per_seed": list(map(float, obs)), "mean": float(np.mean(obs)), "sd_between_seeds": float(np.std(obs, ddof=1)),
                          "ci95": list(map(float, np.percentile(bd, [2.5, 97.5]))), "p_boot": boot_p(bd)}
            per[str(snr)] = e
        out["contrasts"][label] = per
    return out
