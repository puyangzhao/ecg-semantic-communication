#!/usr/bin/env python
"""PLAN §4: lock policies at N = 8 and 32 on fold 9 (existing rule), and report the budget comparison.
N = 16 uses the existing locked policies (comm/locked_comm_policies.json) and cached logits unchanged.

  --lock            fold 9
  --report SPLIT    val or test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import r2common as C  # noqa: E402
import r2comm as RC  # noqa: E402
import comm_report as CR  # noqa: E402
from models_n import R_OPTIONS  # noqa: E402
from sscc_n import CONFIGS_N, NOT_APPLICABLE  # noqa: E402

RES = os.path.join(C.R2, "results", "budget")
NEW_N = (8, 32)
KINDS = ("awgn", "rayleigh")


def cond_name(kind, n):
    return f"{kind}_perfect" if n == 16 else f"{kind}_perfect_n{n}"


def families(cond):
    n = int(cond.rsplit("_n", 1)[1])
    return {"rep_orig": [f"rep_orig_r{r}" for r in R_OPTIONS[n]], "rep_ctrl": [f"rep_ctrl_r{r}" for r in R_OPTIONS[n]],
            "sscc": [f"sscc_{k}" for k in CONFIGS_N[n]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    if a.lock:
        pol = {"conditions": {}}
        for n in NEW_N:
            S = [json.load(open(os.path.join(RES, f"budget_eval_val_n{n}_seed{s}.json"))) for s in C.SEEDS]
            p = RC.lock(S, [cond_name(k, n) for k in KINDS], families, {},
                        f"\n[revision2] applied unchanged at N = {n}; the original scheme is a frozen transfer and uses the rule-selected tau.")
            pol["rules"] = p["rules"]
            pol["conditions"].update(p["conditions"])
        old = json.load(open(os.path.join(C.RA, "comm/locked_comm_policies.json")))
        for k in KINDS:
            pol["conditions"][cond_name(k, 16)] = old["conditions"][cond_name(k, 16)]
        pol["n16_source"] = "revision_audit/comm/locked_comm_policies.json (unchanged)"
        C.dump(pol, os.path.join(RES, "locked_policies_budget.json"))
        for c, v in pol["conditions"].items():
            print(c, v["rep_orig"].get("threshold_tau_selected"), v["rep_ctrl"].get("threshold_tau_selected"), v["sscc"]["best_fixed"])
    if a.report:
        sp = a.report
        pol = json.load(open(os.path.join(RES, "locked_policies_budget.json")))
        conds = [cond_name(k, n) for k in KINDS for n in (8, 16, 32)]

        def cache_of(cond):
            return os.path.join(C.OLD_CACHE, sp) if cond.endswith("_perfect") else os.path.join(C.CACHE, "budget", sp)
        m0 = np.load(os.path.join(cache_of(conds[0]), "seed0", "meta.npz"))
        y, pid = m0["y"], m0["patient_id"]
        for c in conds:
            for s in C.SEEDS:
                ms = np.load(os.path.join(cache_of(c), f"seed{s}", "meta.npz"))
                assert np.array_equal(ms["y"], y) and np.array_equal(ms["patient_id"], pid)
        load = lambda c, k, q, s: np.load(os.path.join(cache_of(c), f"seed{s}", c, f"{k}__{q}.npy"))
        pmap = lambda c: {k: v for k, v in CR.policy_map(c, pol).items() if k != "rep_orig_r1"}
        within = [("jscc_e2e", "rep_orig_policy"), ("jscc_e2e", "rep_ctrl_policy"), ("jscc_ctrl", "rep_ctrl_policy"),
                  ("jscc_ctrl", "rep_orig_policy"), ("sscc_lookup", "rep_ctrl_policy"), ("rep_ctrl_policy", "rep_orig_policy")]
        con = [(f"{c}:{x}-{z}", (c, x), (c, z)) for c in conds for x, z in within]
        con += [(f"{k}:{m}:n{n}-n16", (cond_name(k, n), m), (cond_name(k, 16), m)) for k in KINDS for n in NEW_N
                for m in ("jscc_e2e", "jscc_ctrl", "rep_ctrl_policy", "rep_orig_policy", "sscc_lookup")]
        rep = RC.report(y, pid, conds, pmap, load, con, workers=a.workers)
        # budget accounting table (energy from the transmit checks of this split; N=16 from the locked comm_eval JSONs)
        acc = {}
        for n in (8, 16, 32):
            for s in C.SEEDS:
                src = (os.path.join(C.RA, "comm/results", f"comm_eval_{sp}_seed{s}.json") if n == 16
                       else os.path.join(RES, f"budget_eval_{sp}_n{n}_seed{s}.json"))
                for k, e in json.load(open(src))["energy_checks"].items():
                    r = acc.setdefault(f"n{n}:{k}", {"data_uses": n, "pilots": 0, "energy_mean": [], "energy_min": [], "energy_max": []})
                    for q in ("energy_mean", "energy_min", "energy_max"):
                        r[q].append(e[q])
                    if "info_bits" in e:
                        r["info_bits"] = e["info_bits"]
        for k, r in acc.items():
            for q in ("energy_mean", "energy_min", "energy_max"):
                r[q] = float(np.mean(r[q])) if q == "energy_mean" else float((min if q == "energy_min" else max)(r[q]))
            n = int(k.split(":")[0][1:])
            name = k.split(":")[1]
            if name.startswith("sscc_") and n == 16:
                r["info_bits"] = int(name.split("_")[1][1:])
            r["training"] = ("frozen transfer of the submitted checkpoint" if name.startswith("rep_orig") and n != 16
                             else "existing model" if n == 16 else "retrained at this budget")
        rep.update(split=sp, post_hoc_revision_analysis=True, accounting=acc,
                   not_applicable={"8": {"rep": "none", "sscc": NOT_APPLICABLE[8]}, "32": {"rep": "r = 1 (m would be 32 > 16)"}},
                   n16_source="existing models, cached logits and locked policies")
        C.dump(rep, os.path.join(RES, f"budget_report_{sp}.json"))
        print("budget report written", sp)


if __name__ == "__main__":
    main()
