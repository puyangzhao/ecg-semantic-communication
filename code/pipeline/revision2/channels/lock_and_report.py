#!/usr/bin/env python
"""PLAN §1/§3/§3b: policy locking (fold 9) and paired reports for the Rician and time-variation analyses.

  --check-lock          lock() on the existing fold-9 JSONs must reproduce comm/locked_comm_policies.json
  --check-report SPLIT  report() on the existing cache must reproduce comm/results/comm_report_SPLIT.json
                        (val: B=200 as in the locked fold-9 report; test: B=1000)
  --lock                lock the four Rician conditions from fold 9
  --report SPLIT        Rician (4 conditions) and time-variation reports
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2common as C  # noqa: E402
import r2comm as RC  # noqa: E402
import comm_report as CR  # noqa: E402
import lock_policies as LP  # noqa: E402

RICIAN = ["rician3dB_perfect", "rician3dB_ls2", "rician10dB_perfect", "rician10dB_ls2"]
TV = ["rician3dB_ls2_rho1.0", "rician3dB_ls2_rho0.95", "rician3dB_ls2_rho0.0"]
RES = os.path.join(C.R2, "results", "channels")
EXTRA = [("jscc_ctrl", "rep_orig_policy"), ("sscc_lookup", "rep_orig_policy")]


def meta(cache, split):
    m = np.load(os.path.join(cache, split, "seed0", "meta.npz"))
    for s in C.SEEDS:
        ms = np.load(os.path.join(cache, split, f"seed{s}", "meta.npz"))
        assert np.array_equal(ms["y"], m["y"]) and np.array_equal(ms["patient_id"], m["patient_id"])
    return m["y"], m["patient_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-lock", action="store_true")
    ap.add_argument("--check-report")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    fam = lambda cond: LP.FAMILIES
    if a.check_lock:
        S = [json.load(open(os.path.join(C.RA, "comm/results", f"comm_eval_val_seed{s}.json"))) for s in C.SEEDS]
        ok = RC.lock(S, LP.CONDS, fam, LP.SUBMITTED) == json.load(open(os.path.join(C.RA, "comm/locked_comm_policies.json")))
        C.dump({"lock_reproduces_locked_comm_policies": ok}, os.path.join(C.R2, "results", "checks", "lock_rule.json"))
        print("lock rule reproduces locked_comm_policies.json:", ok)
        assert ok
    if a.check_report:
        sp = a.check_report
        pol = json.load(open(os.path.join(C.RA, "comm/locked_comm_policies.json")))
        y, pid = meta(C.OLD_CACHE, sp)
        load = lambda c, k, q, s: CR.load(sp, s, c, k, q)
        con = [(f"{c}:{x}-{z}", (c, x), (c, z)) for c in CR.CONDS for x, z in CR.CONTRASTS]
        rep = RC.report(y, pid, CR.CONDS, lambda c: CR.policy_map(c, pol), load, con, B=200 if sp == "val" else 1000, workers=a.workers)
        ref = json.load(open(os.path.join(C.RA, "comm/results", f"comm_report_{sp}.json")))
        dev, out = 0.0, {}
        for lab, per in rep["contrasts"].items():
            c, k = lab.split(":")
            for q, e in per.items():
                r = ref["conditions"][c]["contrasts"][k][q]
                dev = max(dev, abs(e["auroc"]["mean"] - r["mean"]), *[abs(u - w) for u, w in zip(e["auroc"]["ci95"], r["ci95"])],
                          abs(e["auroc"]["p_boot"] - r["p_boot"]))
            out[lab] = per["policy"]
        for c in CR.CONDS:
            for name, ab in rep["absolute"][c].items():
                rr = ref["conditions"][c]["absolute"][name]
                dev = max(dev, abs(ab["policy_auroc"] - rr["policy_auroc"]), abs(ab["policy_auprc"] - rr["policy_auprc"]))
        C.dump({"split": sp, "max_abs_dev_vs_comm_report": dev, "pass": dev <= 1e-9, "policy_contrasts": out},
               os.path.join(C.R2, "results", "checks", f"report_reproduction_{sp}.json"))
        print(f"report vs comm_report_{sp}.json: max |dev| =", dev)
        assert dev <= 1e-9
    if a.lock:
        S = [json.load(open(os.path.join(RES, f"channels_eval_val_seed{s}.json"))) for s in C.SEEDS]
        pol = RC.lock(S, RICIAN, fam, {}, "\n[revision2] applied unchanged to the Rician conditions on fold 9; rep_orig uses the rule-selected tau.")
        C.dump(pol, os.path.join(RES, "locked_policies_rician.json"))
        for c, v in pol["conditions"].items():
            print(c, {f: (v[f]["threshold_tau_selected"] if "threshold_tau_selected" in v[f] else v[f]["best_fixed"])
                      for f in ("rep_orig", "rep_ctrl", "sscc")}, "sscc lookup", v["sscc"]["lookup"])
    if a.report:
        sp = a.report
        pol = json.load(open(os.path.join(RES, "locked_policies_rician.json")))
        cache = os.path.join(C.CACHE, "channels")
        y, pid = meta(cache, sp)
        load = lambda c, k, q, s: np.load(os.path.join(cache, sp, f"seed{s}", c, f"{k}__{q}.npy"))
        con = [(f"{c}:{x}-{z}", (c, x), (c, z)) for c in RICIAN for x, z in CR.CONTRASTS + EXTRA]
        rep = RC.report(y, pid, RICIAN, lambda c: CR.policy_map(c, pol), load, con, workers=a.workers)
        rep.update(split=sp, post_hoc_revision_analysis=True, locked_policies="results/channels/locked_policies_rician.json")
        C.dump(rep, os.path.join(RES, f"rician_report_{sp}.json"))
        # time variation: K_R = 3 dB, LS-2; rep_ctrl policy = the one locked for rician3dB_ls2
        c3 = pol["conditions"]["rician3dB_ls2"]["rep_ctrl"]
        tau = c3["threshold_tau_selected"]

        def tvmap(cond):
            pick = {s: f"rep_ctrl_r{4 if (tau is not None and s <= float(tau)) else 2}" for s in RC.POLICY_SNRS}
            return {"rep_ctrl_policy": pick, "rep_ctrl_bestfixed": {s: c3["best_fixed"] for s in RC.POLICY_SNRS},
                    "jscc_e2e": {s: "jscc_e2e" for s in RC.POLICY_SNRS},
                    **{f"rep_ctrl_r{r}": {s: f"rep_ctrl_r{r}" for s in RC.POLICY_SNRS} for r in (1, 2, 4)}}
        con = [(f"{c}:jscc_e2e-rep_ctrl_policy", (c, "jscc_e2e"), (c, "rep_ctrl_policy")) for c in TV]
        con += [(f"{n}:{c}-rho1.0", (c, n), (TV[0], n)) for n in ("jscc_e2e", "rep_ctrl_policy") for c in TV[1:]]
        rtv = RC.report(y, pid, TV, tvmap, load, con, workers=a.workers)
        rtv.update(split=sp, post_hoc_revision_analysis=True, K_R_dB=3, csi="ls2", rep_ctrl_tau=tau)
        C.dump(rtv, os.path.join(RES, f"timevar_report_{sp}.json"))
        print("reports written", sp)


if __name__ == "__main__":
    main()
