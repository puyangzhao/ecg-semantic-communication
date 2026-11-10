#!/usr/bin/env python
"""Lock SNR-adaptive policies on fold 9 (validation) only, for every channel/CSI condition.

Pre-specified rules:
  SNR distribution  : uniform over POLICY_SNRS = {-15,-10,-5,0,5,10,20} dB (40 dB / noiseless reported only)
  target metric     : seed-mean (seeds 0-2) realization-mean macro-AUROC on fold 9
  families          : rep_orig (submitted Phase 2B checkpoints), rep_ctrl (controlled lead), sscc (5 configs)
  best fixed option : the single option maximizing the target averaged over POLICY_SNRS
  threshold policy  : (rep families) r=4 if SNR <= tau else r=2, tau in {-inf,-15,-10,-5,0,5,10,20}
  lookup policy     : per SNR, the option maximizing the target at that SNR (rep: r; sscc: configuration)
  ties              : higher seed-mean macro-AUPRC, then smaller tau / earlier option in the listed order
The submitted policies stay as they were frozen in 2026-08 (AWGN tau=-10, Rayleigh perfect-CSI tau=0)
and are evaluated unchanged; the threshold rule above is re-applied only where no policy existed
(Rayleigh with LS-2 pilots) and, for comparison, to every condition.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_SNRS = [-15, -10, -5, 0, 5, 10, 20]
CONDS = ["awgn_perfect", "rayleigh_perfect", "rayleigh_ls2"]
FAMILIES = {"rep_orig": ["rep_orig_r1", "rep_orig_r2", "rep_orig_r4"],
            "rep_ctrl": ["rep_ctrl_r1", "rep_ctrl_r2", "rep_ctrl_r4"],
            "sscc": ["sscc_S6_RM1_QPSK", "sscc_S16_RM2_QPSK", "sscc_S32_RM2_16QAM", "sscc_S32_UNC_QPSK",
                     "sscc_S64_UNC_16QAM"]}
TAUS = [-np.inf, -15, -10, -5, 0, 5, 10, 20]
SUBMITTED = {"awgn_perfect": -10, "rayleigh_perfect": 0}


def main():
    S = [json.load(open(os.path.join(HERE, "results", f"comm_eval_val_seed{s}.json"))) for s in (0, 1, 2)]
    m = lambda cond, snr, opt, key="macro_auroc": float(np.mean([x["conditions"][f"{cond}_{snr}"][opt][key] for x in S]))
    out = {"rules": __doc__, "policy_snrs": POLICY_SNRS, "conditions": {}}
    for cond in CONDS:
        oc = {}
        for fam, opts in FAMILIES.items():
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
                if fam == "rep_orig" and cond in SUBMITTED:
                    fam_res["submitted_tau"] = SUBMITTED[cond]
                    fam_res["submitted_target_on_this_val_harness"] = tau_score[SUBMITTED[cond]][0]
            oc[fam] = fam_res
        oc["jscc_ctrl_target"] = float(np.mean([m(cond, s, "jscc_ctrl") for s in POLICY_SNRS]))
        oc["jscc_e2e_target"] = float(np.mean([m(cond, s, "jscc_e2e") for s in POLICY_SNRS]))
        out["conditions"][cond] = oc
    json.dump(out, open(os.path.join(HERE, "locked_comm_policies.json"), "w"), indent=2, default=str)
    for c, v in out["conditions"].items():
        print(c, json.dumps({f: (r if not isinstance(r, dict) else {k: r[k] for k in r if k != "lookup"}) for f, r in v.items()}, default=str))


if __name__ == "__main__":
    main()
