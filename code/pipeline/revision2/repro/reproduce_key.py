#!/usr/bin/env python
"""PLAN §1: recompute existing key numbers from cached predictions only (no model is run or trained).

  * selector - matched V5 (and - Lead II) on fold 10 from formal_eval/results/k1_logits_test.npz, every metric,
    vs formal_eval/results/k1_eval_test.json (same draws: patient_draws(pid, 2000, seed=20261110));
  * clean 12-lead reference (fold 10) from outputs/runs/phase2a_ceiling_k12_seed*/test_raw_K12.npz
    vs final_submission/code/data/phase2a_clinical_metrics.json;
  * joint JSCC, AWGN 20 dB (fold 10) from the comm_eval cache vs comm_report_test.json.
The transmission report itself is re-derived by channels/report.py --check --split test.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2common as C  # noqa: E402
import comm_report as CR  # noqa: E402


def main():
    out = {"tolerance": 1e-9}
    d = np.load(os.path.join(C.RA, "formal_eval/results/k1_logits_test.npz"))
    y, pid = d["y"], d["patient_id"]
    P = {(m, s): C.sig(d[f"{m}_seed{s}"]) for m in ("orig_A", "orig_C", "orig_II") for s in C.SEEDS}
    obs = {k: C.metric_vec(y, v) for k, v in P.items()}
    boot = C.bootstrap(y, P, C.patient_draws(pid, 2000))
    ref = json.load(open(os.path.join(C.RA, "formal_eval/results/k1_eval_test.json")))
    dev = 0.0
    for x, z in (("orig_A", "orig_C"), ("orig_A", "orig_II")):
        new = C.paired(obs, boot, x, z)
        for n, v in new.items():
            r = ref["paired"][f"{x}-{z}"][n]
            dev = max(dev, abs(v["mean"] - r["mean"]), *[abs(a - b) for a, b in zip(v["ci95"], r["ci95"])], abs(v["p_boot"] - r["p_boot"]))
        out[f"{x}-{z}"] = {"macro_auroc": new["macro_auroc"], "macro_auprc": new["macro_auprc"]}
    out["k1_max_abs_dev"] = dev
    # clean 12-lead reference
    cm = json.load(open(os.path.join(C.ROOT, "final_submission/code/data/phase2a_clinical_metrics.json")))["per_condition"]["ceiling_k12"]
    cd = []
    for s in C.SEEDS:
        sv = np.load(os.path.join(C.ROOT, f"outputs/runs/phase2a_ceiling_k12_seed{s}/test_raw_K12.npz"))
        assert np.array_equal(sv["ecg_id"], d["ecg_id"])
        mv = C.metric_vec(sv["y_true"] if "y_true" in sv else y, sv["y_prob"].astype(np.float64))
        cd.append(max(abs(mv[10] - cm[str(s)]["12"]["macro_auroc_consistency_check"]), abs(mv[11] - cm[str(s)]["12"]["macro_auprc"])))
        out[f"ceiling_seed{s}"] = {"macro_auroc": float(mv[10]), "macro_auprc": float(mv[11])}
    out["ceiling_mean_macro_auroc"] = float(np.mean([out[f"ceiling_seed{s}"]["macro_auroc"] for s in C.SEEDS]))
    out["ceiling_max_abs_dev"] = float(max(cd))
    # joint JSCC AWGN 20 dB
    rep = json.load(open(os.path.join(C.RA, "comm/results/comm_report_test.json")))
    meta = np.load(os.path.join(C.OLD_CACHE, "test", "seed0", "meta.npz"))
    v = [CR.macro_auroc_batch(meta["y"], expit(CR.load("test", s, "awgn_perfect", "jscc_e2e", 20).astype(np.float64))).mean()
         for s in C.SEEDS]
    out["jscc_e2e_awgn20_per_seed"] = list(map(float, v))
    out["jscc_e2e_awgn20_mean"] = float(np.mean(v))
    r20 = rep["conditions"]["awgn_perfect"]["absolute"]["jscc_e2e"]["per_snr_auroc"]["20"]
    out["jscc_e2e_awgn20_abs_dev"] = abs(out["jscc_e2e_awgn20_mean"] - r20)
    out["pass"] = dev <= 1e-9 and out["ceiling_max_abs_dev"] <= 1e-6 and out["jscc_e2e_awgn20_abs_dev"] <= 1e-9
    C.dump(out, os.path.join(C.R2, "results", "repro_key_numbers.json"))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    print("primary", out["orig_A-orig_C"]["macro_auroc"]["mean"], out["orig_A-orig_C"]["macro_auroc"]["ci95"])


if __name__ == "__main__":
    main()
