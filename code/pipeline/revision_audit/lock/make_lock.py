#!/usr/bin/env python
"""Write revision_audit/lock/LOCK_MANIFEST.json (+ .md) BEFORE any revision access to PTB-XL fold 10 or
any external-cohort model output. Refuses to overwrite an existing manifest.

Fold 10 was already used once by the original submission; the formal runs that follow are an
"evaluation of the locked revision protocol on the original held-out test fold", not a first look.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RA = os.path.join(ROOT, "revision_audit")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hashes(patterns):
    out = {}
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(ROOT, pat), recursive=True))
        assert files, f"nothing matches {pat}"
        for f in files:
            out[os.path.relpath(f, ROOT)] = sha(f)
    return out


def main():
    path = os.path.join(HERE, "LOCK_MANIFEST.json")
    if os.path.exists(path):
        sys.exit("LOCK_MANIFEST.json already exists; the lock is not rewritten")
    best_lead = json.load(open(os.path.join(RA, "k1_fair/results/new_backbone_best_fixed_lead.json")))
    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_note": "revision_audit/ is untracked at lock time; every script and artifact below is pinned by sha256",
        "fold10_status": "PTB-XL fold 10 was used once by the original submission. This is an evaluation of the locked revision protocol on the original held-out test fold.",
        "seeds": {"selector_backbone_comm": [0, 1, 2], "gate_locked": [0, 3, 5],
                  "gate_selection_rule": "fold-9 (gate_eq - margin) at 50% rank-matched > 0 for both CD and STTC AUPRC (6 -> 3); see revision_audit/gate_lock/gate_lock.json",
                  "gate_sensitivity_all_six": "seeds 0-5 also evaluated on fold 10, reported as a pre-specified sensitivity analysis of the selection"},
        "frozen_choices": {
            "new_backbone_best_fixed_lead": best_lead,
            "subsets": {b: json.load(open(os.path.join(RA, f"subset_enum/locked_subsets_{b}.json")))["locked"] for b in ("orig", "new")},
            "thresholds": "revision_audit/thresholds/frozen_thresholds.json (Youden J on fold 9, ties -> lowest threshold)",
            "comm_policies": "revision_audit/comm/locked_comm_policies.json (fold 9; submitted rep_orig policies unchanged)",
            "external_label_map": "revision_audit/external_ecgarr/label_map_frozen.json",
            "temperature_scaling": "one scalar T per (model, seed) fitted on fold 9 by BCE; per-class Platt exploratory",
        },
        "primary_and_secondary": {
            "primary": "PTB-XL fold 10, original backbone, K=1: proposed selector + head (orig_A) minus matched fixed-V5 head (orig_C), macro-AUROC; seeds 0-2; shared patient-level paired bootstrap (2000, rng 20261110), per-seed difference then mean",
            "key_secondary_holm": ["orig_A - orig_C macro-AUPRC", "CD AUROC", "CD AUPRC", "STTC AUROC", "STTC AUPRC"],
            "multiplicity": "Holm-Bonferroni over the 5 key secondary endpoints using two-sided bootstrap p-values; every other comparison is exploratory with unadjusted 95% intervals and no significance claim",
            "external_primary": "ecg-arrhythmia v1.0.0: orig_A - orig_C for CD and STTC-partial AUROC/AUPRC, record-level bootstrap (no patient identifier); reported separately from PTB-XL",
        },
        "exploratory": [
            "orig_A - orig_B (same head, lead swapped); orig_B - orig_C; orig_A vs fixed II and global ranking",
            "revision backbone: 12-lead performance vs original architecture; new_A vs new_C / new_BEST / new_II / new_GL; new_A - orig_A (whole-model comparison)",
            "K=3/K=6 enumerated best fixed subset vs clinical subset vs random per-record vs selector choice (same subset-agnostic receiver), both backbones",
            "K=1 label-assisted oracle (mean BCE primary statistic)",
            "frozen development-set operating points: TP/FP/TN/FN, sensitivity, specificity with intervals",
            "temperature scaling and Platt (exploratory) calibration; descriptive subgroups (sex, age bins)",
            "gate (seeds 0,3,5): 50% rank-matched quota vs margin and vs matched random; frozen-threshold execution and actual switch rate; all-six-seed sensitivity",
            "communication: rep_orig (submitted), rep_ctrl, SSCC (5 configs + lookup), JSCC ctrl, JSCC e2e; AWGN, block Rayleigh perfect CSI, block Rayleigh LS-2 pilots; locked policies vs best fixed",
            "external: orig_A - orig_B and orig_B - orig_C. The gate is NOT evaluated externally: its training target (mean BCE gain over all five PTB-XL classes) is not interpretable when only CD and STTC-partial are mapped (decided before any external model output)",
        ],
        "outputs": ["revision_audit/formal_eval/results/k1_eval_test.json", "revision_audit/subset_enum/results/subset_enum_{orig,new}_test.json",
                    "revision_audit/gate_lock/results/test/", "revision_audit/comm/results/comm_eval_test_seed*.json",
                    "revision_audit/comm/results/comm_report_test.json", "revision_audit/external_ecgarr/results/external_eval.json",
                    "revision_audit/backbone/results/backbone_test.json", "revision_audit/clinical/results/selection_heatmap_test.json"],
        "software_error_rule": "if a software error that affects a conclusion is found after this lock, the pre-correction result is kept and reported next to the corrected one with the fixing change; nothing is tuned on fold 10 or external results",
        "artifacts_sha256": {
            "encoder_and_norm": hashes(["outputs/runs/baseline_run1/checkpoints/best.pt", "outputs/norm_stats.npz"]),
            "phase2a": hashes(["outputs/runs/phase2a_*_seed[012]/checkpoints/best.pt"]),
            "fixed_v5": hashes(["revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed[012]/checkpoints/best.pt"]),
            "phase2a_extra_seeds": hashes(["revision_audit/k1_gate_pilot/phase2a_extra_seeds/runs/phase2a_proposed_seed[345]/checkpoints/best.pt"]),
            "gate_lock": hashes(["revision_audit/gate_lock/gate_lock.json", "revision_audit/gate_lock/gate_seed*.npz"]),
            "phase2b": hashes(["outputs/runs/phase2b_pilot_k1_awgn_seed[012]/checkpoints/best.pt"]),
            "comm_models": hashes(["revision_audit/comm/runs/*/best.pt", "revision_audit/comm/runs/sscc_heads_seed*/*.p*",
                                   "revision_audit/comm/locked_comm_policies.json"]),
            "revision_backbone": hashes(["revision_audit/backbone/runs/new_seed[012]/best.pt", "revision_audit/backbone/runs/orig_seed[012]/best.pt"]),
            "k1_fair_heads": hashes(["revision_audit/k1_fair/runs/*/best.pt"]),
            "frozen_choices": hashes(["revision_audit/thresholds/frozen_thresholds.json", "revision_audit/subset_enum/locked_subsets_*.json",
                                      "revision_audit/k1_fair/results/new_backbone_best_fixed_lead.json"]),
            "external": hashes(["revision_audit/external_ecgarr/label_map_frozen.json", "revision_audit/external_ecgarr/build_labels.py",
                                "revision_audit/external_ecgarr/results/external_labels.csv",
                                "revision_audit/external_validation_plan/external_pipeline.py"]),
            "code": hashes(["revision_audit/**/*.py", "src/**/*.py", "scripts/train_phase2a.py", "scripts/train_phase2b_pilot.py"]),
        },
    }
    json.dump(man, open(path, "w"), indent=2)
    md = ["# Revision lock manifest", "", f"Created (UTC): {man['created_utc']}  ", f"git HEAD: {man['git_head']} ({man['git_note']})", "",
          man["fold10_status"], "", "## Seeds", "```", json.dumps(man["seeds"], indent=1), "```", "", "## Frozen choices", "```",
          json.dumps(man["frozen_choices"], indent=1), "```", "", "## Primary / key secondary / multiplicity", "```",
          json.dumps(man["primary_and_secondary"], indent=1), "```", "", "## Exploratory", ""] + [f"- {e}" for e in man["exploratory"]] + [
          "", "## Software-error rule", "", man["software_error_rule"], "",
          f"Artifact hashes: {sum(len(v) for v in man['artifacts_sha256'].values())} files, see LOCK_MANIFEST.json."]
    open(os.path.join(HERE, "LOCK_MANIFEST.md"), "w").write("\n".join(md) + "\n")
    print("lock written:", path, sum(len(v) for v in man["artifacts_sha256"].values()), "files hashed")


if __name__ == "__main__":
    main()
