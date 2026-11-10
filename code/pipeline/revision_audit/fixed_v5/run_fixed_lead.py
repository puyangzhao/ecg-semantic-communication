#!/usr/bin/env python
"""Matched fixed single-lead baseline: rerun the ORIGINAL Phase 2A
fixed_clinical_k training entry (scripts/train_phase2a.py::main, unmodified)
with exactly two runtime overrides:

1. FIXED_CLINICAL_SUBSETS[1] = [--k1-lead]   (K=3 / K=6 subsets untouched)
2. The end-of-run evaluation is redirected from the test fold to the
   validation fold: load_split(..., "test") returns an empty stub (test.pt is
   never read) and evaluate_and_report() is replaced by a validation-only
   evaluator that saves per-record predictions.

Everything else -- config (configs/phase2a.yaml), K sampling from {1,3,6}
per batch, linear head, class-weighted BCE, Adam, early stopping on the
mean-over-K validation macro-AUROC, 60-epoch budget, patience 10, seeding --
is the original code path. Running with --k1-lead II reproduces the original
fixed Lead II baseline and is used as an implementation check.

NOTE: the reused main() prints "Test K=..." at the end; under this wrapper
those numbers are VALIDATION metrics (the label comes from the unmodified
original print statement).

Usage:
    python revision_audit/fixed_v5/run_fixed_lead.py --k1-lead V5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import scripts.train_phase2a as tp
from src.data import clinical_lead_subsets as cls_mod
from src.evaluation.calibration import compute_auprc
from src.evaluation.metrics import compute_metrics

OUT_ROOT = "revision_audit/fixed_v5/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1-lead", required=True, choices=cls_mod.LEAD_NAMES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase2a.yaml")
    args = ap.parse_args()

    original_subsets = {k: list(v) for k, v in cls_mod.FIXED_CLINICAL_SUBSETS.items()}
    cls_mod.FIXED_CLINICAL_SUBSETS[1] = [args.k1_lead]
    run_name = f"fixed_k1{args.k1_lead}_seed{args.seed}"

    val_data = torch.load(os.path.join("outputs/phase2a/embeddings", "val.pt"))
    original_load_split = tp.load_split

    def load_split_no_test(embeddings_dir, split, subset_n, seed):
        if split == "test":
            return {}  # never read test.pt
        return original_load_split(embeddings_dir, split, subset_n, seed)

    def evaluate_on_val(model, method, k, seed, _test_data, device, threshold, d, run_dir):
        logits, mask = tp.run_inference_at_k(model, val_data["z"], device, k)
        y = val_data["y"].numpy()
        prob = torch.sigmoid(logits).numpy()
        m = compute_metrics(y, prob, threshold=threshold)
        a = compute_auprc(y, prob)
        res = {"split": "val", "k": k, "macro_auroc": m["macro_auroc"], "macro_auprc": a["macro_auprc"],
               "per_class_auroc": m["per_class_auroc"], "per_class_auprc": a["per_class_auprc"],
               "macro_f1": m["macro_f1"],
               "selected_leads_at_k": sorted({cls_mod.LEAD_NAMES[i] for i in np.where(mask.numpy()[0] > 0)[0]})}
        with open(os.path.join(run_dir, f"val_metrics_K{k}.json"), "w") as f:
            json.dump(res, f, indent=2)
        np.savez(os.path.join(run_dir, f"val_raw_K{k}.npz"), logits=logits.numpy(), y_prob=prob, y_true=y,
                 mask=mask.numpy(), ecg_id=val_data["ecg_id"].numpy(),
                 patient_id=val_data["patient_id"].numpy())
        return m

    tp.load_split = load_split_no_test
    tp.evaluate_and_report = evaluate_on_val

    sys.argv = ["train_phase2a.py", "--method", "fixed_clinical_k", "--seed", str(args.seed),
                "--config", args.config, "--run-name", run_name, "--outputs-dir", OUT_ROOT]
    tp.main()

    with open(os.path.join(OUT_ROOT, "runs", run_name, "override_record.json"), "w") as f:
        json.dump({"base_entry": "scripts/train_phase2a.py::main --method fixed_clinical_k",
                   "fixed_subsets_original": original_subsets,
                   "fixed_subsets_used": cls_mod.FIXED_CLINICAL_SUBSETS,
                   "final_eval_split": "val (test.pt never loaded)",
                   "command": " ".join(["python", "revision_audit/fixed_v5/run_fixed_lead.py",
                                        "--k1-lead", args.k1_lead, "--seed", str(args.seed)])}, f, indent=2)


if __name__ == "__main__":
    main()
