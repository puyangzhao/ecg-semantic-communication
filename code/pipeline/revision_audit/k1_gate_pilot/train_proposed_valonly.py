#!/usr/bin/env python
"""Train a Phase 2A proposed (selector + head) checkpoint for an extra seed with
the ORIGINAL scripts/train_phase2a.py::main and configs/phase2a.yaml, unchanged,
except that the end-of-run evaluation is redirected from the test fold to fold 9:
load_split(..., "test") returns an empty stub (fold 10 / test.pt never read).

Implementation check: --seed 0 --check-against outputs/runs/phase2a_proposed_seed0
must reproduce that checkpoint bit for bit.

Usage:
    python revision_audit/k1_gate_pilot/train_proposed_valonly.py --seed 3
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
from src.evaluation.calibration import compute_auprc
from src.evaluation.metrics import compute_metrics

OUT_ROOT = "revision_audit/k1_gate_pilot/phase2a_extra_seeds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/phase2a.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    run_name = args.run_name or f"phase2a_proposed_seed{args.seed}"

    val = torch.load(os.path.join(ROOT, "outputs/phase2a/embeddings/val.pt"))
    original_load_split = tp.load_split
    tp.load_split = lambda d, split, n, s: {} if split == "test" else original_load_split(d, split, n, s)

    def evaluate_on_val(model, method, k, seed, _t, device, threshold, d, run_dir):
        logits, mask = tp.run_inference_at_k(model, val["z"], device, k)
        y, p = val["y"].numpy(), torch.sigmoid(logits).numpy()
        m, a = compute_metrics(y, p, threshold=threshold), compute_auprc(y, p)
        json.dump({"split": "val", "k": k, "macro_auroc": m["macro_auroc"], "macro_auprc": a["macro_auprc"]},
                  open(os.path.join(run_dir, f"val_metrics_K{k}.json"), "w"), indent=2)
        return m

    tp.evaluate_and_report = evaluate_on_val
    sys.argv = ["train_phase2a.py", "--method", "proposed", "--seed", str(args.seed), "--config", args.config,
                "--run-name", run_name, "--outputs-dir", OUT_ROOT]
    tp.main()
    json.dump({"base_entry": "scripts/train_phase2a.py::main --method proposed", "config": args.config,
               "final_eval_split": "val (fold 10 / test.pt never loaded)"},
              open(os.path.join(OUT_ROOT, "runs", run_name, "override_record.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
