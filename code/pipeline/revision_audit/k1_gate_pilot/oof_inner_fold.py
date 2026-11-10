#!/usr/bin/env python
"""Stage 1: out-of-fold A/B predictions + selector features on PTB-XL folds 1-8.

For held-out fold k, runs the ORIGINAL scripts/train_phase2a.py::main
(--method proposed, configs identical except device=cpu) with:
  train data  = folds 1-8 minus {k, es_fold}
  "val" data  = es_fold = k % 8 + 1   (epoch selection / early stopping)
  test        = never loaded; the end-of-run evaluation hook instead writes
                A/B logits and selector features for held-out fold k.
Folds are patient-disjoint, so held-out patients never enter fitting or
epoch selection. The frozen encoder (cached embeddings) did see folds 1-8.

Usage:
    python revision_audit/k1_gate_pilot/oof_inner_fold.py --heldout 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import scripts.train_phase2a as tp
from src.data.clinical_lead_subsets import LEAD_NAMES

OUT = "revision_audit/k1_gate_pilot"
V5 = LEAD_NAMES.index("V5")


@torch.no_grad()
def k1_outputs(model, z: torch.Tensor) -> dict:
    """A/B logits and selector features for one set of records (shared with stage 2)."""
    model.eval()
    lA, mA = tp.run_inference_at_k(model, z, "cpu", 1)
    v5_mask = torch.zeros(z.shape[0], 12)
    v5_mask[:, V5] = 1.0
    lB = model.classifier(model.fusion(z, v5_mask))
    scores = model.selector.raw_scores(z)
    top2 = scores.topk(2, dim=-1).values
    return {"A_logits": lA.numpy(), "B_logits": lB.numpy(), "cand_lead": mA.argmax(1).numpy(),
            "selector_scores": scores.numpy(), "margin_top1_top2": (top2[:, 0] - top2[:, 1]).numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", type=int, required=True, choices=range(1, 9))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    k, es = args.heldout, args.heldout % 8 + 1
    torch.set_num_threads(4)

    full = torch.load(os.path.join(ROOT, "outputs/phase2a/embeddings/train.pt"))
    db = pd.read_csv(os.path.join(ROOT, "data/ptbxl/ptbxl_database.csv"), index_col="ecg_id")
    fold = db.loc[full["ecg_id"].numpy(), "strat_fold"].to_numpy()
    subset = lambda m: {key: v[torch.from_numpy(m)] for key, v in full.items()}
    parts = {"train": subset(~np.isin(fold, [k, es])), "val": subset(fold == es), "heldout": subset(fold == k)}
    pid = {n: set(p["patient_id"].tolist()) for n, p in parts.items()}
    assert not (pid["heldout"] & (pid["train"] | pid["val"])), "held-out patients leak into fitting/epoch selection"

    tp.load_split = lambda _d, split, _n, _s: {} if split == "test" else parts[split]

    def dump_heldout(model, method, kk, seed, _t, device, threshold, d, run_dir):
        if kk == 1:
            h = parts["heldout"]
            out = k1_outputs(model, h["z"])
            name = f"oof_heldout_fold{k}.npz" if args.seed == 0 else f"oof_seed{args.seed}_heldout_fold{k}.npz"
            np.savez(os.path.join(OUT, "results", "oof", name),
                     ecg_id=h["ecg_id"].numpy(), patient_id=h["patient_id"].numpy(), y_true=h["y"].numpy(),
                     z=h["z"].numpy(), heldout_fold=k, es_fold=es, **out)
        return {"macro_auroc": float("nan"), "macro_f1": float("nan")}

    tp.evaluate_and_report = dump_heldout
    os.makedirs(os.path.join(OUT, "results", "oof"), exist_ok=True)
    sys.argv = ["train_phase2a.py", "--method", "proposed", "--seed", str(args.seed),
                "--config", os.path.join(OUT, "configs/phase2a_cpu.yaml"),
                "--run-name", f"inner_proposed_heldout{k}_seed{args.seed}",
                "--outputs-dir", os.path.join(OUT, "inner_runs")]
    tp.main()


if __name__ == "__main__":
    main()
