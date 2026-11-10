#!/usr/bin/env python
"""Descriptive K=1 lead-selection frequency of the proposed selector, by diagnostic class (all five),
with multi-label overlap kept (a record positive for CD and STTC counts in both rows), plus
'no diagnostic label' and 'all records' rows; record and patient counts per row. Seeds 0-2 averaged.
Selection frequency is descriptive; it does not show that a lead causes a miss.

Usage: python revision_audit/clinical/selection_heatmap.py --split val   (test only after the lock)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from scripts.train_phase2a import build_selector  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    if a.split == "test":
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json"))
    d = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{a.split}.pt"))
    y, pid = d["y"].numpy(), d["patient_id"].numpy()
    leads = []
    for s in (0, 1, 2):
        m = Phase2ASelectionModel(build_selector("proposed", 32), latent_dim=32, num_classes=5)
        m.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2a_proposed_seed{s}/checkpoints/best.pt"),
                                     map_location="cpu")["model_state_dict"])
        with torch.no_grad():
            leads.append(m.selector.raw_scores(d["z"]).argmax(1).numpy())
    rows = [(c, y[:, i] == 1) for i, c in enumerate(CLASSES)] + [("no diagnostic label", y.sum(1) == 0),
                                                                  ("all records", np.ones(len(y), bool))]
    F = np.zeros((len(rows), 12))
    meta = []
    for r, (name, msk) in enumerate(rows):
        F[r] = np.mean([[np.mean(l[msk] == j) for j in range(12)] for l in leads], 0)
        meta.append({"row": name, "n_records": int(msk.sum()), "n_patients": int(len(np.unique(pid[msk]))),
                     "freq": dict(zip(LEAD_NAMES, map(float, F[r])))})
    json.dump({"split": a.split, "seeds": [0, 1, 2], "rows": meta,
               "per_seed_frequency_all_records": [dict(zip(LEAD_NAMES, map(float, np.bincount(l, minlength=12) / len(l))))
                                                  for l in leads]},
              open(os.path.join(HERE, "results", f"selection_heatmap_{a.split}.json"), "w"), indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    im = ax.imshow(F, cmap="Blues", vmin=0, vmax=max(0.5, F.max()), aspect="auto")
    ax.set_xticks(range(12), LEAD_NAMES)
    ax.set_yticks(range(len(rows)), [f"{m['row']} (n={m['n_records']}, pts={m['n_patients']})" for m in meta], fontsize=8)
    for i in range(F.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{F[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if F[i, j] > 0.3 else "#222222")
    ax.set_title(f"K=1 proposed selector: lead choice by class ({a.split}; mean of seeds 0-2; labels overlap)",
                 fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, label="share of records")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "results", f"selection_heatmap_{a.split}.{ext}"), dpi=200)
    for m in meta:
        print(m["row"], m["n_records"], m["n_patients"], {k: round(v, 2) for k, v in m["freq"].items() if v > 0.08})


if __name__ == "__main__":
    main()
