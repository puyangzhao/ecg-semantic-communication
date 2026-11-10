#!/usr/bin/env python
"""Code check only: build a small patient-disjoint split from PTB-XL folds 1-8 (no fold 9 or 10).

Reads outputs/phase2a/embeddings/train.pt (folds 1-8 only) and data/ptbxl/ptbxl_database.csv, and writes
<out>/emb/{train,val,test}.pt:
  train: 600 patients from folds 1-6
  val:   150 patients from fold 7   (early stopping in the check run)
  test:  150 patients from fold 8   (end-of-run evaluation in the check run)
PTB-XL folds are patient-disjoint; this is asserted. The result is not used by any reported analysis.

    python checks/make_check_split.py --out /tmp/ecg_code_check
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--embeddings", default="outputs/phase2a/embeddings/train.pt")
    ap.add_argument("--ptbxl-csv", default="data/ptbxl/ptbxl_database.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = torch.load(a.embeddings)
    fold = pd.read_csv(a.ptbxl_csv, index_col="ecg_id").loc[d["ecg_id"].numpy(), "strat_fold"].to_numpy()
    assert set(np.unique(fold)) <= set(range(1, 9)), "input must contain folds 1-8 only"
    pid = d["patient_id"].numpy()
    rng = np.random.default_rng(a.seed)
    parts, info = {}, {}
    for name, folds, n_pat in (("train", range(1, 7), 600), ("val", [7], 150), ("test", [8], 150)):
        pats = np.unique(pid[np.isin(fold, list(folds))])
        keep = rng.choice(pats, size=min(n_pat, len(pats)), replace=False)
        idx = np.flatnonzero(np.isin(pid, keep) & np.isin(fold, list(folds)))
        parts[name] = idx
        info[name] = {"folds": sorted(int(f) for f in np.unique(fold[idx])), "records": int(len(idx)),
                      "patients": int(len(np.unique(pid[idx])))}
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(pid[parts[x]]) & set(pid[parts[y]]), f"patient overlap {x}/{y}"
    os.makedirs(os.path.join(a.out, "emb"), exist_ok=True)
    for name, idx in parts.items():
        t = torch.from_numpy(idx)
        torch.save({k: v[t] for k, v in d.items()}, os.path.join(a.out, "emb", f"{name}.pt"))
    info["patient_overlap"] = "none (asserted)"
    json.dump(info, open(os.path.join(a.out, "check_split.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
