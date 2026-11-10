#!/usr/bin/env python
"""Cache z-scored PTB-XL 100 Hz waveforms per split for CPU training.

Same chain as src/data/dataset.py (wfdb physical mV, header lead order,
per-lead z-score with outputs/norm_stats.npz fit on folds 1-8), same record
order as outputs/phase2a/embeddings/{split}.pt (asserted by ecg_id).
The test split (fold 10) is refused unless --allow-test is given; that flag is
only used by the post-lock formal evaluation (revision_audit/lock/).

Usage: python revision_audit/backbone/cache_waveforms.py --splits train,val
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data.dataset import NormalizationStats, load_ptbxl_metadata, load_waveform  # noqa: E402

DATA = os.path.join(ROOT, "data/ptbxl")
OUT = "cache/ptbxl100"
LABELS = ["label_NORM", "label_MI", "label_STTC", "label_CD", "label_HYP"]


def _load(fn):
    x = load_waveform(DATA, fn)
    assert x.shape == (12, 1000), (fn, x.shape)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    splits = a.splits.split(",")
    if "test" in splits and not a.allow_test:
        sys.exit("refusing to cache the test split without --allow-test (post-lock only)")
    if "test" in splits and not os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json")):
        sys.exit("refusing: revision_audit/lock/LOCK_MANIFEST.json does not exist yet")
    df = load_ptbxl_metadata(DATA)
    ns = NormalizationStats.load(os.path.join(ROOT, "outputs/norm_stats.npz"))
    os.makedirs(OUT, exist_ok=True)
    for split in splits:
        d = df[df["split"] == split].reset_index()
        with Pool(32) as p:
            x = np.stack(p.map(_load, list(d["filename_lr"]), chunksize=64))
        x = ((x - ns.mean[None, :, None]) / ns.std[None, :, None]).astype(np.float32)
        emb = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{split}.pt"))
        assert np.array_equal(emb["ecg_id"].numpy(), d["ecg_id"].values), "order differs from embedding cache"
        assert np.array_equal(emb["y"].numpy(), d[LABELS].values.astype(np.float32))
        np.savez(os.path.join(OUT, f"{split}.npz"), x=x, y=d[LABELS].values.astype(np.float32),
                 ecg_id=d["ecg_id"].values, patient_id=d["patient_id"].values.astype(np.int64),
                 strat_fold=d["strat_fold"].values, age=d["age"].values, sex=d["sex"].values)
        print(split, x.shape, flush=True)


if __name__ == "__main__":
    main()
