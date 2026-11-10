#!/usr/bin/env python
"""Cache frozen revision-backbone per-lead embeddings (each lead encoded on its own, with its lead ID)
and the backbone's own receiver outputs. Same fields as outputs/phase2a/embeddings/{split}.pt.
Test split refused before the lock.

Usage: python revision_audit/backbone/embed.py --seed 0 --splits train,val
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from lead_resnet import LeadAwareModel  # noqa: E402
from train_backbone import NEW_KW  # noqa: E402

CACHE = "cache/ptbxl100"


@torch.no_grad()
def embed(model, x, bs=128):
    model.eval()
    out = []
    ids = torch.arange(12)
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs])
        b = xb.shape[0]
        z = model.encoder.encode_pairs(xb.reshape(b * 12, -1), ids.repeat(b)).reshape(b, 12, -1)
        out.append(z)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    splits = a.splits.split(",")
    if "test" in splits:
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json"))
    m = LeadAwareModel(**NEW_KW)
    m.load_state_dict(torch.load(os.path.join(HERE, f"runs/new_seed{a.seed}/best.pt"))["model_state_dict"])
    out = os.path.join(HERE, "emb", f"new_seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    for sp in splits:
        d = np.load(os.path.join(CACHE, f"{sp}.npz"))
        z = embed(m, d["x"])
        ref = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{sp}.pt"))
        assert np.array_equal(ref["ecg_id"].numpy(), d["ecg_id"])
        torch.save({"z": z, "y": torch.from_numpy(d["y"]), "ecg_id": ref["ecg_id"], "patient_id": ref["patient_id"],
                    "sex": ref["sex"], "age": ref["age"]}, os.path.join(out, f"{sp}.pt"))
        # sanity: the receiver on all-12 embeddings must equal the end-to-end forward
        with torch.no_grad():
            x0 = torch.from_numpy(d["x"][:64])
            e2e = m(x0, torch.ones(64, 12))
            via = m.receiver(z[:64], torch.ones(64, 12))
        print(sp, tuple(z.shape), "receiver-vs-e2e max abs", float((e2e - via).abs().max()), flush=True)


if __name__ == "__main__":
    main()
