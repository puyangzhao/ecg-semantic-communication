#!/usr/bin/env python
"""Precomputes frozen Phase 1 per-lead embeddings for every PTB-XL record,
once, so the 15 Phase 2A method x seed training runs operate on cached
[N, 12, 32] tensors instead of re-reading WFDB waveforms every batch
(docs/phase2a_design.md section 7 and implementation plan step 5).

Usage:
    python scripts/precompute_embeddings.py \
        --baseline-checkpoint outputs/runs/baseline_run1/checkpoints/best.pt \
        --baseline-config configs/baseline.yaml \
        --out-dir outputs/phase2a/embeddings
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NormalizationStats, PTBXLDataset, load_ptbxl_metadata
from src.models.full_model import ECGBaselineModel


@torch.no_grad()
def compute_split_embeddings(encoder, df_split, root, norm_stats, filename_col, device, batch_size=128):
    ds = PTBXLDataset(df_split, root, norm_stats, filename_col=filename_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    encoder.eval()

    all_z, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x)  # [B, 12, 32]
        all_z.append(z.cpu())
        all_y.append(y)
    z = torch.cat(all_z, dim=0)
    y = torch.cat(all_y, dim=0)

    meta = df_split.reset_index()  # keep ecg_id as a column
    return {
        "z": z,
        "y": y,
        "ecg_id": torch.tensor(meta["ecg_id"].values.astype(np.int64)),
        "patient_id": torch.tensor(meta["patient_id"].values.astype(np.int64)),
        "sex": torch.tensor(meta["sex"].values.astype(np.int64)),
        "age": torch.tensor(meta["age"].values.astype(np.float32)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", default="outputs/runs/baseline_run1/checkpoints/best.pt")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--out-dir", default="outputs/phase2a/embeddings")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.baseline_config) as f:
        config = yaml.safe_load(f)

    root = config["data"]["root"]
    filename_col = config["data"]["filename_col"]
    df = load_ptbxl_metadata(root)

    norm_stats = NormalizationStats.load(config["data"]["norm_stats_path"])

    model = ECGBaselineModel(
        num_leads=config["data"]["num_leads"],
        num_classes=5,
        latent_dim=config["model"]["latent_dim"],
        fusion=config["model"]["fusion"],
        base_channels=config["model"]["base_channels"],
        num_blocks=config["model"]["num_blocks"],
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        dropout=config["model"]["dropout"],
    )
    ckpt = torch.load(args.baseline_checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        df_split = df[df["split"] == split]
        print(f"Encoding {split}: {len(df_split)} records...")
        data = compute_split_embeddings(
            model.encoder, df_split, root, norm_stats, filename_col, args.device,
        )
        out_path = os.path.join(args.out_dir, f"{split}.pt")
        torch.save(data, out_path)
        print(f"  wrote {out_path}: z {tuple(data['z'].shape)}")


if __name__ == "__main__":
    main()
