#!/usr/bin/env python
"""Trains the clean-channel baseline model on PTB-XL 100Hz.

Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml --run-name baseline_run1
    python scripts/train_baseline.py --config configs/smoke.yaml --run-name smoke_test
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NormalizationStats, PTBXLDataset, fit_normalization, load_ptbxl_metadata
from src.evaluation.evaluate import evaluate_checkpoint
from src.models.full_model import ECGBaselineModel
from src.training.losses import build_weighted_bce_loss
from src.training.seed import set_seed
from src.training.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    root = config["data"]["root"]
    df = load_ptbxl_metadata(root)

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    subset_train = config["data"].get("subset_n_train")
    if subset_train:
        df_train = df_train.sample(n=min(subset_train, len(df_train)), random_state=config["seed"])
        df_val = df_val.sample(n=min(config["data"]["subset_n_val"], len(df_val)), random_state=config["seed"])
        df_test = df_test.sample(n=min(config["data"]["subset_n_test"], len(df_test)), random_state=config["seed"])

    norm_path = config["data"]["norm_stats_path"]
    if os.path.exists(norm_path):
        norm_stats = NormalizationStats.load(norm_path)
    else:
        print(f"Fitting normalization on {len(df_train)} training records...")
        norm_stats = fit_normalization(df_train, root, filename_col=config["data"]["filename_col"])
        os.makedirs(os.path.dirname(norm_path), exist_ok=True)
        norm_stats.save(norm_path)

    ds_train = PTBXLDataset(df_train, root, norm_stats, filename_col=config["data"]["filename_col"])
    ds_val = PTBXLDataset(df_val, root, norm_stats, filename_col=config["data"]["filename_col"])
    ds_test = PTBXLDataset(df_test, root, norm_stats, filename_col=config["data"]["filename_col"])

    train_loader = DataLoader(ds_train, batch_size=config["train"]["batch_size"], shuffle=True,
                               num_workers=config["train"]["num_workers"])
    val_loader = DataLoader(ds_val, batch_size=config["train"]["batch_size"], shuffle=False,
                             num_workers=config["train"]["num_workers"])
    test_loader = DataLoader(ds_test, batch_size=config["train"]["batch_size"], shuffle=False,
                              num_workers=config["train"]["num_workers"])

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

    label_cols = ["label_NORM", "label_MI", "label_STTC", "label_CD", "label_HYP"]
    train_labels = df_train[label_cols].values.astype(np.float32)
    loss_fn = build_weighted_bce_loss(train_labels)

    device = config["train"]["device"]
    loss_fn_device = loss_fn
    loss_fn_device.pos_weight = loss_fn_device.pos_weight.to(device)

    run_dir = os.path.join(args.outputs_dir, "runs", args.run_name)
    result = train_model(model, train_loader, val_loader, loss_fn_device, config, run_dir, device=device)
    print("Training result:", result["best_val_macro_auroc"], "at epoch", result["best_epoch"])

    best_ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
    test_metrics = evaluate_checkpoint(
        model, best_ckpt, test_loader, device, threshold=config["eval"]["threshold"],
        output_path=os.path.join(run_dir, "test_metrics.json"),
    )
    print("Test macro-AUROC:", test_metrics["macro_auroc"], "Test macro-F1:", test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
