#!/usr/bin/env python
"""Minimal runnable demo of the task-aware ECG transmission pipeline, using
synthetic random tensors only (no PTB-XL data, no trained checkpoints).

Demonstrates, end to end: input shape -> personalized lead selection ->
fixed-budget channel transmission (AWGN and block-Rayleigh) -> output
probabilities.

Run from the repository root:
    python examples/minimal_demo.py
"""
from __future__ import annotations

import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TaskAwareECGModel


def main() -> None:
    torch.manual_seed(0)

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "configs", "example_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = TaskAwareECGModel(
        num_leads=cfg["model"]["num_leads"],
        signal_length=cfg["model"]["signal_length"],
        base_channels=cfg["model"]["base_channels"],
        latent_dim=cfg["model"]["latent_dim"],
        num_blocks=cfg["model"]["num_blocks"],
        num_classes=cfg["model"]["num_classes"],
        classifier_hidden_dim=cfg["model"]["classifier_hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )
    model.eval()

    # Synthetic input only: random tensor standing in for a batch of
    # 12-lead ECG windows. Shape convention: [batch, leads, time_samples].
    batch_size = 4
    x = torch.randn(batch_size, cfg["model"]["num_leads"], cfg["model"]["signal_length"])
    print(f"Input shape: {tuple(x.shape)}  (batch, leads, time_samples)")

    k = cfg["lead_selection"]["k"]
    m = cfg["channel"]["m"]
    r = cfg["channel"]["r"]
    snr_db = cfg["channel"]["snr_db"]

    with torch.no_grad():
        for channel_type in ("awgn", "rayleigh"):
            probs, lead_mask, diagnostics = model(
                x, k=k, m=m, r=r, snr_db=snr_db, channel_type=channel_type,
            )
            selected_leads = [row.nonzero(as_tuple=True)[0].tolist() for row in lead_mask]

            print(f"\n--- channel_type={channel_type}, K={k}, m={m}, r={r} (m*r={m * r}), "
                  f"SNR={snr_db} dB ---")
            print(f"Selected lead indices per sample (K={k} of 12): {selected_leads}")
            print(f"Channel-use / energy accounting (per selected lead): {diagnostics}")
            print(f"Output probabilities [B, num_classes]:\n{probs}")


if __name__ == "__main__":
    main()
