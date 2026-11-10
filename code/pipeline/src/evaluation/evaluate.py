"""Deterministic test-set evaluation using a saved checkpoint."""
from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.training.train import run_inference


def evaluate_checkpoint(model, checkpoint_path: str, test_loader: DataLoader, device: str,
                         threshold: float = 0.5, output_path: str | None = None) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    y_true, y_prob = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["checkpoint_epoch"] = ckpt.get("epoch")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics
