"""Clean-channel baseline training loop.

Checkpointing policy: only `best.pt` (highest validation macro-AUROC so far)
and `final.pt` (last epoch) are ever kept on disk for a given run — no
per-epoch or batch-level checkpoints, per the project's storage rules.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


def collect_env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ).decode().strip()
    except Exception:
        info["git_commit"] = None
    return info


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def train_model(model, train_loader, val_loader, loss_fn, config: dict, run_dir: str,
                 device: str | None = None) -> dict:
    device = device or config["train"]["device"]
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"],
    )

    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "env_info.json"), "w") as f:
        json.dump(collect_env_info(), f, indent=2)

    history = []
    best_val_auroc = -1.0
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()

    num_epochs = config["train"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        y_val, prob_val = run_inference(model, val_loader, device)
        val_metrics = compute_metrics(y_val, prob_val, threshold=config["eval"]["threshold"])
        val_auroc = val_metrics["macro_auroc"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_auroc": val_auroc,
            "val_macro_f1": val_metrics["macro_f1"],
        })
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_macro_auroc={val_auroc:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")

        improved = (not np.isnan(val_auroc)) and (val_auroc > best_val_auroc)
        if improved:
            best_val_auroc = val_auroc
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_auroc": val_auroc}, os.path.join(ckpt_dir, "best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= config["train"]["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch} (no val improvement for "
                  f"{config['train']['early_stopping_patience']} epochs)")
            break

    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                "val_macro_auroc": val_auroc}, os.path.join(ckpt_dir, "final.pt"))

    runtime_sec = time.time() - start_time
    peak_gpu_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() and "cuda" in str(device)
        else None
    )

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_val_auroc,
        "final_epoch": epoch,
        "runtime_sec": runtime_sec,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
    }
    with open(os.path.join(run_dir, "train_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
