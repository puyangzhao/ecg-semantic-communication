#!/usr/bin/env python
"""Phase 2A training/eval: one run = one (method, seed).

Trains the fusion+classifier head (and the selector's own parameters, for
the two learned methods) on top of frozen, precomputed Phase 1 lead
embeddings, with K resampled per training batch from `budgets` (clean
channel only -- no AWGN/Rayleigh/SNR sweep). See docs/phase2a_design.md.

Usage:
    python scripts/train_phase2a.py --method proposed --seed 0 \
        --config configs/phase2a.yaml --run-name phase2a_proposed_seed0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.labels import SUPERCLASSES
from src.evaluation.channel_cost import channel_cost_report
from src.evaluation.metrics import compute_metrics
from src.evaluation.selection_diagnostics import (
    pairwise_co_selection,
    selection_by_label,
    selection_entropy,
    selection_frequency,
)
from src.evaluation.subgroup_metrics import recode_age_group, subgroup_auroc_table, worst_group_and_gap
from src.models.lead_selectors import (
    CrossLeadTopKSelector,
    FixedClinicalKSelector,
    FullLeadSelector,
    GlobalLearnedKSelector,
    RandomKSelector,
)
from src.models.phase2a_model import Phase2ASelectionModel
from src.training.losses import build_weighted_bce_loss
from src.training.seed import set_seed
from src.training.train import collect_env_info

METHODS = ("random_k", "fixed_clinical_k", "global_learned_k", "proposed", "ceiling_k12")


def build_selector(method: str, d: int) -> torch.nn.Module:
    if method == "random_k":
        return RandomKSelector()
    if method == "fixed_clinical_k":
        return FixedClinicalKSelector()
    if method == "global_learned_k":
        return GlobalLearnedKSelector()
    if method == "proposed":
        return CrossLeadTopKSelector(latent_dim=d)
    if method == "ceiling_k12":
        return FullLeadSelector()
    raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")


def load_split(embeddings_dir: str, split: str, subset_n: int | None, seed: int) -> dict:
    data = torch.load(os.path.join(embeddings_dir, f"{split}.pt"))
    if subset_n is not None:
        n = min(subset_n, data["z"].shape[0])
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(data["z"].shape[0], generator=g)[:n]
        data = {k: v[idx] for k, v in data.items()}
    return data


def iterate_batches(z: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int):
    n = z.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield z[idx], y[idx]


@torch.no_grad()
def run_inference_at_k(model, z: torch.Tensor, device: str, k: int, batch_size: int = 512):
    model.eval()
    all_logits, all_masks = [], []
    for start in range(0, z.shape[0], batch_size):
        zb = z[start:start + batch_size].to(device)
        logits, mask = model(zb, k)
        all_logits.append(logits.cpu())
        all_masks.append(mask.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_masks, dim=0)


def evaluate_and_report(model, method: str, k: int, seed: int, test_data: dict, device: str,
                         threshold: float, d: int, run_dir: str) -> dict:
    if method == "random_k":
        torch.manual_seed(seed * 1000 + k)  # deterministic test-time draw for reproducibility
    logits, mask = run_inference_at_k(model, test_data["z"], device, k)
    y_true = test_data["y"].numpy()
    y_prob = torch.sigmoid(logits).numpy()
    mask_np = mask.numpy()

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    with open(os.path.join(run_dir, f"test_metrics_K{k}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        os.path.join(run_dir, f"test_raw_K{k}.npz"),
        y_true=y_true, y_prob=y_prob, mask=mask_np,
        patient_id=test_data["patient_id"].numpy(),
        sex=test_data["sex"].numpy(), age=test_data["age"].numpy(),
        ecg_id=test_data["ecg_id"].numpy(),
    )

    cost = channel_cost_report(method, k, d=d)
    with open(os.path.join(run_dir, f"channel_cost_K{k}.json"), "w") as f:
        json.dump(cost, f, indent=2)

    diagnostics = {
        "selection_frequency": selection_frequency(mask_np),
        "selection_entropy": selection_entropy(mask_np),
        "pairwise_co_selection": pairwise_co_selection(mask_np),
        "selection_by_label": selection_by_label(mask_np, y_true),
    }
    with open(os.path.join(run_dir, f"selection_diagnostics_K{k}.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    sex_table = subgroup_auroc_table(y_true, y_prob, test_data["sex"].numpy().astype(str))
    age_groups = recode_age_group(test_data["age"].numpy())
    age_table = subgroup_auroc_table(y_true, y_prob, age_groups)
    subgroup_report = {
        "sex": {"table": sex_table, "worst_group_and_gap": worst_group_and_gap(sex_table)},
        "age_group": {"table": age_table, "worst_group_and_gap": worst_group_and_gap(age_table)},
    }
    with open(os.path.join(run_dir, f"subgroup_metrics_K{k}.json"), "w") as f:
        json.dump(subgroup_report, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    d = config["d"]
    budgets = [12] if args.method == "ceiling_k12" else list(config["budgets"])
    subset_n_train = config.get("subset_n_train")
    subset_n_val = config.get("subset_n_val")
    subset_n_test = config.get("subset_n_test")

    train_data = load_split(config["embeddings_dir"], "train", subset_n_train, args.seed)
    val_data = load_split(config["embeddings_dir"], "val", subset_n_val, args.seed)
    test_data = load_split(config["embeddings_dir"], "test", subset_n_test, args.seed)

    device = config["train"]["device"]
    selector = build_selector(args.method, d)
    model = Phase2ASelectionModel(
        selector, latent_dim=d, num_classes=5,
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    loss_fn = build_weighted_bce_loss(train_data["y"].numpy())
    loss_fn.pos_weight = loss_fn.pos_weight.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"],
                                  weight_decay=config["train"]["weight_decay"])

    run_dir = os.path.join(args.outputs_dir, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    resolved_config = dict(config)
    resolved_config["method"] = args.method
    resolved_config["seed"] = args.seed
    resolved_config["budgets_used"] = budgets
    with open(os.path.join(run_dir, "resolved_config.json"), "w") as f:
        json.dump(resolved_config, f, indent=2)
    with open(os.path.join(run_dir, "env_info.json"), "w") as f:
        json.dump(collect_env_info(), f, indent=2)

    rng = random.Random(args.seed)
    history = []
    best_composite = -1.0
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()

    num_epochs = config["train"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for zb, yb in iterate_batches(train_data["z"], train_data["y"], config["train"]["batch_size"],
                                       shuffle=True, seed=args.seed * 10_000 + epoch):
            zb, yb = zb.to(device), yb.to(device)
            k = rng.choice(budgets)
            optimizer.zero_grad()
            logits, _ = model(zb, k)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        val_aurocs = {}
        for k in budgets:
            logits, _ = run_inference_at_k(model, val_data["z"], device, k)
            probs = torch.sigmoid(logits).numpy()
            val_aurocs[k] = compute_metrics(val_data["y"].numpy(), probs,
                                             threshold=config["eval"]["threshold"])["macro_auroc"]
        valid_vals = [v for v in val_aurocs.values() if not np.isnan(v)]
        composite = float(np.mean(valid_vals)) if valid_vals else float("nan")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_macro_auroc_by_k": val_aurocs, "composite_val_macro_auroc": composite})
        print(f"[epoch {epoch}] method={args.method} seed={args.seed} train_loss={train_loss:.4f} "
              f"composite_val_macro_auroc={composite:.4f} per_k={val_aurocs}")

        improved = (not np.isnan(composite)) and (composite > best_composite)
        if improved:
            best_composite = composite
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "composite_val_macro_auroc": composite}, os.path.join(ckpt_dir, "best.pt"))
        else:
            patience_counter += 1
        if patience_counter >= config["train"]["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                "composite_val_macro_auroc": composite}, os.path.join(ckpt_dir, "final.pt"))

    runtime_sec = time.time() - start_time
    train_result = {"history": history, "best_epoch": best_epoch,
                     "best_composite_val_macro_auroc": best_composite,
                     "final_epoch": epoch, "runtime_sec": runtime_sec}
    with open(os.path.join(run_dir, "train_result.json"), "w") as f:
        json.dump(train_result, f, indent=2)

    ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_summary = {}
    for k in budgets:
        metrics = evaluate_and_report(model, args.method, k, args.seed, test_data, device,
                                       config["eval"]["threshold"], d, run_dir)
        test_summary[k] = {"macro_auroc": metrics["macro_auroc"], "macro_f1": metrics["macro_f1"]}
        print(f"Test K={k}: macro_auroc={metrics['macro_auroc']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    print("Run complete:", json.dumps(test_summary, indent=2))


if __name__ == "__main__":
    main()
