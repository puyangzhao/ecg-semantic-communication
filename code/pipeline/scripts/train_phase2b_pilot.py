#!/usr/bin/env python
"""Phase 2B validation-only training: K=1, AWGN only, seed in {0,1,2}.

Trains ONE shared Phase2BChannelModel (lead selector + global symbol ranker
+ classifier) with (m,r) and SNR resampled per training batch, on the
TRAINING fold only. Evaluates on the VALIDATION fold only, at every SNR in
the grid, for r in {1,2,4} (fixed) plus the validation-selected adaptive
policy and the validation oracle. The test fold is never loaded.

Originally the seed=0-only pilot (see docs/phase2b_design.md section 13);
generalized to seed in {0,1,2} for the three-seed confirmatory stage
(docs/phase2b_mechanism_diagnostic.md) without changing any modeling or
evaluation logic -- only the seed value and run-name vary across seeds.

Usage:
    python scripts/train_phase2b_pilot.py --config configs/phase2b_pilot.yaml \
        --run-name phase2b_pilot_k1_awgn_seed0
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

from src.channels.repetition import N_SYMBOLS, mr_pairs
from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_with_thresholds,
    select_per_class_thresholds,
)
from src.models.lead_selectors import CrossLeadTopKSelector
from src.models.phase2b_model import Phase2BChannelModel
from src.models.symbol_rank_selector import GlobalSymbolRankSelector
from src.training.losses import build_weighted_bce_loss
from src.training.seed import set_seed
from src.training.train import collect_env_info

K_FIXED = 1
CHANNEL_TYPE = "awgn"


def load_split(embeddings_dir: str, split: str) -> dict:
    assert split in ("train", "val"), "Phase 2B pilot must not load the test split"
    return torch.load(os.path.join(embeddings_dir, f"{split}.pt"))


def iterate_batches(z, y, batch_size, shuffle, seed):
    n = z.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield z[idx], y[idx]


@torch.no_grad()
def run_inference(model, z, device, m, r, snr_db, batch_size=512):
    model.eval()
    all_logits = []
    for start in range(0, z.shape[0], batch_size):
        zb = z[start:start + batch_size].to(device)
        logits, _, diag = model(zb, k=K_FIXED, m=m, r=r, snr_db=snr_db, channel_type=CHANNEL_TYPE)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0), diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    assert seed in (0, 1, 2), "Phase 2B K=1/AWGN stage uses seed in {0,1,2} only"
    set_seed(seed)

    snr_grid = config["snr_grid_db"]
    device = config["train"]["device"]

    train_data = load_split(config["embeddings_dir"], "train")
    val_data = load_split(config["embeddings_dir"], "val")

    lead_selector = CrossLeadTopKSelector(latent_dim=config["d"])
    symbol_ranker = GlobalSymbolRankSelector()
    model = Phase2BChannelModel(
        lead_selector, symbol_ranker, latent_dim=config["d"], num_classes=5,
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

    with open(os.path.join(run_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "env_info.json"), "w") as f:
        json.dump(collect_env_info(), f, indent=2)

    rng = random.Random(seed)
    mr_choices = mr_pairs()  # [(16,1),(8,2),(4,4)]
    history = []
    best_composite, best_epoch = -1.0, -1
    patience_counter = 0
    start_time = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    num_epochs = config["train"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for zb, yb in iterate_batches(train_data["z"], train_data["y"], config["train"]["batch_size"],
                                       shuffle=True, seed=seed * 10_000 + epoch):
            zb, yb = zb.to(device), yb.to(device)
            m, r = rng.choice(mr_choices)
            snr_db = rng.choice(snr_grid)
            optimizer.zero_grad()
            logits, _, _ = model(zb, k=K_FIXED, m=m, r=r, snr_db=snr_db, channel_type=CHANNEL_TYPE)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # composite validation metric: mean macro-AUROC over the full (snr, m/r) grid
        val_aurocs = {}
        for snr_db in snr_grid:
            for m, r in mr_choices:
                logits, _ = run_inference(model, val_data["z"], device, m, r, snr_db)
                probs = torch.sigmoid(logits).numpy()
                auroc = compute_metrics(val_data["y"].numpy(), probs,
                                         threshold=config["eval"]["threshold"])["macro_auroc"]
                val_aurocs[f"snr{snr_db}_r{r}"] = auroc
        valid_vals = [v for v in val_aurocs.values() if not np.isnan(v)]
        composite = float(np.mean(valid_vals)) if valid_vals else float("nan")

        history.append({"epoch": epoch, "train_loss": train_loss, "composite_val_macro_auroc": composite})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} composite_val_macro_auroc={composite:.4f}")

        improved = (not np.isnan(composite)) and (composite > best_composite)
        if improved:
            best_composite, best_epoch = composite, epoch
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
    peak_gpu_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.startswith("cuda") and torch.cuda.is_available() else None
    )
    train_result = {"history": history, "best_epoch": best_epoch,
                     "best_composite_val_macro_auroc": best_composite,
                     "final_epoch": epoch, "runtime_sec": runtime_sec, "peak_gpu_mem_mb": peak_gpu_mem_mb}
    with open(os.path.join(run_dir, "train_result.json"), "w") as f:
        json.dump(train_result, f, indent=2)

    # ---- Load best checkpoint, build the full pilot report on VALIDATION ONLY ----
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Per-class thresholds selected ONCE, on validation, at the reference
    # condition (highest SNR, r=1/m=16 -- least-noisy operating point), then
    # frozen and applied at every other (snr, r) condition. Never re-selected
    # per condition (that would trivially inflate F1 at every point).
    ref_snr = max(snr_grid)
    logits_ref, _ = run_inference(model, val_data["z"], device, m=N_SYMBOLS, r=1, snr_db=ref_snr)
    probs_ref = torch.sigmoid(logits_ref).numpy()
    frozen_thresholds = select_per_class_thresholds(val_data["y"].numpy(), probs_ref)

    report = {"snr_grid_db": snr_grid, "k": K_FIXED, "channel_type": CHANNEL_TYPE, "seed": seed,
              "validation_thresholds": frozen_thresholds, "threshold_reference_condition": {"snr_db": ref_snr, "r": 1},
              "per_snr": {}}

    for snr_db in snr_grid:
        per_r = {}
        for m, r in mr_choices:
            logits, diag = run_inference(model, val_data["z"], device, m, r, snr_db)
            probs = torch.sigmoid(logits).numpy()
            metrics = compute_metrics(val_data["y"].numpy(), probs, threshold=config["eval"]["threshold"])
            metrics_thresh = compute_metrics_with_thresholds(val_data["y"].numpy(), probs, frozen_thresholds)
            per_r[str(r)] = {
                "m": m, "macro_auroc": metrics["macro_auroc"],
                "macro_f1_fixed_thresh": metrics["macro_f1"],
                "macro_f1_val_thresholded": metrics_thresh["macro_f1"],
                "per_class_auroc": metrics["per_class_auroc"],
                "channel_uses_per_lead": diag["channel_uses_per_lead"],
                "total_payload_channel_uses": diag["total_payload_channel_uses"],
                "avg_symbol_power": diag["avg_symbol_power"],
                "total_codeword_energy_per_lead": diag["total_codeword_energy_per_lead"],
            }
        best_r = max(per_r, key=lambda rr: per_r[rr]["macro_auroc"])
        report["per_snr"][str(snr_db)] = {
            "per_r": per_r,
            "validation_selected_r": best_r,
            "validation_oracle_r": best_r,
            "validation_oracle_macro_auroc": per_r[best_r]["macro_auroc"],
        }

    report_path = os.path.join(run_dir, "phase2b_pilot_report_raw.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_path}")
    print("Training result:", json.dumps({"best_epoch": best_epoch, "best_composite_val_macro_auroc": best_composite,
                                           "runtime_sec": runtime_sec, "peak_gpu_mem_mb": peak_gpu_mem_mb}, indent=2))


if __name__ == "__main__":
    main()
