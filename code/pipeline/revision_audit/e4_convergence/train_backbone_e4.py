#!/usr/bin/env python
"""E4: extended training of the 824,320-parameter lead-aware encoder (revision backbone, --arch new).

Identical to revision_audit/backbone/train_backbone.py (imports its model builder, forward,
predict and masks; same data cache, seed handling, optimizer, loss, batch, lead-subset sampling,
validation composite and patience 8). The only differences:
  * max epochs 120 instead of 40 (constant LR, no scheduler -> nothing else depends on the budget);
  * outputs go to revision_audit/e4_convergence/runs/new_seed{S}_e120;
  * a full-state last.pt (model, optimizer, numpy Generator, torch RNG, early-stopping counters)
    is written after every epoch, so the run can be resumed exactly. Saving consumes no randomness.
The test split is never loaded.

Usage: python revision_audit/e4_convergence/train_backbone_e4.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revision_audit/backbone"))
from train_backbone import CACHE, K_CHOICES, K_PROBS, NEW_KW, build, forward, predict, random_masks  # noqa: E402
from src.evaluation.calibration import compute_auprc  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.losses import build_weighted_bce_loss  # noqa: E402
from src.training.seed import set_seed  # noqa: E402

ARCH, MAX_EPOCHS, PATIENCE = "new", 120, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    tr, va = np.load(f"{CACHE}/train.npz"), np.load(f"{CACHE}/val.npz")
    xtr, ytr, xva, yva = tr["x"], tr["y"], va["x"], va["y"]
    run = os.path.join(HERE, "runs", f"new_seed{a.seed}_e120")
    assert not os.path.exists(os.path.join(run, "DONE")), "already trained"
    os.makedirs(run, exist_ok=True)
    model = build(ARCH)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    loss_fn = build_weighted_bce_loss(ytr)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng_eval = np.random.default_rng(777)
    eval_masks = {12: np.ones((len(xva), 12), np.float32)}
    for k in (1, 3, 6):
        eval_masks[k] = random_masks(len(xva), k, rng_eval)
    eval_ks = [12, 1, 3, 6]
    rng = np.random.default_rng(a.seed)
    cfg = {"arch": ARCH, "seed": a.seed, "epochs_max": MAX_EPOCHS, "patience": PATIENCE, "batch": 64, "lr": 1e-3,
           "lr_schedule": "constant (no scheduler), as in the 40-epoch run", "weight_decay": 1e-4,
           "loss": "BCEWithLogits, pos_weight=n_neg/n_pos (train)", "params_total": n_params, "params_encoder": n_enc,
           "model_kw": NEW_KW, "lead_subset_training": {"K": K_CHOICES, "p": K_PROBS},
           "early_stopping": "mean val macro-AUROC over K=12 and fixed random K=1,3,6 subsets (rng 777)",
           "n_train": int(len(xtr)), "n_val": int(len(xva)), "threads": a.threads, "device": "cpu",
           "start_mode": "from scratch (original runs kept only best weights; no optimizer/RNG state)"}
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=2)
    hist, best, best_ep, bad = [], -1.0, -1, 0
    stopped_early = False
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(xtr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            xb, yb = torch.from_numpy(xtr[idx]), torch.from_numpy(ytr[idx])
            mk = np.zeros((len(idx), 12), np.float32)
            ks = rng.choice(K_CHOICES, size=len(idx), p=K_PROBS)
            for j, k in enumerate(ks):
                mk[j, rng.choice(12, size=k, replace=False)] = 1
            mb = torch.from_numpy(mk)
            opt.zero_grad()
            loss = loss_fn(forward(model, ARCH, xb, mb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        rec = {"epoch": ep, "train_loss": tot / nb, "minutes": (time.time() - t0) / 60}
        for k in eval_ks:
            p = predict(model, ARCH, xva, eval_masks[k])
            rec[f"val_auroc_K{k}"] = compute_metrics(yva, p)["macro_auroc"]
            rec[f"val_auprc_K{k}"] = compute_auprc(yva, p)["macro_auprc"]
        comp = float(np.mean([rec[f"val_auroc_K{k}"] for k in eval_ks]))
        rec["composite"] = comp
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if comp > best:
            best, best_ep, bad = comp, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "composite": comp},
                       os.path.join(run, "best.pt"))
        else:
            bad += 1
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "epoch": ep,
                    "np_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                    "best": best, "best_epoch": best_ep, "bad": bad, "history": hist},
                   os.path.join(run, "last.pt"))
        if bad >= PATIENCE:
            stopped_early = True
            break
    json.dump({"history": hist, "best_epoch": best_ep, "best_composite": best, "final_epoch": ep,
               "stopped_by_patience": stopped_early, "runtime_min": (time.time() - t0) / 60,
               "best_row": hist[best_ep]}, open(os.path.join(run, "train_result.json"), "w"), indent=2)
    open(os.path.join(run, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
