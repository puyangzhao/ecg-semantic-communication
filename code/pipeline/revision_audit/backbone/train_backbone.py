#!/usr/bin/env python
"""Train a backbone on cached PTB-XL waveforms (folds 1-8 train, fold 9 val), CPU.

--arch orig : the submitted encoder (src/models/full_model.py ECGBaselineModel:
              SharedLeadEncoder base 16 / 3 blocks / d=32, mean pooling,
              linear head), trained with the submitted recipe
              (configs/baseline.yaml: 12 leads, Adam 1e-3, wd 1e-4, batch 64,
              30 epochs, patience 8, weighted BCE, early stopping on val
              12-lead macro-AUROC). Re-trained with seeds 0-2 on CPU so both
              backbones are compared over the same seeds.
--arch new  : revision_audit/backbone/lead_resnet.py LeadAwareModel
              (lead-ID-conditioned per-lead residual encoder, d=32, masked
              attention-pooling receiver + MLP head). Same optimizer, batch,
              loss, patience; max 40 epochs; per-record random lead subsets
              (K in {1,2,3,4,6,12}, p={.2,.15,.15,.1,.15,.25}); early stopping
              on the mean val macro-AUROC over K=12 and fixed random subsets of
              K=1,3,6 (masks drawn once, rng 777).
The test split is never loaded.

Usage: python revision_audit/backbone/train_backbone.py --arch new --seed 0 [--dev]
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
sys.path.insert(0, HERE)
from lead_resnet import LeadAwareModel  # noqa: E402
from src.evaluation.calibration import compute_auprc  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.full_model import ECGBaselineModel  # noqa: E402
from src.training.losses import build_weighted_bce_loss  # noqa: E402
from src.training.seed import set_seed  # noqa: E402

CACHE = "cache/ptbxl100"
K_CHOICES, K_PROBS = [1, 2, 3, 4, 6, 12], [0.20, 0.15, 0.15, 0.10, 0.15, 0.25]
NEW_KW = dict(widths=(32, 64, 96, 128), blocks_per_stage=2, latent_dim=32)


def random_masks(n, k, rng):
    m = np.zeros((n, 12), np.float32)
    for i in range(n):
        m[i, rng.choice(12, size=k, replace=False)] = 1
    return m


def build(arch):
    if arch == "orig":
        return ECGBaselineModel(num_leads=12, num_classes=5, latent_dim=32, fusion="mean", base_channels=16,
                                num_blocks=3, classifier_hidden_dim=None, dropout=0.1)
    return LeadAwareModel(**NEW_KW)


def forward(model, arch, x, mask):
    return model(x) if arch == "orig" else model(x, mask)


@torch.no_grad()
def predict(model, arch, x, mask, bs=256):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(forward(model, arch, torch.from_numpy(x[i:i + bs]), torch.from_numpy(mask[i:i + bs])))
    return torch.sigmoid(torch.cat(out)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("orig", "new"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dev", action="store_true", help="1500 train records, 2 epochs: data/training sanity check")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    tr, va = np.load(f"{CACHE}/train.npz"), np.load(f"{CACHE}/val.npz")
    xtr, ytr, xva, yva = tr["x"], tr["y"], va["x"], va["y"]
    if a.dev:
        xtr, ytr = xtr[:1500], ytr[:1500]
    epochs = 2 if a.dev else (30 if a.arch == "orig" else 40)
    patience = 8
    name = f"{a.arch}_seed{a.seed}" + ("_dev" if a.dev else "")
    run = os.path.join(HERE, "runs", name)
    os.makedirs(run, exist_ok=True)
    model = build(a.arch)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    loss_fn = build_weighted_bce_loss(ytr)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng_eval = np.random.default_rng(777)
    eval_masks = {12: np.ones((len(xva), 12), np.float32)}
    for k in (1, 3, 6):
        eval_masks[k] = random_masks(len(xva), k, rng_eval)
    eval_ks = [12] if a.arch == "orig" else [12, 1, 3, 6]
    rng = np.random.default_rng(a.seed)
    cfg = {"arch": a.arch, "seed": a.seed, "epochs_max": epochs, "patience": patience, "batch": 64, "lr": 1e-3,
           "weight_decay": 1e-4, "loss": "BCEWithLogits, pos_weight=n_neg/n_pos (train)",
           "params_total": n_params, "params_encoder": n_enc, "model_kw": NEW_KW if a.arch == "new" else "baseline.yaml",
           "lead_subset_training": None if a.arch == "orig" else {"K": K_CHOICES, "p": K_PROBS},
           "early_stopping": "val 12-lead macro-AUROC" if a.arch == "orig" else
           "mean val macro-AUROC over K=12 and fixed random K=1,3,6 subsets (rng 777)",
           "n_train": int(len(xtr)), "n_val": int(len(xva)), "threads": a.threads, "device": "cpu"}
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=2)
    hist, best, best_ep, bad = [], -1.0, -1, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(xtr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            xb, yb = torch.from_numpy(xtr[idx]), torch.from_numpy(ytr[idx])
            if a.arch == "new":
                mk = np.zeros((len(idx), 12), np.float32)
                ks = rng.choice(K_CHOICES, size=len(idx), p=K_PROBS)
                for j, k in enumerate(ks):
                    mk[j, rng.choice(12, size=k, replace=False)] = 1
                mb = torch.from_numpy(mk)
            else:
                mb = None
            opt.zero_grad()
            loss = loss_fn(forward(model, a.arch, xb, mb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        rec = {"epoch": ep, "train_loss": tot / nb, "minutes": (time.time() - t0) / 60}
        for k in eval_ks:
            p = predict(model, a.arch, xva, eval_masks[k])
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
            if bad >= patience:
                break
    json.dump({"history": hist, "best_epoch": best_ep, "best_composite": best, "final_epoch": ep,
               "runtime_min": (time.time() - t0) / 60, "best_row": hist[best_ep]},
              open(os.path.join(run, "train_result.json"), "w"), indent=2)
    open(os.path.join(run, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
