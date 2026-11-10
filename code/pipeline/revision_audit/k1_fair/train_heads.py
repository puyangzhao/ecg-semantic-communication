#!/usr/bin/env python
"""Phase-2A-recipe head/selector training on frozen per-lead embeddings, for either backbone.

Recipe = scripts/train_phase2a.py (unchanged logic): K resampled per batch from
{1,3,6}, Adam 1e-3 / wd 1e-4, batch 64, 60 epochs, patience 10, class-weighted
BCE, early stopping on the mean fold-9 macro-AUROC over K. Same budget for every
method. Fold 10 is refused unless --split-eval test is given after the lock.

--backbone orig     : outputs/phase2a/embeddings (frozen submitted encoder), receiver =
                      mask-aware mean pooling + linear head (Phase2ASelectionModel)
--backbone new_s{0,1,2}: revision_audit/backbone/emb/new_seed{s} (frozen revision encoder), receiver =
                      masked attention pooling + MLP head (lead_resnet.AttnPoolReceiver), fresh init
--method proposed | global_learned_k | random_k | fixed_lead (--lead NAME sets the K=1 lead;
                      K=3 / K=6 keep the submitted clinical subsets II,V2,V5 / limb leads)

Usage: python revision_audit/k1_fair/train_heads.py --backbone new_s0 --method fixed_lead --lead V5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revision_audit/backbone"))
from lead_resnet import AttnPoolReceiver  # noqa: E402
from src.data.clinical_lead_subsets import FIXED_CLINICAL_SUBSETS, LEAD_NAMES, lead_indices  # noqa: E402
from src.evaluation.calibration import compute_auprc  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.lead_selectors import CrossLeadTopKSelector, GlobalLearnedKSelector, RandomKSelector  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from src.training.losses import build_weighted_bce_loss  # noqa: E402
from src.training.seed import set_seed  # noqa: E402

BUDGETS = [1, 3, 6]


class FixedSubsetSelector(nn.Module):
    def __init__(self, k1_lead: str):
        super().__init__()
        self.subsets = {1: [k1_lead], 3: FIXED_CLINICAL_SUBSETS[3], 6: FIXED_CLINICAL_SUBSETS[6]}

    def forward(self, z, k):
        m = torch.zeros(z.shape[0], z.shape[1], device=z.device)
        m[:, lead_indices(self.subsets[k])] = 1.0
        return m


class AttnSelectionModel(nn.Module):
    """Same interface as Phase2ASelectionModel, attention-pool + MLP receiver."""

    def __init__(self, selector, latent_dim=32):
        super().__init__()
        self.selector = selector
        self.receiver = AttnPoolReceiver(latent_dim=latent_dim)

    def forward(self, z, k):
        mask = self.selector(z, k)
        # straight-through masks carry gradient; multiply so the selector still learns
        logits = self.receiver(z * mask.unsqueeze(-1), mask.detach() + (mask - mask.detach()))
        return logits, mask

    def logits_with_mask(self, z, mask):
        return self.receiver(z * mask.unsqueeze(-1), mask)


def emb_dir(backbone):
    if backbone == "orig":
        return os.path.join(ROOT, "outputs/phase2a/embeddings")
    return os.path.join(ROOT, f"revision_audit/backbone/emb/new_seed{backbone.split('_s')[1]}")


def build(method, backbone, lead):
    sel = {"proposed": lambda: CrossLeadTopKSelector(32), "global_learned_k": GlobalLearnedKSelector,
           "random_k": RandomKSelector, "fixed_lead": lambda: FixedSubsetSelector(lead)}[method]()
    if backbone == "orig":
        return Phase2ASelectionModel(sel, latent_dim=32, num_classes=5, classifier_hidden_dim=None, dropout=0.1)
    return AttnSelectionModel(sel)


@torch.no_grad()
def infer(model, z, k, bs=512):
    model.eval()
    L, M = [], []
    for i in range(0, len(z), bs):
        lg, m = model(z[i:i + bs], k)
        L.append(lg)
        M.append(m.detach())
    return torch.cat(L), torch.cat(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", required=True, choices=("proposed", "global_learned_k", "random_k", "fixed_lead"))
    ap.add_argument("--lead", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    assert (a.method == "fixed_lead") == (a.lead is not None)
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    tr = torch.load(os.path.join(emb_dir(a.backbone), "train.pt"))
    va = torch.load(os.path.join(emb_dir(a.backbone), "val.pt"))
    name = f"{a.backbone}_{a.method}{'_' + a.lead if a.lead else ''}_seed{a.seed}"
    run = os.path.join(HERE, "runs", name)
    os.makedirs(run, exist_ok=True)
    model = build(a.method, a.backbone, a.lead)
    loss_fn = build_weighted_bce_loss(tr["y"].numpy())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(a.seed)
    best, best_ep, bad, hist = -1.0, -1, 0, []
    n = tr["z"].shape[0]
    for ep in range(60):
        model.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed * 10_000 + ep))
        for i in range(0, n, 64):
            idx = order[i:i + 64]
            k = rng.choice(BUDGETS)
            opt.zero_grad()
            lg, _ = model(tr["z"][idx], k)
            loss_fn(lg, tr["y"][idx]).backward()
            opt.step()
        if a.method == "random_k":
            torch.manual_seed(a.seed * 1000 + 77)
        au = {k: compute_metrics(va["y"].numpy(), torch.sigmoid(infer(model, va["z"], k)[0]).numpy())["macro_auroc"]
              for k in BUDGETS}
        comp = float(np.mean(list(au.values())))
        hist.append({"epoch": ep, "val_auroc": au, "composite": comp})
        if comp > best:
            best, best_ep, bad = comp, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, os.path.join(run, "best.pt"))
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"))["model_state_dict"])
    res = {"backbone": a.backbone, "method": a.method, "k1_lead": a.lead, "seed": a.seed, "best_epoch": best_ep,
           "history": hist, "params": sum(p.numel() for p in model.parameters())}
    for k in BUDGETS:
        if a.method == "random_k":
            torch.manual_seed(a.seed * 1000 + k)
        lg, m = infer(model, va["z"], k)
        p = torch.sigmoid(lg).numpy()
        res[f"val_K{k}"] = {"macro_auroc": compute_metrics(va["y"].numpy(), p)["macro_auroc"],
                            "macro_auprc": compute_auprc(va["y"].numpy(), p)["macro_auprc"]}
        np.savez(os.path.join(run, f"val_K{k}.npz"), logits=lg.numpy(), mask=m.numpy(), ecg_id=va["ecg_id"].numpy())
    json.dump(res, open(os.path.join(run, "result.json"), "w"), indent=2)
    open(os.path.join(run, "DONE"), "w").write("ok\n")
    print(name, res["val_K1"], "best_epoch", best_ep)


if __name__ == "__main__":
    main()
