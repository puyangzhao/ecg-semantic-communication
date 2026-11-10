#!/usr/bin/env python
"""Train the controlled / end-to-end transmission models (folds 1-8 train, fold 9 early stopping).

--method rep_ctrl | jscc_ctrl | jscc_e2e | sscc_heads ; --seed s (0,1,2)
Lead for *_ctrl and SSCC = Phase 2A proposed seed-s selector's K=1 choice
(outputs/runs/phase2a_proposed_seed{s}); embeddings = frozen submitted encoder.
Training channel = AWGN only, SNR drawn per batch from {-5,0,5,10,15,20} dB
(the Phase 2B recipe); rep_ctrl additionally draws (m,r) per batch from
{(16,1),(8,2),(4,4)}. 40 epochs max, patience 8, Adam 1e-3 / wd 1e-4, batch 64,
class-weighted BCE; early stopping on mean fold-9 macro-AUROC over the same SNR
set (and r for rep_ctrl) with fixed torch noise seeds.
SSCC heads: linear 32->5 on noiseless dequantized embeddings, Phase-2A head
recipe (60 epochs, patience 10, val macro-AUROC on noiseless dequantized z).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from comm_models import JSCC, RepCtrl  # noqa: E402
from scripts.train_phase2a import build_selector  # noqa: E402
from sscc import CONFIGS, SSCC  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from src.training.losses import build_weighted_bce_loss  # noqa: E402
from src.training.seed import set_seed  # noqa: E402

EMB = os.path.join(ROOT, "outputs/phase2a/embeddings")
TRAIN_SNRS = [-5, 0, 5, 10, 15, 20]
MR = [(16, 1), (8, 2), (4, 4)]


def proposed_leads(seed, split):
    path = os.path.join(HERE, "leads", f"proposed_seed{seed}_{split}.npy")
    if os.path.exists(path):
        return np.load(path)
    assert split != "test" or os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json"))
    m = Phase2ASelectionModel(build_selector("proposed", 32), latent_dim=32, num_classes=5)
    m.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2a_proposed_seed{seed}/checkpoints/best.pt"),
                                 map_location="cpu")["model_state_dict"])
    z = torch.load(os.path.join(EMB, f"{split}.pt"))["z"]
    with torch.no_grad():
        lead = m.selector.raw_scores(z).argmax(1).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, lead)
    return lead


def auroc(y, logits):
    return compute_metrics(y, torch.sigmoid(logits).numpy())["macro_auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=("rep_ctrl", "jscc_ctrl", "jscc_e2e", "sscc_heads"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    tr, va = torch.load(os.path.join(EMB, "train.pt")), torch.load(os.path.join(EMB, "val.pt"))
    ltr, lva = proposed_leads(a.seed, "train"), proposed_leads(a.seed, "val")
    z1tr = tr["z"][torch.arange(len(ltr)), torch.from_numpy(ltr)]
    z1va = va["z"][torch.arange(len(lva)), torch.from_numpy(lva)]
    ytr, yva = tr["y"], va["y"]
    run = os.path.join(HERE, "runs", f"{a.method}_seed{a.seed}")
    os.makedirs(run, exist_ok=True)
    loss_fn = build_weighted_bce_loss(ytr.numpy())

    if a.method == "sscc_heads":
        out = {}
        for name in CONFIGS:
            sc = SSCC(name, seed=a.seed).fit(z1tr.numpy())
            qtr = torch.from_numpy(sc.pq.decode(sc.pq.encode(z1tr.numpy())))
            qva = torch.from_numpy(sc.pq.decode(sc.pq.encode(z1va.numpy())))
            head = torch.nn.Linear(32, 5)
            opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            best, bad, best_ep = -1, 0, -1
            for ep in range(60):
                order = torch.randperm(len(qtr), generator=torch.Generator().manual_seed(a.seed * 10_000 + ep))
                for i in range(0, len(order), 64):
                    idx = order[i:i + 64]
                    opt.zero_grad()
                    loss_fn(head(qtr[idx]), ytr[idx]).backward()
                    opt.step()
                with torch.no_grad():
                    au = auroc(yva.numpy(), head(qva))
                if au > best:
                    best, bad, best_ep, state = au, 0, ep, {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10:
                        break
            torch.save(state, os.path.join(run, f"head_{name}.pt"))
            with open(os.path.join(run, f"codec_{name}.pkl"), "wb") as f:
                pickle.dump({"books": sc.pq.books, "scale": sc.scale, "cfg": sc.cfg}, f)
            e_tr = float((np.abs(sc.tx(z1tr.numpy())) ** 2).sum(1).mean())
            out[name] = {"val_noiseless_macro_auroc": best, "best_epoch": best_ep, "info_bits": sc.info_bits,
                         "coded_bits": sc.coded_bits, "constellation_scale": sc.scale, "train_mean_energy": e_tr,
                         "val_quant_mse": float(((qva - z1va) ** 2).mean())}
            print(name, out[name], flush=True)
        json.dump(out, open(os.path.join(run, "result.json"), "w"), indent=2)
        open(os.path.join(run, "DONE"), "w").write("ok\n")
        return

    model = RepCtrl() if a.method == "rep_ctrl" else JSCC(32 if a.method == "jscc_ctrl" else 384)
    src_tr = z1tr if a.method != "jscc_e2e" else tr["z"].reshape(len(ltr), -1)
    src_va = z1va if a.method != "jscc_e2e" else va["z"].reshape(len(lva), -1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(a.seed)
    best, bad, best_ep, hist = -1, 0, -1, []
    for ep in range(40):
        model.train()
        order = torch.randperm(len(src_tr), generator=torch.Generator().manual_seed(a.seed * 10_000 + ep))
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            snr = rng.choice(TRAIN_SNRS)
            opt.zero_grad()
            if a.method == "rep_ctrl":
                m, r = rng.choice(MR)
                lg = model(src_tr[idx], m, r, snr)
            else:
                lg = model(src_tr[idx], snr)
            loss_fn(lg, ytr[idx]).backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for si, snr in enumerate(TRAIN_SNRS):
                torch.manual_seed(900_000 + si)
                if a.method == "rep_ctrl":
                    vals += [auroc(yva.numpy(), model(src_va, m, r, snr)) for m, r in MR]
                else:
                    vals.append(auroc(yva.numpy(), model(src_va, snr)))
        comp = float(np.mean(vals))
        hist.append({"epoch": ep, "composite": comp})
        if comp > best:
            best, bad, best_ep = comp, 0, ep
            torch.save(model.state_dict(), os.path.join(run, "best.pt"))
        else:
            bad += 1
            if bad >= 8:
                break
    json.dump({"method": a.method, "seed": a.seed, "best_epoch": best_ep, "best_composite": best, "history": hist,
               "params": sum(p.numel() for p in model.parameters()),
               "params_tx": sum(p.numel() for n_, p in model.named_parameters() if n_.startswith(("enc", "ranker"))),
               "params_rx": sum(p.numel() for n_, p in model.named_parameters() if n_.startswith(("dec", "head"))),
               "train_snr_db_per_batch": TRAIN_SNRS, "train_channel": "AWGN"},
              open(os.path.join(run, "result.json"), "w"), indent=2)
    open(os.path.join(run, "DONE"), "w").write("ok\n")
    print(a.method, a.seed, "best", best, "epoch", best_ep)


if __name__ == "__main__":
    main()
