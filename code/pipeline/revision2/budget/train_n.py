#!/usr/bin/env python
"""PLAN §4: train the transmission models at N in {8, 32} complex channel uses with the unchanged recipe of
revision_audit/comm/train_comm.py (folds 1-8 train, fold 9 early stopping; AWGN, SNR per batch from
{-5,0,5,10,15,20} dB; (m, r) per batch from the N-specific set; Adam 1e-3 / wd 1e-4, batch 64, class-weighted BCE;
40 epochs max, patience 8; SSCC heads: linear 32->5 on noiseless dequantized embeddings, 60 epochs, patience 10).

Usage: train_n.py --method {rep_ctrl,jscc_ctrl,jscc_e2e,sscc_heads} --n {8,32} --seed s
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
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import r2common as C  # noqa: E402
from models_n import MR, JSCCN, RepCtrlN  # noqa: E402
from sscc_n import CONFIGS_N, SSCCN  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.losses import build_weighted_bce_loss  # noqa: E402
from src.training.seed import set_seed  # noqa: E402
from train_comm import EMB, TRAIN_SNRS, proposed_leads  # noqa: E402


def auroc(y, logits):
    return compute_metrics(y, torch.sigmoid(logits).numpy())["macro_auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=("rep_ctrl", "jscc_ctrl", "jscc_e2e", "sscc_heads"))
    ap.add_argument("--n", type=int, required=True, choices=(8, 32))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    tr, va = torch.load(os.path.join(EMB, "train.pt")), torch.load(os.path.join(EMB, "val.pt"))
    ltr, lva = proposed_leads(a.seed, "train"), proposed_leads(a.seed, "val")
    z1tr = tr["z"][torch.arange(len(ltr)), torch.from_numpy(ltr)]
    z1va = va["z"][torch.arange(len(lva)), torch.from_numpy(lva)]
    ytr, yva = tr["y"], va["y"]
    run = os.path.join(HERE, "runs", f"{a.method}_n{a.n}_seed{a.seed}")
    os.makedirs(run, exist_ok=True)
    loss_fn = build_weighted_bce_loss(ytr.numpy())

    if a.method == "sscc_heads":
        out = {}
        for name in CONFIGS_N[a.n]:
            sc = SSCCN(name, a.n, seed=a.seed).fit(z1tr.numpy())
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
                pickle.dump({"books": sc.pq.books, "scale": sc.scale, "cfg": sc.cfg, "n": a.n}, f)
            e_tr = (np.abs(sc.tx(z1tr.numpy())) ** 2).sum(1)
            out[name] = {"val_noiseless_macro_auroc": best, "best_epoch": best_ep, "info_bits": sc.info_bits,
                         "coded_bits": sc.coded_bits, "constellation_scale": sc.scale, "train_mean_energy": float(e_tr.mean()),
                         "train_energy_min_max": [float(e_tr.min()), float(e_tr.max())],
                         "val_quant_mse": float(((qva - z1va) ** 2).mean())}
            print(name, out[name], flush=True)
        C.dump(out, os.path.join(run, "result.json"))
        open(os.path.join(run, "DONE"), "w").write("ok\n")
        return

    mr = MR[a.n]
    model = RepCtrlN(a.n) if a.method == "rep_ctrl" else JSCCN(32 if a.method == "jscc_ctrl" else 384, a.n)
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
                m, r = rng.choice(mr)
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
                    vals += [auroc(yva.numpy(), model(src_va, m, r, snr)) for m, r in mr]
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
    C.dump({"method": a.method, "n": a.n, "seed": a.seed, "best_epoch": best_ep, "best_composite": best, "history": hist,
            "epochs_max": 40, "patience": 8, "mr": mr,
            "params": sum(p.numel() for p in model.parameters()), "train_snr_db_per_batch": TRAIN_SNRS, "train_channel": "AWGN"},
           os.path.join(run, "result.json"))
    open(os.path.join(run, "DONE"), "w").write("ok\n")
    print(a.method, a.n, a.seed, "best", best, "epoch", best_ep)


if __name__ == "__main__":
    main()
