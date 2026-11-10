#!/usr/bin/env python
"""PLAN §4: evaluate every method at N in {8, 32} complex channel uses (AWGN and block Rayleigh, perfect CSI),
paired draws nested in the existing CRNBank. --check16 runs the same code at N = 16 with the existing checkpoints
and must reproduce the cached awgn_perfect / rayleigh_perfect logits.

  rep_orig : submitted Phase-2B checkpoint, FROZEN TRANSFER (its ranking and head applied to the feasible (m, r))
  rep_ctrl, jscc_ctrl, jscc_e2e, sscc_* : retrained at N (revision2/budget/runs), existing ones at N = 16

Usage: eval_n.py --split val --seed 0 --n 8 ; eval_n.py --split val --seed 0 --n 16 --check16
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import r2common as C  # noqa: E402
import eval_comm as EC  # noqa: E402
from channel_sim import CRNBank, SNR_GRID  # noqa: E402
from comm_models import from_complex  # noqa: E402
from models_n import R_OPTIONS, JSCCN, RepCtrlN, RepSchemeN  # noqa: E402
from sscc_n import CONFIGS_N, SSCCN  # noqa: E402

EXTRA_SEED = {"val": 20261321, "test": 20261322}
KINDS = ["awgn", "rayleigh"]


def noise_bank(split, n_rec, J, n):
    bank = CRNBank(split, n_rec, J)
    if n <= 16:
        w = bank.w[..., :n]
    else:
        rng = np.random.default_rng(EXTRA_SEED[split])
        s = (J, n_rec, n - 16)
        extra = ((rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)).astype(np.complex64)
        w = np.concatenate([bank.w, extra], -1)
    return w, bank.h


def channel(x, snr_db, w, h, kind):
    xx = np.broadcast_to(x[None], (w.shape[0],) + x.shape)
    if snr_db == "noiseless":
        return xx.copy(), {"noise_var_eff": np.full(xx.shape[:2] + (1,), 1e-9, np.float32)}
    s2 = 10.0 ** (-snr_db / 10.0)
    n = w * np.sqrt(s2)
    if kind == "awgn":
        return xx + n, {"noise_var_eff": np.full(xx.shape[:2] + (1,), s2, np.float32)}
    return (h * xx + n) / h, {"noise_var_eff": (s2 / np.abs(h) ** 2).astype(np.float32)}


def load_models(seed, split, n):
    zd, M16, _, leads = EC.load_all(seed, split)
    Z = zd["z"]
    N = Z.shape[0]
    z1 = Z[torch.arange(N), torch.from_numpy(leads["lead_ctrl"])].numpy()
    M = {}
    sch, z1_2b = M16["rep_orig"]
    M["rep_orig"] = (RepSchemeN(sch.scores, sch.head, n), z1_2b)
    if n == 16:
        M["rep_ctrl"] = (RepSchemeN(M16["rep_ctrl"][0].scores, M16["rep_ctrl"][0].head, 16), z1)
        for k in ("jscc_ctrl", "jscc_e2e"):
            net = JSCCN(32 if k == "jscc_ctrl" else 384, 16)
            net.load_state_dict(M16[k][0].state_dict())
            M[k] = (net.eval(), M16[k][1])
        cfg_dir = os.path.join(C.RA, f"comm/runs/sscc_heads_seed{seed}")
    else:
        rc = RepCtrlN(n)
        rc.load_state_dict(torch.load(os.path.join(HERE, f"runs/rep_ctrl_n{n}_seed{seed}/best.pt")))
        rc.eval()
        M["rep_ctrl"] = (RepSchemeN(rc.ranker.raw_scores().detach(), rc.head, n), z1)
        for k, src in (("jscc_ctrl", z1), ("jscc_e2e", Z.reshape(N, -1).numpy())):
            net = JSCCN(32 if k == "jscc_ctrl" else 384, n)
            net.load_state_dict(torch.load(os.path.join(HERE, f"runs/{k}_n{n}_seed{seed}/best.pt")))
            M[k] = (net.eval(), src)
        cfg_dir = os.path.join(HERE, f"runs/sscc_heads_n{n}_seed{seed}")
    for name in CONFIGS_N[n]:
        st = pickle.load(open(os.path.join(cfg_dir, f"codec_{name}.pkl"), "rb"))
        sc = SSCCN(name, n)
        sc.pq.books, sc.scale = st["books"], st["scale"]
        hd = torch.nn.Linear(32, 5)
        hd.load_state_dict(torch.load(os.path.join(cfg_dir, f"head_{name}.pt")))
        M[f"sscc_{name}"] = (sc, hd.eval(), z1)
    return zd, M, leads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True, choices=(8, 16, 32))
    ap.add_argument("--J", type=int, default=30)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check16", action="store_true")
    a = ap.parse_args()
    assert (a.n == 16) == a.check16, "N = 16 is only run as the reproduction check"
    if a.split == "test":
        assert not a.check16
        C.require_frozen()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    zd, M, leads = load_models(a.seed, a.split, a.n)
    y = zd["y"].numpy()
    N = len(y)
    w, h = noise_bank(a.split, N, a.J, a.n)
    TX, energy = {}, {}
    for r in R_OPTIONS[a.n]:
        for fam in ("rep_orig", "rep_ctrl"):
            sch, z1 = M[fam]
            TX[f"{fam}_r{r}"] = sch.tx(z1, r)
    with torch.no_grad():
        for fam in ("jscc_ctrl", "jscc_e2e"):
            net, src = M[fam]
            x = net.tx(torch.from_numpy(src)).numpy()
            TX[fam] = ((x[:, 0] + 1j * x[:, 1]).astype(np.complex64), None)
    for name in CONFIGS_N[a.n]:
        sc, _, z1 = M[f"sscc_{name}"]
        TX[f"sscc_{name}"] = (sc.tx(z1), None)
    for k, (x, _) in TX.items():
        e = (np.abs(x) ** 2).sum(1)
        assert x.shape == (N, a.n), (k, x.shape)
        energy[k] = {"data_uses": a.n, "pilots": 0, "energy_mean": float(e.mean()), "energy_min": float(e.min()),
                     "energy_max": float(e.max())}
        if k.startswith("sscc_"):
            sc = M[k][0]
            energy[k].update(info_bits=sc.info_bits, coded_bits=sc.coded_bits)
    res_all, worst = {}, 0.0
    for kind in KINDS:
        cond = f"{kind}_perfect_n{a.n}"
        d = os.path.join(C.CACHE, "budget", a.split, f"seed{a.seed}", cond)
        if not a.check16:
            os.makedirs(d, exist_ok=True)
        for snr in SNR_GRID + ["noiseless"]:
            res = {}
            for k, (x, aux) in TX.items():
                xh, info = channel(x, snr, w, h, kind)
                if k.startswith("rep_"):
                    fam, r = k.rsplit("_r", 1)
                    lg = M[fam][0].rx(xh, aux, int(r))
                elif k.startswith("jscc"):
                    with torch.no_grad():
                        lg = M[k][0].rx(from_complex(xh.reshape(-1, a.n))).numpy().reshape(xh.shape[0], N, 5)
                else:
                    sc, hd, _ = M[k]
                    zh = sc.rx(xh, info["noise_var_eff"])
                    with torch.no_grad():
                        lg = hd(torch.from_numpy(zh.astype(np.float32))).numpy()
                if a.check16:
                    old = np.load(os.path.join(C.OLD_CACHE, a.split, f"seed{a.seed}", f"{kind}_perfect", f"{k}__{snr}.npy"))
                    worst = max(worst, float(np.abs(lg.astype(np.float32) - old).max()))
                else:
                    np.save(os.path.join(d, f"{k}__{snr}.npy"), lg.astype(np.float32))
                    res[k] = EC.summarize(y, lg)
            res_all[f"{cond}_{snr}"] = res
            print(cond, snr, f"{time.time() - t0:.0f}s", flush=True)
    if a.check16:
        C.dump({"split": a.split, "seed": a.seed, "worst_abs_dlogit_vs_cache": worst, "pass": worst <= 1e-5},
               os.path.join(C.R2, "results", "checks", f"budget_n16_{a.split}_seed{a.seed}.json"))
        print("N=16 generic path vs cached logits: worst", worst)
        return
    meta = os.path.join(C.CACHE, "budget", a.split, f"seed{a.seed}", "meta.npz")
    np.savez(meta, y=y, patient_id=zd["patient_id"].numpy(), ecg_id=zd["ecg_id"].numpy(), **leads)
    C.dump({"split": a.split, "seed": a.seed, "n": a.n, "J": a.J, "post_hoc_revision_analysis": True,
            "rep_orig": "frozen transfer of the submitted checkpoint", "energy_checks": energy, "conditions": res_all},
           os.path.join(C.R2, "results", "budget", f"budget_eval_{a.split}_n{a.n}_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
