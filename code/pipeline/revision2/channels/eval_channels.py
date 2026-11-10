#!/usr/bin/env python
"""PLAN §3/§3b: frozen K=1 transmission methods under block Rician fading (perfect and LS-2 CSI) and a
time-varying Rician channel (K_R = 3 dB, LS-2, rho in {1, 0.95, 0}).

K_R = Rician K-factor (direct/scattered power ratio); one lead is transmitted throughout.
Reuses eval_comm.load_all / RepScheme / summarize and the transmit and receive steps of eval_comm.main unchanged.

  block:      h = sqrt(K_R/(K_R+1)) e^{j theta} + sqrt(1/(K_R+1)) g      (g = CRNBank.h, theta from THETA_SEED)
  perfect:    16 data uses, x_hat = y / h
  ls2:        uses 1-2 pilots (x_p = 1), uses 3-18 data; h_hat = mean(y_p); x_hat = y / h_hat (true h never used)
  timevar:    same LOS term; scattered part g_t = rho g_{t-1} + sqrt(1-rho^2) e_t over the 18 uses (g_1 = g);
              h_hat from the two pilots (uses 1-2) applied to all 16 data uses.

Usage:
  eval_channels.py --split val --seed 0 --check-k0       # K_R=0 reproduces cached rayleigh_perfect and rayleigh_ls2
  eval_channels.py --split val --seed 0                  # block Rician, perfect + ls2, K_R in {3, 10} dB
  eval_channels.py --split val --seed 0 --timevar        # K_R = 3 dB, ls2, rho in {1, 0.95, 0}
  (--split test only after the freeze)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2common as C  # noqa: E402
import eval_comm as EC  # noqa: E402
from channel_sim import CRNBank, SNR_GRID  # noqa: E402
from comm_models import from_complex  # noqa: E402
from sscc import CONFIGS  # noqa: E402

KR_DB = [3, 10]
TV_KR_DB = 3
RHOS = [1.0, 0.95, 0.0]
TV_METHODS = ["jscc_e2e", "rep_ctrl_r1", "rep_ctrl_r2", "rep_ctrl_r4"]
THETA_SEED = {"val": 20261301, "test": 20261302}
EVOL_SEED = {"val": 20261311, "test": 20261312}


def los_phase(split, J, N):
    return np.random.default_rng(THETA_SEED[split]).uniform(0.0, 2 * np.pi, (J, N, 1))


def los_term(theta, kr):
    return (np.sqrt(kr / (kr + 1)) * np.exp(1j * theta)).astype(np.complex64)


def rician_h(bank, theta, kr):
    return los_term(theta, kr) + np.float32(np.sqrt(1.0 / (kr + 1))) * bank.h


def block(x, snr_db, bank, h, csi):
    """block fading, h [J, N, 1]; identical to channel_sim.channel(kind='rayleigh') with h supplied"""
    xx = np.broadcast_to(x[None], (bank.w.shape[0],) + x.shape)
    if snr_db == "noiseless":
        return xx.copy(), {"noise_var_eff": np.full(xx.shape[:2] + (1,), 1e-9, np.float32)}
    s2 = 10.0 ** (-snr_db / 10.0)
    y = h * xx + bank.w * np.sqrt(s2)
    if csi == "perfect":
        h_rx = h
    elif csi == "ls2":
        yp = h + bank.wp * np.sqrt(s2)  # pilot observations only
        h_rx = yp.mean(-1, keepdims=True)
    else:
        raise ValueError(csi)
    return y / h_rx, {"noise_var_eff": (s2 / np.abs(h_rx) ** 2).astype(np.float32)}


def timevar_h(bank, theta, kr, rho, e):
    """[J, N, 18] channel trajectory: uses 0-1 pilots, 2-17 data"""
    g = [bank.h[..., 0]]
    a, b = np.float32(rho), np.float32(np.sqrt(1.0 - rho ** 2))
    for t in range(1, 18):
        g.append(a * g[-1] + b * e[..., t - 1])
    g = np.stack(g, -1)
    return los_term(theta, kr) + np.float32(np.sqrt(1.0 / (kr + 1))) * g


def timevar_ls2(x, snr_db, bank, ht):
    xx = np.broadcast_to(x[None], (bank.w.shape[0],) + x.shape)
    s2 = 10.0 ** (-snr_db / 10.0)
    y = ht[..., 2:] * xx + bank.w * np.sqrt(s2)
    yp = ht[..., :2] + bank.wp * np.sqrt(s2)
    h_rx = yp.mean(-1, keepdims=True)
    return y / h_rx, {"noise_var_eff": (s2 / np.abs(h_rx) ** 2).astype(np.float32)}


def transmit(M, N):
    """verbatim transmit step of eval_comm.main"""
    TX = {}
    for r in (1, 2, 4):
        for fam in ("rep_orig", "rep_ctrl"):
            sch, z1 = M[fam]
            TX[f"{fam}_r{r}"] = sch.tx(z1, r)
    with torch.no_grad():
        for fam in ("jscc_ctrl", "jscc_e2e"):
            net, src = M[fam]
            x = net.tx(torch.from_numpy(src)).numpy()
            TX[fam] = ((x[:, 0] + 1j * x[:, 1]).astype(np.complex64), None)
    for cfg in CONFIGS:
        sc, h, z1 = M[f"sscc_{cfg}"]
        TX[f"sscc_{cfg}"] = (sc.tx(z1), None)
    energy = {}
    for k, (x, _) in TX.items():
        e = (np.abs(x) ** 2).sum(1)
        assert x.shape == (N, 16), (k, x.shape)
        energy[k] = {"data_uses": 16, "energy_mean": float(e.mean()), "energy_min": float(e.min()), "energy_max": float(e.max())}
    return TX, energy


def receive(M, k, aux, xh, info, N):
    """verbatim receive step of eval_comm.main"""
    if k.startswith("rep_"):
        fam, r = k.rsplit("_r", 1)
        return M[fam][0].rx(xh, aux, int(r))
    if k.startswith("jscc"):
        with torch.no_grad():
            return M[k][0].rx(from_complex(xh.reshape(-1, 16))).numpy().reshape(xh.shape[0], N, 5)
    sc, h, _ = M[k]
    zh = sc.rx(xh, info["noise_var_eff"])
    with torch.no_grad():
        return h(torch.from_numpy(zh.astype(np.float32))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--J", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--check-k0", action="store_true")
    ap.add_argument("--timevar", action="store_true")
    a = ap.parse_args()
    if a.split == "test":
        assert not a.check_k0
        C.require_frozen()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    zd, M, refs, leads = EC.load_all(a.seed, a.split)
    y = zd["y"].numpy()
    N = len(y)
    bank = CRNBank(a.split, N, a.J)
    theta = los_phase(a.split, a.J, N)
    TX, energy = transmit(M, N)
    out_dir = os.path.join(C.CACHE, "channels", a.split, f"seed{a.seed}")

    if a.check_k0:
        h = rician_h(bank, theta, 0.0)
        assert np.array_equal(h, bank.h)
        res, worst = {}, 0.0
        for csi in ("perfect", "ls2"):
            for snr in SNR_GRID + ["noiseless"]:
                for k, (x, aux) in TX.items():
                    xh, info = block(x, snr, bank, h, csi)
                    lg = receive(M, k, aux, xh, info, N)
                    old = np.load(os.path.join(C.OLD_CACHE, a.split, f"seed{a.seed}", f"rayleigh_{csi}", f"{k}__{snr}.npy"))
                    d = float(np.abs(lg.astype(np.float32) - old).max())
                    res[f"{csi}/{k}__{snr}"] = d
                    worst = max(worst, d)
        C.dump({"split": a.split, "seed": a.seed, "worst": worst, "pass": worst <= 1e-5, "max_abs_dlogit": res},
               os.path.join(C.R2, "results", "checks", f"channels_k0_{a.split}_seed{a.seed}.json"))
        print("K_R=0 vs cached rayleigh_perfect + rayleigh_ls2: worst", worst, flush=True)
        return

    summary = {"split": a.split, "seed": a.seed, "J": a.J, "N": N, "post_hoc_revision_analysis": True,
               "energy_checks": energy, "channel_checks": {}, "conditions": {}}
    if a.timevar:
        kr = 10.0 ** (TV_KR_DB / 10.0)
        e = (np.random.default_rng(EVOL_SEED[a.split]).standard_normal((a.J, N, 17, 2)) / np.sqrt(2)).astype(np.float32)
        e = (e[..., 0] + 1j * e[..., 1]).astype(np.complex64)
        hb = rician_h(bank, theta, kr)
        for rho in RHOS:
            ht = timevar_h(bank, theta, kr, rho, e)
            p2 = np.abs(ht) ** 2
            g = ht - los_term(theta, kr)
            corr = np.abs((g[..., 3:] * np.conj(g[..., 2:-1])).mean()) / (np.abs(g[..., 2:-1]) ** 2).mean()
            summary["channel_checks"][f"rho{rho}"] = {"mean_abs_h2_per_use_min_max": [float(p2.mean((0, 1)).min()), float(p2.mean((0, 1)).max())],
                                                       "lag1_corr_scattered": float(corr),
                                                       "rho1_equals_block": bool(np.array_equal(ht, np.broadcast_to(hb, ht.shape))) if rho == 1.0 else None}
            cname = f"rician{TV_KR_DB}dB_ls2_rho{rho}"
            d = os.path.join(out_dir, cname)
            os.makedirs(d, exist_ok=True)
            for snr in [q for q in SNR_GRID]:
                res = {}
                for k in TV_METHODS:
                    x, aux = TX[k]
                    xh, info = timevar_ls2(x, snr, bank, ht)
                    lg = receive(M, k, aux, xh, info, N)
                    np.save(os.path.join(d, f"{k}__{snr}.npy"), lg.astype(np.float32))
                    res[k] = EC.summarize(y, lg)
                summary["conditions"][f"{cname}_{snr}"] = res
                print(cname, snr, f"{time.time() - t0:.0f}s", {k: round(v["macro_auroc"], 4) for k, v in res.items()}, flush=True)
        C.dump(summary, os.path.join(C.R2, "results", "channels", f"timevar_eval_{a.split}_seed{a.seed}.json"))
        return

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "meta.npz"), y=y, patient_id=zd["patient_id"].numpy(), ecg_id=zd["ecg_id"].numpy(), **leads, **refs)
    for kr_db in KR_DB:
        kr = 10.0 ** (kr_db / 10.0)
        h = rician_h(bank, theta, kr)
        p2 = np.abs(h) ** 2
        summary["channel_checks"][f"{kr_db}dB"] = {
            "K_R_linear": kr, "mean_abs_h2": float(p2.mean()), "los_fraction_expected": kr / (kr + 1),
            "mean_los_power": float((np.abs(los_term(theta, kr)) ** 2).mean()),
            "frac_abs_h2_below_0.1": float((p2 < 0.1).mean()), "rayleigh_frac_abs_h2_below_0.1": float((np.abs(bank.h) ** 2 < 0.1).mean())}
        for csi in ("perfect", "ls2"):
            cname = f"rician{kr_db}dB_{csi}"
            d = os.path.join(out_dir, cname)
            os.makedirs(d, exist_ok=True)
            for snr in SNR_GRID + ["noiseless"]:
                res = {}
                for k, (x, aux) in TX.items():
                    xh, info = block(x, snr, bank, h, csi)
                    lg = receive(M, k, aux, xh, info, N)
                    np.save(os.path.join(d, f"{k}__{snr}.npy"), lg.astype(np.float32))
                    res[k] = EC.summarize(y, lg)
                summary["conditions"][f"{cname}_{snr}"] = res
                print(cname, snr, f"{time.time() - t0:.0f}s", flush=True)
    C.dump(summary, os.path.join(C.R2, "results", "channels", f"channels_eval_{a.split}_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
