#!/usr/bin/env python
"""Revision analysis (PLAN §2), folds 9 and 10: one joint-JSCC checkpoint with the channel bypassed, AWGN with noise off, and AWGN 20 dB.

See PLAN.md (Experiment 2). Per seed s, checkpoint comm/runs/jscc_e2e_seed{s}/best.pt:
  A  : net.rx(net.tx(src))                                         channel bypassed, torch only
  B  : eval_comm path: tx -> complex64 -> channel_sim.channel(x, +inf dB, bank, "awgn") -> from_complex -> rx
       (AWGN branch, noise variance 0; J = 30 copies)
  B' : training path: net.rx(awgn_torch(net.tx(src), +inf dB))
  B0 : cached comm_eval/val/seed{s}/awgn_perfect/jscc_e2e__noiseless.npy (channel() bypass in the existing code)
  C  : channel(x, 20, CRNBank("val", N, 30), "awgn"), must equal the cached jscc_e2e__20.npy
Clean 12-lead reference (separate model): outputs/runs/phase2a_ceiling_k12_seed{s}: Linear(mean of 12 lead embeddings).
Bootstrap: stats.patient_draws(pid, 2000, seed=20261110); per-seed difference, then mean over seeds 0-2.

Usage: python noiseless_check.py --split val ; --split test (after the freeze)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import r2common as R2C  # noqa: E402
RA, ROOT = R2C.RA, R2C.ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(RA, "comm"))
sys.path.insert(0, os.path.join(RA, "common"))
from channel_sim import CRNBank, channel  # noqa: E402
from comm_models import JSCC, awgn_torch, from_complex  # noqa: E402
from comm_report import macro_auprc_batch, macro_auroc_batch  # noqa: E402
from scripts.train_phase2a import build_selector  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from stats import boot_p, patient_draws  # noqa: E402

EMB = os.path.join(ROOT, "outputs/phase2a/embeddings")
OLD_ROOT = "cache/comm_eval"
SEEDS = (0, 1, 2)
J = 30
TOL = 1e-5
G = {}


def metrics(y, lg):
    """lg [N,5] or [J,N,5] -> (macro AUROC, macro AUPRC) per realization"""
    p = expit(np.asarray(lg, np.float64))
    p = p[None] if p.ndim == 2 else p
    return macro_auroc_batch(y, p), macro_auprc_batch(y, p)


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def _worker(idxs):
    y, P = G["y"], G["P"]
    out = []
    for idx in idxs:
        yy = y[idx]
        out.append({k: (macro_auroc_batch(yy, v[:, idx]).mean(), macro_auprc_batch(yy, v[:, idx]).mean()) for k, v in P.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    a = ap.parse_args()
    if a.split == "test":
        R2C.require_frozen()
    OLD = os.path.join(OLD_ROOT, a.split)
    torch.set_num_threads(8)
    d = torch.load(os.path.join(EMB, f"{a.split}.pt"))
    Z, y, pid = d["z"], d["y"].numpy(), d["patient_id"].numpy()
    N = Z.shape[0]
    src = torch.from_numpy(Z.reshape(N, -1).numpy())  # identical to eval_comm.load_all
    bank = CRNBank(a.split, N, J)
    res = {"split": a.split, "N": int(N), "post_hoc": True, "tolerance_max_abs_dlogit": TOL, "seeds": {}}
    P = {}
    for s in SEEDS:
        net = JSCC(384)
        net.load_state_dict(torch.load(os.path.join(RA, f"comm/runs/jscc_e2e_seed{s}/best.pt")))
        net.eval()
        with torch.no_grad():
            xt = net.tx(src)
            lA = net.rx(xt).numpy()
            torch.manual_seed(0)
            lBp = net.rx(awgn_torch(net.tx(src), float("inf"))).numpy()
        x = xt.numpy()
        xc = (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
        e = (np.abs(xc) ** 2).sum(1)
        xh0, info0 = channel(xc, float("inf"), bank, "awgn")
        assert float(info0["noise_var_eff"].max()) == 0.0
        with torch.no_grad():
            lB = net.rx(from_complex(xh0.reshape(-1, 16))).numpy().reshape(J, N, 5)
            xh20, _ = channel(xc, 20, bank, "awgn")
            lC = net.rx(from_complex(xh20.reshape(-1, 16))).numpy().reshape(J, N, 5)
        lB0 = np.load(os.path.join(OLD, f"seed{s}", "awgn_perfect", "jscc_e2e__noiseless.npy"))
        lC_old = np.load(os.path.join(OLD, f"seed{s}", "awgn_perfect", "jscc_e2e__20.npy"))
        # clean 12-lead reference (different model)
        ref = Phase2ASelectionModel(build_selector("ceiling_k12", 32), latent_dim=32, num_classes=5)
        ref.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2a_ceiling_k12_seed{s}/checkpoints/best.pt"),
                                       map_location="cpu")["model_state_dict"])
        ref.eval()
        with torch.no_grad():
            lR = ref(Z, 12)[0].numpy()
            lR2 = ref.classifier(Z.mean(1)).numpy()
        dmax = lambda a, b: float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
        eq = {"A_vs_B_awgn_noise_off": dmax(lA[None], lB), "A_vs_Bprime_torch_awgn_noise_off": dmax(lA, lBp),
              "A_vs_B0_cached_noiseless": dmax(lA[None], lB0), "B_realizations_identical": dmax(lB, lB[:1]),
              "C_vs_cached_20dB": dmax(lC, lC_old), "ref_forward_vs_linear_of_mean": dmax(lR, lR2)}
        if a.split == "test":  # clean 12-lead reference vs the submission's saved fold-10 probabilities
            sv = np.load(os.path.join(ROOT, f"outputs/runs/phase2a_ceiling_k12_seed{s}/test_raw_K12.npz"))
            assert np.array_equal(sv["ecg_id"], d["ecg_id"].numpy())
            eq["ref_prob_vs_saved_submission"] = float(np.abs(sv["y_prob"] - 1 / (1 + np.exp(-lR.astype(np.float64)))).max())
            assert eq["ref_prob_vs_saved_submission"] < 1e-5
        m = {k: metrics(y, v) for k, v in (("A_bypass", lA), ("B_awgn_noise_off", lB), ("Bprime_torch_noise_off", lBp),
                                            ("B0_cached_noiseless", lB0), ("C_awgn_20dB", lC), ("ref_clean_12lead", lR))}
        mm = {k: {"macro_auroc": float(au.mean()), "macro_auprc": float(ap_.mean()),
                  **({"auroc_sd_over_realizations": float(au.std()), "auprc_sd_over_realizations": float(ap_.std()),
                      "auroc_min_max": [float(au.min()), float(au.max())]} if len(au) > 1 else {})}
              for k, (au, ap_) in m.items()}
        a0 = mm["A_bypass"]
        metric_dev = max(abs(mm[k][q] - a0[q]) for k in ("B_awgn_noise_off", "Bprime_torch_noise_off", "B0_cached_noiseless")
                         for q in ("macro_auroc", "macro_auprc"))
        res["seeds"][str(s)] = {"max_abs_dlogit": eq, "metric_max_abs_dev_vs_A": metric_dev,
                                "equivalent": all(eq[k] <= TOL for k in eq) and metric_dev <= 1e-9,
                                "tx_energy_per_record_min_max": [float(e.min()), float(e.max())], "metrics": mm,
                                "params": {"jscc_e2e_encoder": n_params(net.enc), "jscc_e2e_decoder": n_params(net.dec),
                                           "ref_clean_12lead_head": n_params(ref)}}
        P[("A", s)] = expit(lA[None].astype(np.float64))
        P[("C", s)] = expit(lC.astype(np.float64))
        P[("R", s)] = expit(lR[None].astype(np.float64))
        print("seed", s, json.dumps(eq), {k: round(v["macro_auroc"], 4) for k, v in mm.items()}, flush=True)
    # seed means
    res["seed_mean"] = {k: {q: float(np.mean([res["seeds"][str(s)]["metrics"][k][q] for s in SEEDS]))
                            for q in ("macro_auroc", "macro_auprc")} for k in res["seeds"]["0"]["metrics"]}
    # paired patient bootstrap (all 30 realizations for C)
    G.update(y=y, P=P)
    draws = patient_draws(pid, 2000, seed=20261110)
    W = 32
    with Pool(W) as pool:
        parts = pool.map(_worker, [draws[i::W] for i in range(W)])
    boot = [None] * len(draws)
    for i, part in enumerate(parts):
        for j, r in enumerate(part):
            boot[i + j * W] = r
    obs ={k: (macro_auroc_batch(y, v).mean(), macro_auprc_batch(y, v).mean()) for k, v in P.items()}
    res["contrasts"] = {}
    for name, x, z, label in (("C_minus_A", "C", "A", "same checkpoint: effect of 20 dB AWGN noise"),
                              ("A_minus_R", "A", "R", "between-model difference (joint JSCC noise-free vs clean 12-lead mean-pool+linear); NOT a noise effect"),
                              ("C_minus_R", "C", "R", "between-model difference at 20 dB; NOT a noise effect")):
        e = {"meaning": label}
        for i, met in ((0, "auroc"), (1, "auprc")):
            per_seed = [float(obs[(x, s)][i] - obs[(z, s)][i]) for s in SEEDS]
            bd = np.array([np.mean([b[(x, s)][i] - b[(z, s)][i] for s in SEEDS]) for b in boot])
            e[met] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                      "ci95": list(map(float, np.percentile(bd, [2.5, 97.5]))), "p_boot": boot_p(bd)}
        res["contrasts"][name] = e
        print(name, round(e["auroc"]["mean"], 4), [round(q, 4) for q in e["auroc"]["ci95"]],
              round(e["auprc"]["mean"], 4), [round(q, 4) for q in e["auprc"]["ci95"]], flush=True)
    res["all_equivalent"] = all(v["equivalent"] for v in res["seeds"].values())
    R2C.dump(res, os.path.join(R2C.R2, "results", "noiseless", f"noiseless_check_{a.split}.json"))
    print("all A/B/B'/B0 equivalent:", res["all_equivalent"])


if __name__ == "__main__":
    main()
