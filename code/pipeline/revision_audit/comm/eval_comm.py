#!/usr/bin/env python
"""Unified, CRN-paired evaluation of every K=1 transmission method (16 complex uses / record).

Methods (per seed s in 0,1,2):
  rep_orig_r{1,2,4} : submitted Phase 2B checkpoint (own lead selector, ranker, head)
  rep_ctrl_r{1,2,4} : repetition scheme retrained with the Phase 2A proposed lead (controlled)
  jscc_ctrl         : learned JSCC on the same selected-lead embedding (controlled)
  jscc_e2e          : learned JSCC from all 12 lead embeddings (end-to-end alternative)
  sscc_<cfg>        : 5 conventional quantize -> code -> modulate configurations (controlled lead)
  ref_A_clean, ref_C_clean : Phase 2A proposed and matched fixed-V5 heads, no channel (digital ideal)
Channels: awgn/perfect, rayleigh/perfect, rayleigh/ls2 (2 pilots, +2 uses for every method);
SNR in SNR_GRID + noiseless; J realizations from channel_sim.CRNBank (shared by all methods).
Writes logits [J, N, 5] per condition to /scratch and a summary JSON of realization-mean metrics.

Usage: python revision_audit/comm/eval_comm.py --split val --seed 0 [--J 30]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from channel_sim import CRNBank, SNR_GRID, channel  # noqa: E402
from comm_models import JSCC, RepCtrl, from_complex  # noqa: E402
from scripts.train_phase2a import build_selector  # noqa: E402
from sscc import CONFIGS, SSCC  # noqa: E402
from src.channels.repetition import mr_pairs  # noqa: E402
from src.channels.wireless import iq_to_latent, latent_to_iq  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402
from src.models.lead_selectors import CrossLeadTopKSelector  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from src.models.phase2b_model import Phase2BChannelModel  # noqa: E402
from src.models.symbol_rank_selector import GlobalSymbolRankSelector  # noqa: E402
from train_comm import proposed_leads  # noqa: E402

EMB = os.path.join(ROOT, "outputs/phase2a/embeddings")
CACHE = "cache/comm_eval"
CHANNELS = [("awgn", "perfect"), ("rayleigh", "perfect"), ("rayleigh", "ls2")]
V5 = LEAD_NAMES.index("V5")
R_TO_M = {1: 16, 2: 8, 4: 4}


class RepScheme:
    """numpy re-implementation of src.channels.repetition.transmit_symbols around the CRN channel."""

    def __init__(self, scores, head):
        self.scores, self.head = scores, head

    def tx(self, z1, r):
        m = R_TO_M[r]
        iq = latent_to_iq(torch.from_numpy(z1))  # [N, 2, 16]
        idx = self.scores.topk(m).indices  # global ranking, descending
        kept = iq[..., idx].repeat_interleave(r, dim=-1)
        p = kept.pow(2).sum(-2, keepdim=True).mean(-1, keepdim=True)
        x = (kept * torch.rsqrt(p + 1e-12)).numpy()
        return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64), idx

    @torch.no_grad()
    def rx(self, xh, idx, r):
        m = R_TO_M[r]
        J, N = xh.shape[:2]
        comb = xh.reshape(J, N, m, r).mean(-1)
        full = np.zeros((J, N, 16), np.complex64)
        full[..., idx.numpy()] = comb
        lat = iq_to_latent(from_complex(full))
        return self.head(lat).numpy()


def load_all(seed, split):
    z = torch.load(os.path.join(EMB, f"{split}.pt"))
    Z = z["z"]
    N = Z.shape[0]
    lead = proposed_leads(seed, split)
    z1 = Z[torch.arange(N), torch.from_numpy(lead)].numpy()
    M = {}
    # submitted Phase 2B checkpoint
    m2b = Phase2BChannelModel(CrossLeadTopKSelector(32), GlobalSymbolRankSelector(), latent_dim=32, num_classes=5)
    m2b.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2b_pilot_k1_awgn_seed{seed}/checkpoints/best.pt"),
                                   map_location="cpu")["model_state_dict"])
    m2b.eval()
    with torch.no_grad():
        lead2b = m2b.lead_selector.raw_scores(Z).argmax(1).numpy()
    z1_2b = Z[torch.arange(N), torch.from_numpy(lead2b)].numpy()
    M["rep_orig"] = (RepScheme(m2b.symbol_ranker.raw_scores().detach(), m2b.classifier), z1_2b)
    rc = RepCtrl()
    rc.load_state_dict(torch.load(os.path.join(HERE, f"runs/rep_ctrl_seed{seed}/best.pt")))
    rc.eval()
    M["rep_ctrl"] = (RepScheme(rc.ranker.raw_scores().detach(), rc.head), z1)
    for name, src in (("jscc_ctrl", z1), ("jscc_e2e", Z.reshape(N, -1).numpy())):
        j = JSCC(32 if name == "jscc_ctrl" else 384)
        j.load_state_dict(torch.load(os.path.join(HERE, f"runs/{name}_seed{seed}/best.pt")))
        j.eval()
        M[name] = (j, src)
    for cfg in CONFIGS:
        sc = SSCC(cfg)
        st = pickle.load(open(os.path.join(HERE, f"runs/sscc_heads_seed{seed}/codec_{cfg}.pkl"), "rb"))
        sc.pq.books, sc.scale = st["books"], st["scale"]
        h = torch.nn.Linear(32, 5)
        h.load_state_dict(torch.load(os.path.join(HERE, f"runs/sscc_heads_seed{seed}/head_{cfg}.pt")))
        h.eval()
        M[f"sscc_{cfg}"] = (sc, h, z1)
    # clean references
    refs = {}
    pa = Phase2ASelectionModel(build_selector("proposed", 32), latent_dim=32, num_classes=5)
    pa.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2a_proposed_seed{seed}/checkpoints/best.pt"),
                                  map_location="cpu")["model_state_dict"])
    pc = Phase2ASelectionModel(build_selector("fixed_clinical_k", 32), latent_dim=32, num_classes=5)
    pc.load_state_dict(torch.load(os.path.join(ROOT, f"revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed{seed}/checkpoints/best.pt"),
                                  map_location="cpu")["model_state_dict"])
    with torch.no_grad():
        refs["ref_A_clean"] = pa(Z, 1)[0].numpy()[None]
        v5 = torch.zeros(N, 12)
        v5[:, V5] = 1
        refs["ref_C_clean"] = pc.classifier(pc.fusion(Z, v5)).numpy()[None]
    return z, M, refs, {"lead_ctrl": lead, "lead_rep_orig": lead2b}


def summarize(y, logits):
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    au = [np.mean([roc_auc_score(y[:, c], pj[:, c]) for c in range(5)]) for pj in p]
    ap = [np.mean([average_precision_score(y[:, c], pj[:, c]) for c in range(5)]) for pj in p]
    cd = [average_precision_score(y[:, 3], pj[:, 3]) for pj in p]
    st = [average_precision_score(y[:, 2], pj[:, 2]) for pj in p]
    return {"macro_auroc": float(np.mean(au)), "macro_auroc_sd_real": float(np.std(au)),
            "macro_auprc": float(np.mean(ap)), "CD_auprc": float(np.mean(cd)), "STTC_auprc": float(np.mean(st))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--J", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.split == "test":
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json")), "test only after lock"
    torch.set_num_threads(a.threads)
    t0 = time.time()
    zd, M, refs, leads = load_all(a.seed, a.split)
    y = zd["y"].numpy()
    N = len(y)
    bank = CRNBank(a.split, N, a.J)
    out_dir = os.path.join(CACHE, a.split, f"seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "meta.npz"), y=y, patient_id=zd["patient_id"].numpy(), ecg_id=zd["ecg_id"].numpy(),
             **leads, **refs)
    summary = {"split": a.split, "seed": a.seed, "J": a.J, "N": N, "refs": {k: summarize(y, v) for k, v in refs.items()},
               "energy_checks": {}, "conditions": {}}
    # transmit once per method (transmitter never depends on channel state)
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
    for k, (x, _) in TX.items():
        e = (np.abs(x) ** 2).sum(1)
        assert x.shape == (N, 16), (k, x.shape)
        summary["energy_checks"][k] = {"uses": 16, "energy_mean": float(e.mean()), "energy_min": float(e.min()),
                                       "energy_max": float(e.max())}
    for kind, csi in CHANNELS:
        for snr in SNR_GRID + ["noiseless"]:
            cond = f"{kind}_{csi}_{snr}"
            res = {}
            for k, (x, aux) in TX.items():
                xh, info = channel(x, snr, bank, kind, csi)
                if k.startswith("rep_"):
                    fam, r = k.rsplit("_r", 1)
                    lg = M[fam][0].rx(xh, aux, int(r))
                elif k.startswith("jscc"):
                    with torch.no_grad():
                        net = M[k][0]
                        lg = net.rx(from_complex(xh.reshape(-1, 16))).numpy().reshape(xh.shape[0], N, 5)
                else:
                    sc, h, _ = M[k]
                    zh = sc.rx(xh, info["noise_var_eff"])
                    with torch.no_grad():
                        lg = h(torch.from_numpy(zh.astype(np.float32))).numpy()
                d = os.path.join(out_dir, f"{kind}_{csi}")
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, f"{k}__{snr}.npy"), lg.astype(np.float32))
                res[k] = summarize(y, lg)
            summary["conditions"][cond] = res
            print(cond, f"{time.time() - t0:.0f}s", {k: round(v["macro_auroc"], 4) for k, v in res.items()}, flush=True)
    json.dump(summary, open(os.path.join(HERE, "results", f"comm_eval_{a.split}_seed{a.seed}.json"), "w"), indent=2)
    open(os.path.join(HERE, "results", f"comm_eval_{a.split}_seed{a.seed}.DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
