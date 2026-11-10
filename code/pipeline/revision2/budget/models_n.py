"""PLAN §4: transmission models at a budget of N complex channel uses (N in {8, 16, 32}).

At N = 16 these classes are numerically the existing comm_models.JSCC / RepCtrl and eval_comm.RepScheme
(same parameter names, so the existing checkpoints load unchanged; checked by eval_n.py --check16).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from comm_models import awgn_torch, from_complex, power_normalize  # noqa: F401
from src.channels.repetition import combine_repetitions, rank_and_select, repeat_to_n, scatter_to_full
from src.channels.wireless import iq_to_latent, latent_to_iq
from src.models.symbol_rank_selector import GlobalSymbolRankSelector

# (m, r) with m * r = N and m <= 16 (the lead latent has 16 complex symbols)
MR = {8: [(8, 1), (4, 2), (2, 4)], 16: [(16, 1), (8, 2), (4, 4)], 32: [(16, 2), (8, 4), (4, 8)]}
R_OPTIONS = {n: [r for _, r in v] for n, v in MR.items()}


def r_to_m(n, r):
    return n // r


class JSCCN(nn.Module):
    """MLP encoder -> 2N reals = N complex symbols (energy N) -> MLP decoder from 2N reals; hidden widths unchanged"""

    def __init__(self, in_dim, n):
        super().__init__()
        self.n = n
        hid = 128 if in_dim == 32 else 256
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, 2 * n))
        self.dec = nn.Sequential(nn.Linear(2 * n, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 5))

    def tx(self, src):
        return power_normalize(self.enc(src).reshape(-1, 2, self.n))

    def rx(self, y):
        return self.dec(y.reshape(y.shape[0], -1))

    def forward(self, src, snr_db):
        return self.rx(awgn_torch(self.tx(src), snr_db))


class RepCtrlN(nn.Module):
    """same-lead repetition: global symbol ranking over the 16 latent symbols, top-m repeated r times, m*r = N"""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.ranker = GlobalSymbolRankSelector()
        self.head = nn.Linear(32, 5)

    def tx(self, z1, m, r):
        iq = latent_to_iq(z1)
        sc = self.ranker.raw_scores().expand(iq.shape[0], -1)
        idx, kept = rank_and_select(iq, sc, m)
        return power_normalize(repeat_to_n(kept, r, self.n)), idx

    def rx(self, y, idx, m, r):
        return self.head(iq_to_latent(scatter_to_full(combine_repetitions(y, m, r), idx)))

    def forward(self, z1, m, r, snr_db):
        x, idx = self.tx(z1, m, r)
        return self.rx(awgn_torch(x, snr_db), idx, m, r)


class RepSchemeN:
    """numpy evaluation path (eval_comm.RepScheme with the budget N as a parameter)"""

    def __init__(self, scores, head, n):
        self.scores, self.head, self.n = scores, head, n

    def tx(self, z1, r):
        m = r_to_m(self.n, r)
        iq = latent_to_iq(torch.from_numpy(z1))
        idx = self.scores.topk(m).indices
        kept = iq[..., idx].repeat_interleave(r, dim=-1)
        p = kept.pow(2).sum(-2, keepdim=True).mean(-1, keepdim=True)
        x = (kept * torch.rsqrt(p + 1e-12)).numpy()
        return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64), idx

    @torch.no_grad()
    def rx(self, xh, idx, r):
        m = r_to_m(self.n, r)
        J, N = xh.shape[:2]
        comb = xh.reshape(J, N, m, r).mean(-1)
        full = np.zeros((J, N, 16), np.complex64)
        full[..., idx.numpy()] = comb
        return self.head(iq_to_latent(from_complex(full))).numpy()
