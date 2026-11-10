"""Transmission models compared at a fixed budget of 16 complex channel uses per record (K=1).

RepCtrl   : the submitted nested-representation + repetition scheme
            (GlobalSymbolRankSelector, m*r=16, per-record power normalization,
            equal-gain repetition combining, zero-fill, linear head), but with the
            transmitted lead fixed to the Phase 2A proposed selector's choice
            (controlled comparison; an adapted re-implementation of the original).
JSCCCtrl  : learned JSCC: MLP 32 -> 128 -> 32 real = 16 complex symbols of the SAME
            selected-lead embedding, per-record power normalization to energy 16,
            channel, MLP decoder 32 -> 128 -> 5 logits. Receiver sees only the 16
            equalized complex symbols.
JSCCE2E   : learned JSCC from ALL 12 lead embeddings (384 -> 256 -> 32 real); the
            transmitter has all 12 leads in this problem, so this is the end-to-end
            alternative to lead selection at the same 16-use budget.
SSCC heads: linear head trained on noiseless dequantized embeddings (sscc.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.channels.repetition import N_SYMBOLS, combine_repetitions, repeat_to_n, scatter_to_full
from src.channels.wireless import iq_to_latent, latent_to_iq
from src.models.symbol_rank_selector import GlobalSymbolRankSelector


def power_normalize(x):  # [..., 2, 16] real/imag -> unit average power per record
    p = x.pow(2).sum(-2, keepdim=True).mean(-1, keepdim=True)
    return x * torch.rsqrt(p + 1e-12)


def awgn_torch(x, snr_db):
    s2 = 10.0 ** (-snr_db / 10.0)
    return x + torch.randn_like(x) * (s2 / 2) ** 0.5


def to_complex(x):  # [N, 2, 16] -> complex numpy [N, 16]
    return (x[:, 0, :] + 1j * x[:, 1, :]).detach().cpu().numpy()


def from_complex(y):  # complex numpy [..., 16] -> torch [..., 2, 16]
    import numpy as np
    return torch.from_numpy(np.stack([y.real, y.imag], -2).astype("float32"))


class RepCtrl(nn.Module):
    def __init__(self):
        super().__init__()
        self.ranker = GlobalSymbolRankSelector()
        self.head = nn.Linear(32, 5)

    def tx(self, z1, m, r):
        """z1 [N, 32] -> (x [N, 2, 16] unit power, idx [N, m])"""
        from src.channels.repetition import rank_and_select
        iq = latent_to_iq(z1)
        sc = self.ranker.raw_scores().expand(iq.shape[0], -1)
        idx, kept = rank_and_select(iq, sc, m)
        return power_normalize(repeat_to_n(kept, r)), idx

    def rx(self, y, idx, m, r):
        return self.head(iq_to_latent(scatter_to_full(combine_repetitions(y, m, r), idx)))

    def forward(self, z1, m, r, snr_db):
        x, idx = self.tx(z1, m, r)
        return self.rx(awgn_torch(x, snr_db), idx, m, r)


class JSCC(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 128 if in_dim == 32 else 256), nn.ReLU(),
                                 nn.Linear(128 if in_dim == 32 else 256, 2 * N_SYMBOLS))
        self.dec = nn.Sequential(nn.Linear(2 * N_SYMBOLS, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 5))

    def tx(self, src):
        return power_normalize(self.enc(src).reshape(-1, 2, N_SYMBOLS))

    def rx(self, y):
        return self.dec(y.reshape(y.shape[0], -1))

    def forward(self, src, snr_db):
        return self.rx(awgn_torch(self.tx(src), snr_db))
