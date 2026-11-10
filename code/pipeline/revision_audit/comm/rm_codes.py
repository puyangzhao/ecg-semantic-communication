"""Reed-Muller codes RM(r,5) (length 32) with exhaustive soft-decision ML decoding.

RM(1,5): (32, 6, 16)   RM(2,5): (32, 16, 8). Generator rows = monomials of
degree <= r in 5 Boolean variables evaluated at the 32 points. Decoding:
maximum-correlation over all 2^k codewords using bit LLRs (exact ML for
BPSK/QPSK-Gray on AWGN; max-log BICM metric for 16-QAM).
"""
from __future__ import annotations

import itertools

import numpy as np


def rm_generator(r: int, m: int = 5) -> np.ndarray:
    pts = np.array(list(itertools.product([0, 1], repeat=m)))[:, ::-1]  # [32, m]
    rows = []
    for deg in range(r + 1):
        for comb in itertools.combinations(range(m), deg):
            rows.append(np.prod(pts[:, list(comb)], axis=1) if comb else np.ones(len(pts), int))
    return np.array(rows, dtype=np.uint8)  # [k, 32]


class RMCode:
    def __init__(self, r: int):
        self.G = rm_generator(r)
        self.k, self.n = self.G.shape
        msgs = np.array(list(itertools.product([0, 1], repeat=self.k)), dtype=np.uint8)  # [2^k, k]
        self.msgs = msgs
        self.cw = (msgs.astype(np.int64) @ self.G.astype(np.int64)) % 2  # [2^k, n]
        self.cw_pm = (1 - 2 * self.cw).astype(np.float32)  # bit 0 -> +1
        assert self.min_distance() == 2 ** (5 - r)

    def min_distance(self):
        w = self.cw.sum(1)
        return int(w[w > 0].min())

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """bits [..., k] -> codeword bits [..., n]"""
        return (bits.astype(np.int64) @ self.G.astype(np.int64)) % 2

    def decode_ml(self, llr: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """llr [M, n] (positive favours bit 0) -> message bits [M, k] maximizing sum_i llr_i * (1-2c_i)."""
        out = np.empty((llr.shape[0], self.k), np.uint8)
        for s in range(0, llr.shape[0], chunk):
            best = (llr[s:s + chunk].astype(np.float32) @ self.cw_pm.T).argmax(1)
            out[s:s + chunk] = self.msgs[best]
        return out


# ---------------- exact fast ML decoding (coset + fast Hadamard transform) ----------------
def _fht(x):
    """Walsh-Hadamard transform along the last axis (length 32), natural order:
    out[..., a] = sum_i x[..., i] * (-1)^popcount(a & i)."""
    import torch
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x


class FastRM:
    """Exact ML soft decoding of RM(1,5) / RM(2,5) with the same message-bit order as rm_generator:
    [const, x0..x4, (x0x1, x0x2, ..., x3x4)]. RM(2,5) = union of 2^10 cosets of RM(1,5) (quadratic part);
    each coset is searched with one 32-point FHT. Same argmax as exhaustive correlation."""

    def __init__(self, r):
        import torch
        self.r = r
        G = rm_generator(r)
        self.k = G.shape[0]
        self.Q = G[6:] if r == 2 else np.zeros((0, 32), np.uint8)
        nq = self.Q.shape[0]
        qs = np.array(list(itertools.product([0, 1], repeat=nq)), np.uint8) if nq else np.zeros((1, 0), np.uint8)
        self.qs = qs
        leaders = (qs.astype(np.int64) @ self.Q.astype(np.int64)) % 2 if nq else np.zeros((1, 32), np.int64)
        self.signs = torch.from_numpy((1 - 2 * leaders).astype(np.float32))  # [n_cos, 32]

    def decode_ml(self, llr, chunk=2048):
        import torch
        L = torch.from_numpy(np.ascontiguousarray(llr, dtype=np.float32))
        out = np.empty((L.shape[0], self.k), np.uint8)
        a_bits = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.uint8)  # [32, 5]
        for s in range(0, L.shape[0], chunk):
            Lc = L[s:s + chunk]
            T = _fht(Lc[:, None, :] * self.signs[None])          # [c, n_cos, 32]
            absT = T.abs().reshape(Lc.shape[0], -1)
            best = absT.argmax(1)
            cos, a = (best // 32).numpy(), (best % 32).numpy()
            tval = T.reshape(Lc.shape[0], -1).gather(1, best[:, None]).squeeze(1).numpy()
            const = (tval < 0).astype(np.uint8)
            out[s:s + chunk, 0] = const
            out[s:s + chunk, 1:6] = a_bits[a]
            if self.k > 6:
                out[s:s + chunk, 6:] = self.qs[cos]
        return out
