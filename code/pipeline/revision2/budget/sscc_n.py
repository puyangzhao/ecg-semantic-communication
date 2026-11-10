"""PLAN §4: SSCC at N complex channel uses, built from the existing comm/sscc.py components
(ResidualVQ, Reed-Muller RM(1,5)/RM(2,5) with soft ML decoding, Gray QPSK/16-QAM, the 2-codeword 16-QAM
interleaver) without changing them. Average-power convention as before: QPSK has energy exactly N per record;
16-QAM uses one fixed constellation scale so that the TRAINING-fold mean energy is N (no per-record renormalization).

code = (rm_order or None, number_of_codewords). Multi-codeword 16-QAM: codeword pairs (0,1), (2,3), ... are each
interleaved with the existing 64-bit interleaver and placed on consecutive blocks of 16 symbols.
"""
from __future__ import annotations

import numpy as np

import sscc as S

CONFIGS_N = {
    8: {"S6_RM1_16QAM": dict(n_sub=1, bits_per_sub=6, code=(1, 1), mod="16QAM"),
        "S16_RM2_16QAM": dict(n_sub=2, bits_per_sub=8, code=(2, 1), mod="16QAM"),
        "S16_UNC_QPSK": dict(n_sub=2, bits_per_sub=8, code=(None, 0), mod="QPSK"),
        "S32_UNC_16QAM": dict(n_sub=4, bits_per_sub=8, code=(None, 0), mod="16QAM")},
    16: {"S6_RM1_QPSK": dict(n_sub=1, bits_per_sub=6, code=(1, 1), mod="QPSK"),
         "S16_RM2_QPSK": dict(n_sub=2, bits_per_sub=8, code=(2, 1), mod="QPSK"),
         "S32_RM2_16QAM": dict(n_sub=4, bits_per_sub=8, code=(2, 2), mod="16QAM"),
         "S32_UNC_QPSK": dict(n_sub=4, bits_per_sub=8, code=(None, 0), mod="QPSK"),
         "S64_UNC_16QAM": dict(n_sub=8, bits_per_sub=8, code=(None, 0), mod="16QAM")},
    32: {"S12_RM1x2_QPSK": dict(n_sub=2, bits_per_sub=6, code=(1, 2), mod="QPSK"),
         "S32_RM2x2_QPSK": dict(n_sub=4, bits_per_sub=8, code=(2, 2), mod="QPSK"),
         "S64_RM2x4_16QAM": dict(n_sub=8, bits_per_sub=8, code=(2, 4), mod="16QAM"),
         "S64_UNC_QPSK": dict(n_sub=8, bits_per_sub=8, code=(None, 0), mod="QPSK"),
         "S128_UNC_16QAM": dict(n_sub=16, bits_per_sub=8, code=(None, 0), mod="16QAM")},
}
NOT_APPLICABLE = {8: "RM(1,5)/RM(2,5) + QPSK needs 32 coded bits per codeword but only 16 fit in 8 QPSK symbols"}


def modulate(bits, mod, n, scale=1.0):
    bpd = 1 if mod == "QPSK" else 2
    lab = S.LABELS[mod]
    b = bits.reshape(bits.shape[0], n, 2, bpd)
    lvl_idx = np.zeros(b.shape[:3], int)
    for li, l in enumerate(lab):
        lvl_idx[(b == l).all(-1)] = li
    v = S.LEVELS[mod][lvl_idx] * scale
    return (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)


def llr(y, noise_var, mod, n, scale=1.0):
    bpd = 1 if mod == "QPSK" else 2
    lv, lab = S.LEVELS[mod] * scale, S.LABELS[mod]
    dims = np.stack([y.real, y.imag], -1)
    d2 = (dims[..., None] - lv) ** 2 / (noise_var[..., None, None] / 1.0)
    out = []
    for k in range(bpd):
        m0 = np.where(lab[:, k] == 0)[0]
        m1 = np.where(lab[:, k] == 1)[0]
        out.append(d2[..., m1].min(-1) - d2[..., m0].min(-1))
    return np.stack(out, -1).reshape(*y.shape[:-1], n * 2 * bpd).astype(np.float32)


class SSCCN:
    def __init__(self, name, n, seed=0):
        self.name, self.n, self.cfg = name, n, CONFIGS_N[n][name]
        self.pq = S.ResidualVQ(self.cfg["n_sub"], self.cfg["bits_per_sub"], seed=seed)
        self.scale = 1.0

    @property
    def info_bits(self):
        return self.cfg["n_sub"] * self.cfg["bits_per_sub"]

    @property
    def coded_bits(self):
        return self.n * 2 * (1 if self.cfg["mod"] == "QPSK" else 2)

    def fit(self, z_train):
        self.pq.fit(z_train)
        self.scale = 1.0
        if self.cfg["mod"] == "16QAM":
            e = (np.abs(self.tx(z_train)) ** 2).sum(1).mean()
            self.scale = float(np.sqrt(self.n / e))
        return self

    def _encode_bits(self, bits):
        r, ncw = self.cfg["code"]
        if r is None:
            return bits
        code = S.rm(r)
        cws = [code.encode(bits[:, i * code.k:(i + 1) * code.k]) for i in range(ncw)]
        if self.cfg["mod"] == "16QAM" and ncw >= 2:
            return np.concatenate([np.concatenate(cws[i:i + 2], 1)[:, S._il(64)] for i in range(0, ncw, 2)], 1)
        return np.concatenate(cws, 1)

    def _decode_llr(self, L):
        r, ncw = self.cfg["code"]
        if r is None:
            return (L < 0).astype(np.uint8)
        code = S.rm(r)
        if self.cfg["mod"] == "16QAM" and ncw >= 2:
            Ld = np.empty_like(L)
            for i in range(0, ncw, 2):
                blk = Ld[:, i * 32:(i + 2) * 32]
                blk[:, S._il(64)] = L[:, i * 32:(i + 2) * 32]
            L = Ld
        return np.concatenate([code.decode_ml(L[:, i * 32:(i + 1) * 32]) for i in range(ncw)], 1)

    def tx(self, z):
        bits = self.pq.idx_to_bits(self.pq.encode(z))
        cw = self._encode_bits(bits)
        assert cw.shape[1] == self.coded_bits, (self.name, cw.shape)
        x = modulate(cw, self.cfg["mod"], self.n, self.scale)
        assert x.shape[1] == self.n
        return x

    def rx(self, y, noise_var):
        J, N = y.shape[:2]
        L = llr(y.reshape(J * N, self.n), noise_var.reshape(J * N, 1), self.cfg["mod"], self.n, self.scale)
        bits = self._decode_llr(L)
        return self.pq.decode(self.pq.bits_to_idx(bits)).reshape(J, N, 32)
