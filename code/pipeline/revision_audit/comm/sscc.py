"""Conventional separate source-channel coding (SSCC) baseline for one 32-dim lead embedding.

Chain: z (32 real) -> residual vector quantizer (RVQ: a sequence of full-vector
k-means codebooks, each fit on the previous stage's residual of the
training-fold embeddings of the transmitted lead; natural-binary index
labelling; a product quantizer was tried first and discarded because its
16-bit MSE was worse than a single 6-bit full-vector VQ on fold 9) -> bits -> [Reed-Muller RM(1,5) or RM(2,5) | uncoded] -> Gray
QPSK or Gray 16-QAM (unit average symbol energy) -> exactly 16 complex
channel uses -> equalized symbols -> max-log bit LLRs -> soft ML decoding
(exhaustive maximum correlation) or hard decision -> centroid lookup -> the
receiver's classifier (trained on noiseless dequantized training embeddings).
No CRC / no erasure flag: a wrong ML codeword silently yields wrong centroids
(failure behaviour = graceful-or-not, as decoded; no retransmission).

Configurations (all 16 complex uses):
  S6_RM1_QPSK   : RVQ 1 x 6 bits  -> RM(1,5) 32 coded bits -> QPSK
  S16_RM2_QPSK  : RVQ 2 x 8 bits  -> RM(2,5) 32 coded bits -> QPSK
  S32_RM2_16QAM : RVQ 4 x 8 bits  -> 2 x RM(2,5), 64 coded bits -> 16-QAM
  S32_UNC_QPSK  : RVQ 4 x 8 bits  -> uncoded 32 bits -> QPSK
  S64_UNC_16QAM : RVQ 8 x 8 bits  -> uncoded 64 bits -> 16-QAM
Energy: QPSK has exactly 16 per record. 16-QAM energy depends on the coded
bits, so one fixed constellation scale (a constant known to both ends) is set
so that the mean per-record energy over the TRAINING fold equals 16; the
realized per-record energy distribution is reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from rm_codes import RMCode

CONFIGS = {
    "S6_RM1_QPSK": dict(n_sub=1, bits_per_sub=6, code="RM1", mod="QPSK"),
    "S16_RM2_QPSK": dict(n_sub=2, bits_per_sub=8, code="RM2", mod="QPSK"),
    "S32_RM2_16QAM": dict(n_sub=4, bits_per_sub=8, code="RM2x2", mod="16QAM"),
    "S32_UNC_QPSK": dict(n_sub=4, bits_per_sub=8, code="none", mod="QPSK"),
    "S64_UNC_16QAM": dict(n_sub=8, bits_per_sub=8, code="none", mod="16QAM"),
}
_RM = {}


def rm(r):
    if r not in _RM:
        _RM[r] = RMCode(r)
    return _RM[r]


# ---------------- source coding ----------------
class ResidualVQ:
    """n_sub stages, each a full 32-d k-means codebook with 2^bits_per_sub centroids on the residual."""

    def __init__(self, n_sub, bits_per_sub, d=32, seed=0):
        self.n_sub, self.bits, self.d, self.seed = n_sub, bits_per_sub, d, seed
        self.books = []

    @staticmethod
    def _nearest(x, B):
        return ((x ** 2).sum(1, keepdims=True) - 2 * x @ B.T + (B ** 2).sum(1)[None]).argmin(1)

    def fit(self, z):
        self.books, r = [], z.astype(np.float64).copy()
        for s in range(self.n_sub):
            km = KMeans(n_clusters=2 ** self.bits, n_init=2, random_state=self.seed + s, max_iter=200)
            km.fit(r)
            B = km.cluster_centers_
            self.books.append(B.astype(np.float32))
            r = r - B[self._nearest(r, B)]
        return self

    def encode(self, z):
        idx, r = [], z.astype(np.float32).copy()
        for B in self.books:
            i = self._nearest(r, B)
            idx.append(i)
            r = r - B[i]
        return np.stack(idx, 1)

    def decode(self, idx):
        return sum(B[idx[..., s]] for s, B in enumerate(self.books))

    def idx_to_bits(self, idx):
        sh = np.arange(self.bits - 1, -1, -1)
        return ((idx[..., None] >> sh) & 1).reshape(*idx.shape[:-1], -1).astype(np.uint8)

    def bits_to_idx(self, bits):
        b = bits.reshape(*bits.shape[:-1], self.n_sub, self.bits).astype(np.int64)
        return (b << np.arange(self.bits - 1, -1, -1)).sum(-1)

    def state(self):
        return {f"book{s}": B for s, B in enumerate(self.books)}


ProductQuantizer = ResidualVQ  # name kept for the SSCC class below


# ---------------- modulation (Gray, unit average energy) ----------------
LEVELS = {"QPSK": np.array([1.0, -1.0]) / np.sqrt(2),                      # 1 bit / real dim: 0->+, 1->-
          "16QAM": np.array([3.0, 1.0, -1.0, -3.0]) / np.sqrt(10)}         # 2 bits / real dim, Gray: 00,01,11,10
LABELS = {"QPSK": np.array([[0], [1]]), "16QAM": np.array([[0, 0], [0, 1], [1, 1], [1, 0]])}


def modulate(bits, mod, scale=1.0):
    """bits [N, 2*16*bpd] -> complex [N, 16] (times a fixed constellation scale)"""
    bpd = 1 if mod == "QPSK" else 2
    lab = LABELS[mod]
    b = bits.reshape(bits.shape[0], 16, 2, bpd)  # symbol, (I,Q), bits per dim
    lvl_idx = np.zeros(b.shape[:3], int)
    for li, l in enumerate(lab):
        lvl_idx[(b == l).all(-1)] = li
    v = LEVELS[mod][lvl_idx] * scale
    return (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)


def llr(y, noise_var, mod, scale=1.0):
    """y [..., 16] equalized complex, noise_var [..., 1] complex-noise variance -> max-log LLR [..., 16*2*bpd]
    (positive favours bit 0)."""
    bpd = 1 if mod == "QPSK" else 2
    lv, lab = LEVELS[mod] * scale, LABELS[mod]
    dims = np.stack([y.real, y.imag], -1)  # [..., 16, 2]
    d2 = (dims[..., None] - lv) ** 2 / (noise_var[..., None, None] / 1.0)  # per real dim var = nv/2 -> (x-s)^2/(nv)
    out = []
    for k in range(bpd):
        m0 = np.where(lab[:, k] == 0)[0]
        m1 = np.where(lab[:, k] == 1)[0]
        out.append(d2[..., m1].min(-1) - d2[..., m0].min(-1))
    return np.stack(out, -1).reshape(*y.shape[:-1], 16 * 2 * bpd).astype(np.float32)


# ---------------- full transmitter / receiver ----------------
INTERLEAVE_16QAM = None


def _il(n):  # 16-QAM: alternate which codeword gets the reliable (MSB) bit position
    global INTERLEAVE_16QAM
    if INTERLEAVE_16QAM is None:
        a, b = np.arange(32), np.arange(32, 64)
        order = []
        for i in range(16):  # symbol i: (I msb, I lsb, Q msb, Q lsb)
            p, q = (a, b) if i % 2 == 0 else (b, a)
            order += [p[2 * i], q[2 * i], p[2 * i + 1], q[2 * i + 1]]
        INTERLEAVE_16QAM = np.array(order)
    return INTERLEAVE_16QAM


class SSCC:
    def __init__(self, name, seed=0):
        self.name, self.cfg = name, CONFIGS[name]
        self.pq = ProductQuantizer(self.cfg["n_sub"], self.cfg["bits_per_sub"], seed=seed)
        self.scale = 1.0

    def fit(self, z_train):
        """Fit the quantizer and (16-QAM) the fixed constellation scale on the training fold."""
        self.pq.fit(z_train)
        self.scale = 1.0
        if self.cfg["mod"] == "16QAM":
            e = (np.abs(self.tx(z_train)) ** 2).sum(1).mean()
            self.scale = float(np.sqrt(16.0 / e))
        return self

    @property
    def info_bits(self):
        return self.cfg["n_sub"] * self.cfg["bits_per_sub"]

    @property
    def coded_bits(self):
        return 32 if self.cfg["mod"] == "QPSK" else 64

    def tx(self, z):
        bits = self.pq.idx_to_bits(self.pq.encode(z))
        c = self.cfg["code"]
        if c == "RM1" or c == "RM2":
            cw = rm(int(c[-1])).encode(bits)
        elif c == "RM2x2":
            cw = np.concatenate([rm(2).encode(bits[:, :16]), rm(2).encode(bits[:, 16:])], 1)[:, _il(64)]
        else:
            cw = bits
        assert cw.shape[1] == self.coded_bits
        x = modulate(cw, self.cfg["mod"], self.scale)
        assert x.shape[1] == 16
        return x

    def rx(self, y, noise_var):
        """y [J, N, 16] -> dequantized z_hat [J, N, 32]"""
        J, N = y.shape[:2]
        L = llr(y.reshape(J * N, 16), noise_var.reshape(J * N, 1), self.cfg["mod"], self.scale)
        c = self.cfg["code"]
        if c in ("RM1", "RM2"):
            bits = rm(int(c[-1])).decode_ml(L)
        elif c == "RM2x2":
            Ld = np.empty_like(L)
            Ld[:, _il(64)] = L
            bits = np.concatenate([rm(2).decode_ml(Ld[:, :32]), rm(2).decode_ml(Ld[:, 32:])], 1)
        else:
            bits = (L < 0).astype(np.uint8)
        return self.pq.decode(self.pq.bits_to_idx(bits)).reshape(J, N, 32)
