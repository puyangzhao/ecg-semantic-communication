"""Common channel simulation for every transmission method in the revision.

Conventions (identical for all methods):
  * per record: N = 16 complex payload channel uses; transmit symbols have
    average energy 1 per use (total payload energy 16 per record);
  * SNR (dB) = Es/N0 per complex use; noise n ~ CN(0, 1/SNR_lin) per use;
  * AWGN: y = x + n;
  * block Rayleigh: one h ~ CN(0,1) per (record, realization), constant over
    the whole block (pilots included); receiver equalizes y / h_rx;
  * CSI conditions: "perfect" (h_rx = h, no pilot) and "ls2" (2 known pilot
    symbols of energy 1 each sent before the payload -> 18 channel uses and
    energy 18 per record for EVERY method; h_rx = LS estimate = mean(y_p/x_p);
    estimation error variance = 1/(2 SNR_lin)); the transmitter never knows h,
    only the nominal SNR;
  * common random numbers: noise/fading/pilot-noise draws are generated once
    per (split, record, realization) with unit variance from a fixed seed and
    rescaled per SNR, so every method, every SNR and every r sees the same
    underlying draws.
Complex arrays are numpy complex64 of shape [J, N_rec, 16].
"""
from __future__ import annotations

import numpy as np

N_USES = 16
N_PILOT = 2
SNR_GRID = [-15, -10, -5, 0, 5, 10, 20, 40]  # plus "noiseless"
BANK_SEED = {"val": 20261201, "test": 20261202}


class CRNBank:
    def __init__(self, split: str, n_rec: int, n_real: int = 30):
        rng = np.random.default_rng(BANK_SEED[split])
        c = lambda *s: ((rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)).astype(np.complex64)
        self.w = c(n_real, n_rec, N_USES)         # payload noise, unit variance
        self.h = c(n_real, n_rec, 1)              # block fading coefficient
        self.wp = c(n_real, n_rec, N_PILOT)       # pilot noise, unit variance
        self.n_real, self.n_rec = n_real, n_rec


def channel(x: np.ndarray, snr_db, bank: CRNBank, kind: str, csi: str = "perfect", j=slice(None)):
    """x: [N_rec, 16] complex (avg energy 1 per use) -> (x_hat [J, N_rec, 16] equalized, info dict).

    info["noise_var_eff"]: [J, N_rec, 1] per-symbol noise variance after equalization as the RECEIVER
    believes it (uses h_rx), for LLR computation.
    """
    xx = np.broadcast_to(x[None], (bank.w[j].shape[0],) + x.shape)
    if snr_db == "noiseless":
        return xx.copy(), {"noise_var_eff": np.full(xx.shape[:2] + (1,), 1e-9, np.float32)}
    s2 = 10.0 ** (-snr_db / 10.0)
    n = bank.w[j] * np.sqrt(s2)
    if kind == "awgn":
        return xx + n, {"noise_var_eff": np.full(xx.shape[:2] + (1,), s2, np.float32)}
    h = bank.h[j]
    y = h * xx + n
    if csi == "perfect":
        h_rx = h
    elif csi == "ls2":
        # pilots x_p = 1 (energy 1 each); y_p = h + n_p; LS estimate averages the 2 pilots
        yp = h + bank.wp[j] * np.sqrt(s2)
        h_rx = yp.mean(-1, keepdims=True)
    else:
        raise ValueError(csi)
    x_hat = y / h_rx
    return x_hat, {"noise_var_eff": (s2 / np.abs(h_rx) ** 2).astype(np.float32)}
