"""Simulated wireless channel + fixed-budget symbol repetition allocation.

SNR convention: given a complex baseband signal x with average power
P_signal = E[|x|^2] and a target SNR_dB, the noise power is
P_noise = P_signal / 10^(SNR_dB/10). AWGN is circularly symmetric complex
Gaussian noise. All channel inputs are first power-normalized so that
E[|x|^2] = 1 per transmitted symbol, so SNR_dB is the actual linear SNR at
the receiver input.

Tensors representing N complex symbols are real-valued float tensors of
shape [..., 2, N] (index 0 = real part, index 1 = imaginary part along
dim=-2), which keeps everything differentiable in plain PyTorch.

Fixed budget: each selected lead transmits exactly N_SYMBOLS=16 complex
channel uses, split between m unique semantic symbols and a repetition
factor r, with m * r = 16 exactly for every allowed r in {1, 2, 4}
(ALLOWED_R_TO_M). Lower SNR favors more repetition (larger r, smaller m);
higher SNR favors more unique content (smaller r, larger m).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.lead_selector import straight_through_topk_mask

N_SYMBOLS = 16
ALLOWED_R_TO_M = {1: 16, 2: 8, 4: 4}  # r -> m, m * r == N_SYMBOLS


def mr_pairs() -> list[tuple[int, int]]:
    """[(m, r), ...] for all allowed operating points, e.g. [(16,1),(8,2),(4,4)]."""
    return [(m, r) for r, m in ALLOWED_R_TO_M.items()]


def latent_to_iq(z: torch.Tensor) -> torch.Tensor:
    """Reinterpret a real latent vector as N/2 complex I/Q symbols.
    z: [..., d] with d even -> [..., 2, d//2]."""
    if z.shape[-1] % 2 != 0:
        raise ValueError(f"latent_to_iq requires an even last dimension, got {z.shape[-1]}")
    *lead, d = z.shape
    z = z.reshape(*lead, d // 2, 2)
    return z.transpose(-1, -2)  # [..., 2, d//2]


def iq_to_latent(x: torch.Tensor) -> torch.Tensor:
    """Inverse of latent_to_iq: [..., 2, N] -> [..., 2N]."""
    x = x.transpose(-1, -2)  # [..., N, 2]
    *lead, n, two = x.shape
    return x.reshape(*lead, n * two)


class PowerNormalization(nn.Module):
    """Normalizes a batch of complex symbol tensors to unit average power."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power = x.pow(2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(power + self.eps)
        return x * scale


class AWGNChannel(nn.Module):
    """Additive white Gaussian noise channel with an explicit SNR (dB) convention."""

    def __init__(self, assume_unit_power: bool = True):
        super().__init__()
        self.assume_unit_power = assume_unit_power

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        if self.assume_unit_power:
            signal_power = torch.ones((), device=x.device, dtype=x.dtype)
        else:
            signal_power = x.pow(2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power / 2.0)
        noise = torch.randn_like(x) * noise_std
        return x + noise


class RayleighFadingChannel(nn.Module):
    """Flat (single-tap, block-constant) Rayleigh fading channel + AWGN.

    Draws one complex fading coefficient h ~ CN(0,1) per sample (and per
    leading-batch-dim entry, e.g. per lead), applies y = h*x + n, and
    returns both y and h so the receiver can perform coherent
    equalization.
    """

    def forward(self, x: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = x.shape[:-2]
        h_real = torch.randn(*batch_shape, 1, 1, device=x.device, dtype=x.dtype) * (0.5 ** 0.5)
        h_imag = torch.randn(*batch_shape, 1, 1, device=x.device, dtype=x.dtype) * (0.5 ** 0.5)

        x_real, x_imag = x[..., 0, :], x[..., 1, :]
        x_real = x_real.unsqueeze(-2)
        x_imag = x_imag.unsqueeze(-2)

        y_real = h_real * x_real - h_imag * x_imag
        y_imag = h_real * x_imag + h_imag * x_real
        y = torch.cat([y_real, y_imag], dim=-2)

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise_std = (noise_power / 2.0) ** 0.5
        y = y + torch.randn_like(y) * noise_std

        h = torch.cat([h_real, h_imag], dim=-2)  # [..., 2, 1]
        return y, h


class RayleighEqualizer(nn.Module):
    """Zero-forcing (coherent) equalizer for the flat Rayleigh channel above:
    x_hat = (y * conj(h)) / |h|^2, with perfect receiver channel-state info."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_real, y_imag = y[..., 0, :], y[..., 1, :]
        h_real, h_imag = h[..., 0, :], h[..., 1, :]

        h_pow = h_real.pow(2) + h_imag.pow(2) + self.eps
        x_real = (y_real * h_real + y_imag * h_imag) / h_pow
        x_imag = (y_imag * h_real - y_real * h_imag) / h_pow
        return torch.stack([x_real, x_imag], dim=-2)


class GlobalSymbolRanker(nn.Module):
    """One learned score per complex-symbol position (0..N_SYMBOLS-1),
    shared across every lead and every sample: decides which m of the 16
    symbol slots survive at a given (m, r) operating point. Top-m prefixes
    are nested by construction (top-4 subset of top-8 subset of top-16)."""

    def __init__(self, num_symbols: int = N_SYMBOLS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_symbols))

    def forward(self) -> torch.Tensor:
        return self.logits


def rank_and_select(iq: torch.Tensor, scores: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """iq: [..., 2, 16], scores: [..., 16] -> (idx [..., m], retained [..., 2, m])."""
    mask = straight_through_topk_mask(scores, m)
    idx = scores.topk(m, dim=-1).indices
    idx_expand = idx.unsqueeze(-2).expand(*idx.shape[:-1], 2, m)
    retained = torch.gather(iq, dim=-1, index=idx_expand)
    mask_gathered = torch.gather(mask, dim=-1, index=idx)
    retained = retained * mask_gathered.unsqueeze(-2)
    return idx, retained


def repeat_to_n(retained: torch.Tensor, r: int, n: int = N_SYMBOLS) -> torch.Tensor:
    """retained: [..., 2, m] -> [..., 2, n], each symbol repeated r times consecutively."""
    m = retained.shape[-1]
    if m * r != n:
        raise ValueError(f"m*r must equal N={n}, got m={m} r={r}")
    return retained.repeat_interleave(r, dim=-1)


def combine_repetitions(received: torch.Tensor, m: int, r: int) -> torch.Tensor:
    """received: [..., 2, m*r] (already equalized if Rayleigh) -> [..., 2, m]
    by averaging each symbol's r noisy copies (equal-gain combining)."""
    *lead, two, n = received.shape
    if n != m * r:
        raise ValueError(f"received length {n} != m*r ({m}*{r})")
    received = received.reshape(*lead, two, m, r)
    return received.mean(dim=-1)


def scatter_to_full(combined: torch.Tensor, idx: torch.Tensor, n: int = N_SYMBOLS) -> torch.Tensor:
    """combined: [..., 2, m], idx: [..., m] -> [..., 2, n], zero-filled at
    every untransmitted symbol position (no learned mask token, no decoder)."""
    *lead, two, m = combined.shape
    out = torch.zeros(*lead, two, n, device=combined.device, dtype=combined.dtype)
    idx_expand = idx.unsqueeze(-2).expand(*idx.shape[:-1], two, m)
    out.scatter_(-1, idx_expand, combined)
    return out


def transmit_symbols(z_l: torch.Tensor, scores: torch.Tensor, m: int, r: int, snr_db: float,
                      channel_type: str = "awgn") -> dict:
    """z_l: [..., 32] one selected lead's latent. scores: [16], the global
    symbol ranking (broadcast to every sample/lead).

    Returns a dict: z_hat [..., 32] (reconstructed latent, zero-filled at
    untransmitted positions), channel_uses (int, always N_SYMBOLS),
    avg_symbol_power (~1.0), total_codeword_energy (~N_SYMBOLS), h (the
    fading coefficient used, or None for AWGN).
    """
    iq = latent_to_iq(z_l)  # [..., 2, 16]
    if scores.dim() == 1:
        scores = scores.expand(*iq.shape[:-2], -1)
    idx, retained = rank_and_select(iq, scores, m)
    repeated = repeat_to_n(retained, r)  # [..., 2, N_SYMBOLS]
    tx = PowerNormalization()(repeated)
    avg_power = tx.pow(2).sum(dim=-2).mean().item()
    total_energy = avg_power * tx.shape[-1]

    if channel_type == "awgn":
        rx = AWGNChannel(assume_unit_power=True)(tx, snr_db)
        h = None
    elif channel_type == "rayleigh":
        rx, h = RayleighFadingChannel()(tx, snr_db)
        rx = RayleighEqualizer()(rx, h)
    else:
        raise ValueError(f"Unknown channel_type {channel_type!r}, expected 'awgn' or 'rayleigh'")

    combined = combine_repetitions(rx, m, r)
    z_l_hat_iq = scatter_to_full(combined, idx)
    z_l_hat = iq_to_latent(z_l_hat_iq)

    return {
        "z_hat": z_l_hat,
        "channel_uses": tx.shape[-1],
        "avg_symbol_power": avg_power,
        "total_codeword_energy": total_energy,
        "h": h,
    }
