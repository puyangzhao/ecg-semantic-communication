"""Differentiable simulated wireless channel layers for semantic feature transmission.

SNR convention (documented explicitly, used consistently across this module):
    Given a complex baseband signal x with average power P_signal = E[|x|^2],
    and a target SNR_dB, the noise power is:
        P_noise = P_signal / 10^(SNR_dB / 10)
    AWGN is drawn as circularly symmetric complex Gaussian noise n ~ CN(0, P_noise),
    i.e. n = n_real + j*n_imag with n_real, n_imag ~ N(0, P_noise / 2) each, so that
    E[|n|^2] = P_noise.

    All channel inputs are first power-normalized (via PowerNormalization) so that
    E[|x|^2] = 1 per transmitted symbol, unless normalize=False is explicitly passed.
    This makes SNR_dB in every channel call the *actual* linear SNR at the receiver
    input, independent of the semantic feature scale produced by the encoder.

Tensors are real-valued float tensors of shape [..., 2, N] representing N complex
symbols, with index 0 = real part, index 1 = imaginary part along dim=-2. This keeps
everything differentiable in plain PyTorch (no complex-tensor autograd surprises).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _split_ri(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., 0, :], x[..., 1, :]


def latent_to_iq(z: torch.Tensor) -> torch.Tensor:
    """Reinterpret a real latent vector as N/2 complex I/Q symbols.

    z: [..., d] with d even -> [..., 2, d//2], splitting consecutive pairs of
    latent dimensions into (real, imag). This is the interface boundary
    between the (real-valued) encoder/classifier and the (complex-baseband)
    channel modules.
    """
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
    """Normalizes a batch of complex symbol tensors to unit average power.

    Input: x of shape [..., 2, N] (real/imag stacked on dim=-2).
    Output: same shape, with E[|x|^2] == 1 averaged over the last two dims
    (per-sample normalization, i.e. per batch element / per lead).
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E[|x|^2] = E[x_real^2 + x_imag^2]: sum across the real/imag axis
        # (dim=-2, size 2), then average across symbols (dim=-1).
        power = x.pow(2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(power + self.eps)
        return x * scale


class AWGNChannel(nn.Module):
    """Additive white Gaussian noise channel with an explicit SNR (dB) convention.

    See module docstring for the exact SNR definition. Assumes unit-power input
    unless `assume_unit_power=False`, in which case per-sample signal power is
    measured empirically each forward call (still correct, just uses measured
    power instead of assuming 1.0).
    """

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
        # per-component variance is noise_power / 2 so that total complex power is noise_power
        noise_std = torch.sqrt(noise_power / 2.0)
        noise = torch.randn_like(x) * noise_std
        return x + noise


class RayleighFadingChannel(nn.Module):
    """Flat (single-tap, block-constant) Rayleigh fading channel + AWGN.

    Each forward call draws one complex fading coefficient h ~ CN(0, 1) per
    sample (and, if `per_channel=True`, per leading-batch-dim entry, e.g. per
    lead), applies y = h * x + n, and returns both y and h so the receiver can
    perform coherent equalization. This models a slow, flat-fading link
    (channel constant over the whole transmitted feature vector).

    Input x: [..., 2, N]. h is broadcast over the last two dims.
    """

    def __init__(self):
        super().__init__()
        self.awgn = AWGNChannel(assume_unit_power=True)

    def forward(self, x: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = x.shape[:-2]
        h_real = torch.randn(*batch_shape, 1, 1, device=x.device, dtype=x.dtype) * (0.5 ** 0.5)
        h_imag = torch.randn(*batch_shape, 1, 1, device=x.device, dtype=x.dtype) * (0.5 ** 0.5)

        x_real, x_imag = _split_ri(x)
        x_real = x_real.unsqueeze(-2)
        x_imag = x_imag.unsqueeze(-2)

        y_real = h_real * x_real - h_imag * x_imag
        y_imag = h_real * x_imag + h_imag * x_real
        y = torch.cat([y_real, y_imag], dim=-2)

        # noise power is referenced to the *transmit* power (unit, post PowerNormalization),
        # consistent with the AWGNChannel convention above.
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise_std = (noise_power / 2.0) ** 0.5
        y = y + torch.randn_like(y) * noise_std

        h = torch.cat([h_real, h_imag], dim=-2)  # [..., 2, 1]
        return y, h


class RayleighEqualizer(nn.Module):
    """Zero-forcing (coherent) equalizer for the flat Rayleigh channel above.

    Given received y and the (assumed perfectly known, i.e. genie-aided channel
    estimate) fading coefficient h, computes x_hat = y / h via complex division,
    implemented with real arithmetic:
        x_hat = (y * conj(h)) / |h|^2
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y_real, y_imag = _split_ri(y)
        h_real, h_imag = _split_ri(h)  # broadcast shape [..., 1]

        h_pow = h_real.pow(2) + h_imag.pow(2) + self.eps
        x_real = (y_real * h_real + y_imag * h_imag) / h_pow
        x_imag = (y_imag * h_real - y_real * h_imag) / h_pow
        return torch.stack([x_real, x_imag], dim=-2)
