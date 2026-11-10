"""Symbol-level repetition/redundancy allocation: Phase 2B simplified
Formulation C, N = m x r.

Fixed N=16 complex-symbol budget per selected lead (matching the existing
d=32 -> 16-complex-symbol per-lead payload convention), split between m
unique retained semantic symbols and repetition factor r, with
m * r = N exactly for every allowed (m, r) pair. Reuses
PowerNormalization / AWGNChannel / RayleighFadingChannel / RayleighEqualizer
from src.channels.wireless UNMODIFIED, and straight_through_topk_mask from
src.models.lead_selectors UNMODIFIED. See docs/phase2b_design.md.

Receiver CSI is perfect (idealized assumption, see docs/phase2b_design.md
section 4) -- not an oracle-performance claim, just the same simplifying
assumption Phase 1 already uses for RayleighEqualizer.
"""
from __future__ import annotations

import torch

from src.channels.wireless import (
    AWGNChannel,
    PowerNormalization,
    RayleighEqualizer,
    RayleighFadingChannel,
    iq_to_latent,
    latent_to_iq,
)
from src.models.lead_selectors import straight_through_topk_mask

N_SYMBOLS = 16
ALLOWED_R_TO_M = {1: 16, 2: 8, 4: 4}  # r -> m, m * r == N_SYMBOLS


def mr_pairs() -> list[tuple[int, int]]:
    """[(m, r), ...] for all allowed operating points, e.g. [(16,1),(8,2),(4,4)]."""
    return [(m, r) for r, m in ALLOWED_R_TO_M.items()]


def rank_and_select(iq: torch.Tensor, scores: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """iq: [..., 2, 16], scores: [..., 16] (same ranking used for every sample
    when scores is a broadcast of one global vector -- see GlobalSymbolRankSelector).

    Returns (idx [..., m] retained positions in score-descending order,
    retained [..., 2, m] gathered symbols, still carrying the straight-through
    mask's gradient).
    """
    mask = straight_through_topk_mask(scores, m)  # [..., 16], m ones (STE)
    idx = scores.topk(m, dim=-1).indices  # [..., m]
    idx_expand = idx.unsqueeze(-2).expand(*idx.shape[:-1], 2, m)
    retained = torch.gather(iq, dim=-1, index=idx_expand)
    mask_gathered = torch.gather(mask, dim=-1, index=idx)  # [..., m]
    retained = retained * mask_gathered.unsqueeze(-2)
    return idx, retained


def repeat_to_n(retained: torch.Tensor, r: int, n: int = N_SYMBOLS) -> torch.Tensor:
    """retained: [..., 2, m] -> [..., 2, n] by repeating each symbol r times
    consecutively. Requires m * r == n."""
    m = retained.shape[-1]
    if m * r != n:
        raise ValueError(f"m*r must equal N={n}, got m={m} r={r}")
    return retained.repeat_interleave(r, dim=-1)


def combine_repetitions(received: torch.Tensor, m: int, r: int) -> torch.Tensor:
    """received: [..., 2, m*r] (already equalized if Rayleigh) -> [..., 2, m]
    by averaging each symbol's r noisy copies. Degenerate (equal-gain) MRC
    under a shared fading coefficient -- see docs/phase2b_design.md section 4.
    """
    *lead, two, n = received.shape
    if n != m * r:
        raise ValueError(f"received length {n} != m*r ({m}*{r})")
    received = received.reshape(*lead, two, m, r)
    return received.mean(dim=-1)


def scatter_to_full(combined: torch.Tensor, idx: torch.Tensor, n: int = N_SYMBOLS) -> torch.Tensor:
    """combined: [..., 2, m], idx: [..., m] positions in [0,n) -> [..., 2, n],
    zero-filled at every position not in idx (docs/phase2b_design.md section 3,
    "reconstruction": zero-fill, no learned mask token, no decoder).
    """
    *lead, two, m = combined.shape
    out = torch.zeros(*lead, two, n, device=combined.device, dtype=combined.dtype)
    idx_expand = idx.unsqueeze(-2).expand(*idx.shape[:-1], two, m)
    out.scatter_(-1, idx_expand, combined)
    return out


def measure_tx_energy(tx: torch.Tensor) -> tuple[float, float]:
    """tx: [..., 2, N] post-PowerNormalization. Returns (avg_symbol_power,
    total_codeword_energy = avg_symbol_power * N), both should be ~(1.0, N)
    by construction."""
    n = tx.shape[-1]
    avg_power = tx.pow(2).sum(dim=-2).mean().item()
    return avg_power, avg_power * n


def transmit_symbols(z_l: torch.Tensor, scores: torch.Tensor, m: int, r: int, snr_db: float,
                      channel_type: str = "awgn") -> dict:
    """z_l: [..., 32] one selected lead's latent. scores: [16] (global ranking,
    broadcast) or [..., 16] (per-sample, not used by the pilot's global-ranking
    design but supported for generality).

    Returns a dict: z_hat [..., 32], channel_uses (int, always N_SYMBOLS),
    avg_symbol_power (float, ~1.0), total_codeword_energy (float, ~N),
    h (fading coefficient used, or None for AWGN).
    """
    iq = latent_to_iq(z_l)  # [..., 2, 16]
    if scores.dim() == 1:
        scores = scores.expand(*iq.shape[:-2], -1)
    idx, retained = rank_and_select(iq, scores, m)
    repeated = repeat_to_n(retained, r)  # [..., 2, N_SYMBOLS]
    tx = PowerNormalization()(repeated)
    avg_power, total_energy = measure_tx_energy(tx)

    if channel_type == "awgn":
        rx = AWGNChannel(assume_unit_power=True)(tx, snr_db)
        h = None
    elif channel_type == "rayleigh":
        rx, h = RayleighFadingChannel()(tx, snr_db)
        rx = RayleighEqualizer()(rx, h)
    elif channel_type == "none":
        # Bypasses AWGN/fading entirely (post-PowerNormalization identity
        # channel) -- isolates the representation/information-bottleneck
        # effect of a given (m,r) from any channel-noise-averaging benefit.
        # See docs/phase2b_design.md's mechanism diagnostic.
        rx = tx
        h = None
    else:
        raise ValueError(f"Unknown channel_type {channel_type!r}")

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
