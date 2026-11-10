"""Global (population-level, not sample-specific) symbol-within-lead ranking
for Phase 2B simplified Formulation C.

Deliberately not sample-specific -- unlike Phase 2A's lead selector, which
IS sample-specific by design. See docs/phase2b_design.md section 3: lead
identity stays content/sample-driven (Phase 2A's contribution, unchanged);
how much of a selected lead to transmit and how much to repeat is
channel/SNR-driven, and keeping the symbol-rate mechanism global keeps that
separation clean and avoids a second per-sample signaling cost.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.channels.repetition import N_SYMBOLS


class GlobalSymbolRankSelector(nn.Module):
    """One learned score per complex-symbol position (0..15), shared across
    every lead and every patient. Top-m prefixes of this one ranking are
    nested by construction (top-4 subset of top-8 subset of top-16), mirroring
    Phase 2A's S1 subset S3 subset S6 property -- one shared model across all
    three (m, r) operating points, not three separate models.
    """

    def __init__(self, num_symbols: int = N_SYMBOLS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_symbols))

    def raw_scores(self) -> torch.Tensor:
        return self.logits
