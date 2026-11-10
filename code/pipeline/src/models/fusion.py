"""Lead-embedding fusion strategies.

Both operate on z of shape [B, L, d] (optionally with a lead mask for the
later Top-K lead-selection setting) and return a fused embedding [B, d].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolFusion(nn.Module):
    def forward(self, z: torch.Tensor, lead_mask: torch.Tensor | None = None) -> torch.Tensor:
        if lead_mask is None:
            return z.mean(dim=1)
        mask = lead_mask.unsqueeze(-1).to(z.dtype)  # [B, L, 1]
        summed = (z * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count


class AttentionPoolFusion(nn.Module):
    """Lightweight single-head additive attention pooling over leads."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.score = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor, lead_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(z).squeeze(-1)  # [B, L]
        if lead_mask is not None:
            logits = logits.masked_fill(~lead_mask.bool(), float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (z * weights).sum(dim=1)
