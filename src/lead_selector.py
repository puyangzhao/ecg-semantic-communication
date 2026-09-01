"""Sample-specific Top-K lead selection.

`CrossLeadTopKSelector` is the proposed method: it scores each lead from
its own content and its deviation from the cross-lead mean (a redundancy
proxy), not from each lead in isolation, then selects a hard, per-sample
subset of K of the 12 leads. This is the method reported in the
manuscript; the matched-budget baselines it is compared against (random,
fixed-clinical, population-level-learned selection) are not reproduced in
this minimal release.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def straight_through_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """scores: [B, L]. Returns a [B, L] mask with exactly k ones per row.

    The forward value is the hard top-k mask; the backward gradient flows
    as if the mask were sigmoid(scores) (a standard straight-through
    relaxation for discrete subset selection).
    """
    if k < 1 or k > scores.shape[-1]:
        raise ValueError(f"k={k} out of range for {scores.shape[-1]} leads")
    topk_idx = scores.topk(k, dim=-1).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, topk_idx, 1.0)
    soft = torch.sigmoid(scores)
    return hard + soft - soft.detach()


class CrossLeadTopKSelector(nn.Module):
    """Proposed method: sample-specific hard Top-K lead selection, scored
    from each lead's own content plus its deviation from the cross-lead
    mean.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def raw_scores(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, d] -> per-lead scores [B, L]."""
        context = z.mean(dim=1, keepdim=True).expand_as(z)  # [B, L, d]
        feat = torch.cat([z, context, z - context], dim=-1)  # [B, L, 3d]
        return self.net(feat).squeeze(-1)  # [B, L]

    def forward(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """z: [B, L, d], k: number of leads to keep -> mask [B, L] with
        exactly k ones per row."""
        return straight_through_topk_mask(self.raw_scores(z), k)
