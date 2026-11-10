"""Phase 2A lead-selection policies.

All four selectors share one interface: `forward(z, k) -> mask` where
`z` is `[B, L, d]` per-lead embeddings (L=12) and `mask` is `[B, L]` with
exactly `k` ones per row (`mask.sum(dim=-1) == k` is a hard invariant,
unit-tested). Learned selectors (`GlobalLearnedKSelector`,
`CrossLeadTopKSelector`) additionally expose `.raw_scores(z)` for
diagnostics/interpretability, and route the hard top-K mask through a
straight-through estimator so gradients flow.

See docs/phase2a_design.md sections 1-3 for the rationale behind exactly
these four methods and this scorer architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.clinical_lead_subsets import NUM_LEADS, fixed_clinical_indices


def straight_through_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """scores: [B, L]. Returns a [B, L] mask with exactly k ones per row.

    Forward value is the hard top-k mask; backward gradient flows as if the
    mask were sigmoid(scores) (a standard straight-through relaxation for
    discrete subset selection, lightweight relative to e.g. Gumbel-Sinkhorn).
    """
    if k < 1 or k > scores.shape[-1]:
        raise ValueError(f"k={k} out of range for {scores.shape[-1]} leads")
    topk_idx = scores.topk(k, dim=-1).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, topk_idx, 1.0)
    soft = torch.sigmoid(scores)
    return hard + soft - soft.detach()


class RandomKSelector(nn.Module):
    """Uniform random K-subset per sample. No parameters, no gradient."""

    def forward(self, z: torch.Tensor, k: int) -> torch.Tensor:
        b, l, _ = z.shape
        scores = torch.rand(b, l, device=z.device)
        idx = scores.topk(k, dim=-1).indices
        mask = torch.zeros(b, l, device=z.device)
        mask.scatter_(-1, idx, 1.0)
        return mask


class FixedClinicalKSelector(nn.Module):
    """Same prespecified subset (docs/phase2a_design.md section 6) for every patient."""

    def forward(self, z: torch.Tensor, k: int) -> torch.Tensor:
        b, l, _ = z.shape
        idx = fixed_clinical_indices(k)
        mask = torch.zeros(b, l, device=z.device)
        mask[:, idx] = 1.0
        return mask


class GlobalLearnedKSelector(nn.Module):
    """One learned score per lead, shared across all samples (not sample-specific)."""

    def __init__(self, num_leads: int = NUM_LEADS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_leads))

    def raw_scores(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        return self.logits.unsqueeze(0).expand(b, -1)

    def forward(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return straight_through_topk_mask(self.raw_scores(z), k)


class CrossLeadTopKSelector(nn.Module):
    """Proposed method: sample-specific hard Top-K, scored from lead content
    and each lead's deviation from the cross-lead mean (a redundancy proxy),
    not from each lead in isolation. See docs/phase2a_design.md section 3.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def raw_scores(self, z: torch.Tensor) -> torch.Tensor:
        context = z.mean(dim=1, keepdim=True).expand_as(z)  # [B, L, d]
        feat = torch.cat([z, context, z - context], dim=-1)  # [B, L, 3d]
        return self.net(feat).squeeze(-1)  # [B, L]

    def forward(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return straight_through_topk_mask(self.raw_scores(z), k)


class FullLeadSelector(nn.Module):
    """K=12 ceiling reference: no selection, all leads always used."""

    def forward(self, z: torch.Tensor, k: int | None = None) -> torch.Tensor:
        b, l, _ = z.shape
        if k is not None and k != l:
            raise ValueError(f"FullLeadSelector only supports k={l}, got k={k}")
        return torch.ones(b, l, device=z.device)


SELECTOR_REGISTRY = {
    "random_k": RandomKSelector,
    "fixed_clinical_k": FixedClinicalKSelector,
    "global_learned_k": GlobalLearnedKSelector,
    "proposed": lambda latent_dim: CrossLeadTopKSelector(latent_dim),
    "ceiling_k12": FullLeadSelector,
}
