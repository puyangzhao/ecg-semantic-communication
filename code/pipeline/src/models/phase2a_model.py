"""Phase 2A model: a lead-selection policy on top of frozen Phase 1 per-lead
embeddings, feeding the same mask-aware fusion + classifier used in Phase 1.

Operates on precomputed embeddings z [B, 12, d] (see
scripts/precompute_embeddings.py), not raw waveforms — the encoder is
frozen for all of Phase 2A (docs/phase2a_design.md section 7), so its
output is cached once rather than recomputed every batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.classifier import MultilabelClassifier
from src.models.fusion import MeanPoolFusion


class Phase2ASelectionModel(nn.Module):
    def __init__(self, selector: nn.Module, latent_dim: int = 32, num_classes: int = 5,
                 classifier_hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.selector = selector
        self.fusion = MeanPoolFusion()
        self.classifier = MultilabelClassifier(
            latent_dim, num_classes, hidden_dim=classifier_hidden_dim, dropout=dropout,
        )

    def forward(self, z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [B, L, d], k: int -> (logits [B, num_classes], mask [B, L])"""
        mask = self.selector(z, k)
        fused = self.fusion(z, mask)
        logits = self.classifier(fused)
        return logits, mask
