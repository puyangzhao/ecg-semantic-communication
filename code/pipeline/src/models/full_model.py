"""Clean-channel baseline model and the interface for the later channel-in-the-loop pipeline.

Phase 1 baseline (this file's `ECGBaselineModel`):
    x [B, 12, T] -> SharedLeadEncoder -> z [B, 12, d] -> fusion -> [B, d] -> classifier -> logits [B, 5]

Designed future pipeline (Phase 2+, NOT implemented here — no Gumbel Top-K, no
channel-in-the-loop training in Phase 1):
    per-lead encoder -> Top-K lead selector -> power normalization -> channel
    -> receiver fusion -> multilabel classifier

The interface is kept compatible today: `SharedLeadEncoder` output [B, L, d] can
already be converted to complex I/Q symbols via `src.channels.wireless.latent_to_iq`
per-lead, passed through `PowerNormalization` + a channel module, converted back
with `iq_to_latent`, and fed into the same `fusion` + `classifier` modules used
here. A lead selector would only need to subset the L dimension before that step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import SharedLeadEncoder
from src.models.fusion import AttentionPoolFusion, MeanPoolFusion
from src.models.classifier import MultilabelClassifier


class ECGBaselineModel(nn.Module):
    def __init__(self, num_leads: int = 12, num_classes: int = 5, latent_dim: int = 32,
                 fusion: str = "mean", base_channels: int = 16, num_blocks: int = 3,
                 classifier_hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.num_leads = num_leads
        self.latent_dim = latent_dim

        self.encoder = SharedLeadEncoder(
            in_channels=1, base_channels=base_channels, latent_dim=latent_dim, num_blocks=num_blocks,
        )
        if fusion == "mean":
            self.fusion = MeanPoolFusion()
        elif fusion == "attention":
            self.fusion = AttentionPoolFusion(latent_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        self.classifier = MultilabelClassifier(
            latent_dim, num_classes, hidden_dim=classifier_hidden_dim, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, lead_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, T] -> logits: [B, num_classes]"""
        z = self.encoder(x)                    # [B, L, d]
        fused = self.fusion(z, lead_mask)       # [B, d]
        return self.classifier(fused)           # [B, num_classes]
