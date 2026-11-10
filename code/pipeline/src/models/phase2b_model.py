"""Phase 2B model: Phase 2A's frozen-embedding lead selector, plus the
simplified Formulation C symbol-rate/repetition channel (N=m*r), plus the
existing mask-aware fusion and classifier.

Operates on precomputed Phase 1 embeddings z [B, 12, 32], exactly like
Phase 2A's Phase2ASelectionModel -- the encoder stays frozen and cached.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.channels.repetition import transmit_symbols
from src.models.classifier import MultilabelClassifier
from src.models.fusion import MeanPoolFusion


class Phase2BChannelModel(nn.Module):
    def __init__(self, lead_selector: nn.Module, symbol_ranker: nn.Module,
                 latent_dim: int = 32, num_classes: int = 5,
                 classifier_hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.lead_selector = lead_selector
        self.symbol_ranker = symbol_ranker
        self.fusion = MeanPoolFusion()
        self.classifier = MultilabelClassifier(
            latent_dim, num_classes, hidden_dim=classifier_hidden_dim, dropout=dropout,
        )

    def forward(self, z: torch.Tensor, k: int, m: int, r: int, snr_db: float,
                channel_type: str = "awgn") -> tuple[torch.Tensor, torch.Tensor, dict]:
        """z: [B, L, d]. Returns (logits [B, num_classes], lead_mask [B, L], diagnostics).

        The channel is applied to every one of the L leads (not just the K
        selected ones) purely as a vectorization convenience; the
        differentiable `lead_mask` multiplication below is what actually
        determines which leads' (post-channel) content reaches the
        classifier, and is also what keeps the gradient path back into the
        lead selector's parameters intact (mirrors how MeanPoolFusion's own
        masking already works in Phase 2A). Reported channel-use/energy
        accounting reflects only the K logically-selected leads' budget
        (K*N), not the L*N leads actually run through the channel call.
        """
        b, l, d = z.shape
        lead_mask = self.lead_selector(z, k)  # [B, L]

        scores = self.symbol_ranker.raw_scores()  # [16], global
        z_flat = z.reshape(b * l, d)
        out = transmit_symbols(z_flat, scores, m, r, snr_db, channel_type=channel_type)
        z_hat = out["z_hat"].reshape(b, l, d)
        z_hat = z_hat * lead_mask.unsqueeze(-1)

        fused = self.fusion(z_hat, lead_mask)
        logits = self.classifier(fused)

        diagnostics = {
            "channel_uses_per_lead": out["channel_uses"],
            "avg_symbol_power": out["avg_symbol_power"],
            "total_codeword_energy_per_lead": out["total_codeword_energy"],
            "total_payload_channel_uses": k * out["channel_uses"],
            "total_payload_energy": k * out["total_codeword_energy"],
        }
        return logits, lead_mask, diagnostics
