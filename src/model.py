"""End-to-end assembly: shared encoder -> personalized Top-K lead
selection -> fixed-budget channel transmission -> mean-pool fusion over
selected leads -> multilabel classification head.

This mirrors the manuscript's K=1 pipeline (Fig. "System pipeline"): a
frozen shared per-lead encoder produces a 12x32 embedding per record; a
budget-aware lead selector picks a sample-specific K-of-12 subset; each
selected lead's 32-d latent is split into 16 complex symbols, of which m
are transmitted (each repeated r times, m*r=16) over a simulated AWGN or
block-Rayleigh channel; the receiver equalizes/combines repetitions and
zero-fills untransmitted positions; the resulting per-lead latents are
mean-pooled over the selected leads and passed to a shared multilabel
classifier head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.channel import transmit_symbols
from src.encoder import SharedLeadEncoder
from src.lead_selector import CrossLeadTopKSelector


class MeanPoolFusion(nn.Module):
    """Mask-aware mean pooling over leads: z [B, L, d], lead_mask [B, L] -> [B, d]."""

    def forward(self, z: torch.Tensor, lead_mask: torch.Tensor | None = None) -> torch.Tensor:
        if lead_mask is None:
            return z.mean(dim=1)
        mask = lead_mask.unsqueeze(-1).to(z.dtype)
        summed = (z * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count


class MultilabelClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(latent_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, d] -> logits: [B, num_classes] (unnormalized; use sigmoid for probabilities)."""
        return self.net(z)


class TaskAwareECGModel(nn.Module):
    """Full pipeline: raw 12-lead ECG -> shared encoder -> personalized
    Top-K lead selection -> fixed-budget channel transmission -> fusion ->
    multilabel classifier.

    Note: the global per-symbol ranking used inside the channel step
    (`self.symbol_scores`) is a plain learned parameter here, exactly as
    in the manuscript's implementation -- it is untrained in this minimal
    release (random init), so transmitted probabilities are illustrative
    only, not calibrated diagnostic output. See README.md.
    """

    def __init__(self, num_leads: int = 12, signal_length: int = 1000, base_channels: int = 16,
                 latent_dim: int = 32, num_blocks: int = 3, num_classes: int = 5,
                 classifier_hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.encoder = SharedLeadEncoder(
            base_channels=base_channels, latent_dim=latent_dim, num_blocks=num_blocks,
        )
        self.lead_selector = CrossLeadTopKSelector(latent_dim=latent_dim)
        self.symbol_scores = nn.Parameter(torch.zeros(16))  # global per-symbol ranking, see docstring
        self.fusion = MeanPoolFusion()
        self.classifier = MultilabelClassifier(
            latent_dim, num_classes, hidden_dim=classifier_hidden_dim, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, k: int, m: int, r: int, snr_db: float,
                channel_type: str = "awgn") -> tuple[torch.Tensor, torch.Tensor, dict]:
        """x: [B, L, T] raw multi-lead ECG. k: number of leads to transmit.
        m, r: symbol-budget split (m*r must equal 16). snr_db: channel SNR.
        channel_type: 'awgn' or 'rayleigh'.

        Returns (probs [B, num_classes], lead_mask [B, L], diagnostics).
        """
        b, l, t = x.shape
        z = self.encoder(x)  # [B, L, d]
        lead_mask = self.lead_selector(z, k)  # [B, L], exactly k ones per row

        z_flat = z.reshape(b * l, -1)
        out = transmit_symbols(z_flat, self.symbol_scores, m, r, snr_db, channel_type=channel_type)
        z_hat = out["z_hat"].reshape(b, l, -1)
        z_hat = z_hat * lead_mask.unsqueeze(-1)  # zero out non-selected leads

        fused = self.fusion(z_hat, lead_mask)
        logits = self.classifier(fused)
        probs = torch.sigmoid(logits)

        diagnostics = {
            "channel_uses_per_lead": out["channel_uses"],
            "avg_symbol_power": out["avg_symbol_power"],
            "total_codeword_energy_per_lead": out["total_codeword_energy"],
            "total_payload_channel_uses": k * out["channel_uses"],
            "total_payload_energy": k * out["total_codeword_energy"],
        }
        return probs, lead_mask, diagnostics
