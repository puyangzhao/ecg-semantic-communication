"""Multilabel classification head."""
from __future__ import annotations

import torch
import torch.nn as nn


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
        """z: [B, d] -> logits: [B, num_classes] (unnormalized, use with BCEWithLogitsLoss)"""
        return self.net(z)
