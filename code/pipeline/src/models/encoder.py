"""Lightweight shared per-lead 1D ResNet encoder.

x_l -> shared encoder -> z_l

The same encoder weights are applied independently to each of the 12 leads
(parameter sharing across leads is intentional: it keeps the model small and
lets a later Top-K lead selector reuse the same per-lead representations).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class SharedLeadEncoder(nn.Module):
    """Shared 1D ResNet encoder applied independently to each ECG lead.

    Input:  x of shape [B, L, T]  (L leads, T time samples)
    Output: z of shape [B, L, d]  (per-lead latent embedding)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16, latent_dim: int = 32,
                 num_blocks: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        channels = base_channels
        for i in range(num_blocks):
            stride = 2 if i > 0 else 1
            blocks.append(ResidualBlock1D(channels, stride=stride))
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels, latent_dim)

    def encode_single_lead(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 1, T] -> [N, d]"""
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, T] -> z: [B, L, d]"""
        b, l, t = x.shape
        x = x.reshape(b * l, 1, t)
        z = self.encode_single_lead(x)
        return z.reshape(b, l, self.latent_dim)
