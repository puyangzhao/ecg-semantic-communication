"""Weighted BCEWithLogitsLoss using training-set class weights.

Per-class positive weight = n_negative / n_positive (computed on the training
split only), passed to `pos_weight` in BCEWithLogitsLoss. This upweights rare
positive classes (e.g. HYP) without touching the negative-class loss term.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(train_labels: np.ndarray, eps: float = 1.0) -> torch.Tensor:
    """train_labels: [N, C] multi-hot array. Returns pos_weight tensor [C]."""
    n_pos = train_labels.sum(axis=0)
    n_neg = train_labels.shape[0] - n_pos
    weight = n_neg / np.clip(n_pos, eps, None)
    return torch.tensor(weight, dtype=torch.float32)


def build_weighted_bce_loss(train_labels: np.ndarray) -> nn.BCEWithLogitsLoss:
    pos_weight = compute_pos_weight(train_labels)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
