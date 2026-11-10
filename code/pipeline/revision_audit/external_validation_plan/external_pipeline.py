#!/usr/bin/env python
"""Frozen raw-waveform -> K=1 prediction pipeline for a future external cohort.

Mirrors the PTB-XL training-time chain exactly where it is defined, and makes
every external-only step explicit and prespecified:

  training chain (src/data/dataset.py, scripts/precompute_embeddings.py):
    wfdb.rdsamp physical signal (mV), header lead order I..V6, 100 Hz,
    exactly 1000 samples (10 s) -> per-lead z-score with outputs/norm_stats.npz
    (fit on PTB-XL folds 1-8) -> frozen SharedLeadEncoder from
    outputs/runs/baseline_run1/checkpoints/best.pt -> z [12, 32]

  external-only steps (prespecified here, not tuned on external data):
    1. reorder leads by case-insensitive name to I,II,III,aVR,aVL,aVF,V1..V6;
       refuse if any lead is missing or duplicated
    2. convert physical units to mV (mV / uV / V accepted; anything else refused)
    3. resample to 100 Hz with scipy.signal.resample_poly (polyphase FIR,
       Kaiser window); PTB-XL's own 500->100 Hz filter is undocumented, so a
       residual filtering mismatch remains (stated as a limitation)
    4. keep the FIRST 10 s (1000 samples); refuse records shorter than 10 s
    5. refuse records with NaN/inf after steps 1-4

Then the three frozen K=1 predictions per seed:
    A = Phase 2A proposed head + proposed selector's lead
    B = same proposed head + V5 forced
    C = matched fixed-V5 head + V5

This module never reads labels.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import torch
import yaml
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.train_phase2a import build_selector
from src.data.clinical_lead_subsets import LEAD_NAMES
from src.data.dataset import NormalizationStats
from src.models.full_model import ECGBaselineModel
from src.models.phase2a_model import Phase2ASelectionModel

TARGET_FS = 100
TARGET_LEN = 1000
UNIT_TO_MV = {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "v": 1e3}
ENCODER_CKPT = "outputs/runs/baseline_run1/checkpoints/best.pt"
BASELINE_CONFIG = "configs/baseline.yaml"
NORM_STATS = "outputs/norm_stats.npz"
PROPOSED_CKPT = "outputs/runs/phase2a_proposed_seed{s}/checkpoints/best.pt"
FIXED_V5_CKPT = "revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed{s}/checkpoints/best.pt"
V5 = LEAD_NAMES.index("V5")


def to_training_format(signal: np.ndarray, sig_names: list[str], units: list[str], fs: float) -> np.ndarray:
    """signal: [T, n_leads] physical values as returned by wfdb.rdsamp.
    Returns [12, 1000] float32 in mV, PTB-XL lead order, 100 Hz, first 10 s
    (not yet z-scored)."""
    lookup = {n.lower(): i for i, n in enumerate(sig_names)}
    if len(lookup) != len(sig_names):
        raise ValueError(f"duplicate lead names (case-insensitive): {sig_names}")
    missing = [n for n in LEAD_NAMES if n.lower() not in lookup]
    if missing:
        raise ValueError(f"missing leads {missing}; have {sig_names}")
    cols = [lookup[n.lower()] for n in LEAD_NAMES]
    scale = []
    for c in cols:
        u = units[c].strip().lower()
        if u not in UNIT_TO_MV:
            raise ValueError(f"unsupported unit {units[c]!r} for lead {sig_names[c]}")
        scale.append(UNIT_TO_MV[u])
    x = signal[:, cols].astype(np.float64) * np.array(scale)[None, :]  # [T, 12] mV
    ratio = Fraction(TARGET_FS, 1) / Fraction(fs).limit_denominator(10_000)
    if ratio != 1:
        x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
    if x.shape[0] < TARGET_LEN:
        raise ValueError(f"record shorter than 10 s after resampling ({x.shape[0]} samples)")
    x = x[:TARGET_LEN]
    if not np.isfinite(x).all():
        raise ValueError("non-finite samples")
    return x.T.astype(np.float32)


class FrozenK1Pipeline:
    def __init__(self, device: str, seeds=(0, 1, 2)):
        self.device = device
        cfg = yaml.safe_load(open(os.path.join(ROOT, BASELINE_CONFIG)))
        base = ECGBaselineModel(num_leads=12, num_classes=5, latent_dim=cfg["model"]["latent_dim"],
                                fusion=cfg["model"]["fusion"], base_channels=cfg["model"]["base_channels"],
                                num_blocks=cfg["model"]["num_blocks"],
                                classifier_hidden_dim=cfg["model"]["classifier_hidden_dim"],
                                dropout=cfg["model"]["dropout"])
        base.load_state_dict(torch.load(os.path.join(ROOT, ENCODER_CKPT), map_location=device)["model_state_dict"])
        self.encoder = base.encoder.to(device).eval()
        ns = NormalizationStats.load(os.path.join(ROOT, NORM_STATS))
        self.mean = torch.from_numpy(ns.mean).view(12, 1)
        self.std = torch.from_numpy(ns.std).view(12, 1)
        self.heads = {}
        for s in seeds:
            self.heads[s] = {"proposed": self._load(PROPOSED_CKPT.format(s=s), "proposed"),
                             "fixed_V5": self._load(FIXED_V5_CKPT.format(s=s), "fixed_clinical_k")}

    def _load(self, rel, method):
        m = Phase2ASelectionModel(build_selector(method, 32), latent_dim=32, num_classes=5).to(self.device)
        m.load_state_dict(torch.load(os.path.join(ROOT, rel), map_location=self.device)["model_state_dict"])
        return m.eval()

    @torch.no_grad()
    def embed(self, x_mv: np.ndarray) -> torch.Tensor:
        """x_mv: [N, 12, 1000] mV -> z [N, 12, 32]"""
        x = (torch.from_numpy(x_mv) - self.mean) / self.std
        return self.encoder(x.to(self.device)).cpu()

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> dict:
        """Returns {seed: {A_logits, B_logits, C_logits, A_lead_idx}} (numpy)."""
        z = z.to(self.device)
        v5_mask = torch.zeros(z.shape[0], 12, device=self.device)
        v5_mask[:, V5] = 1.0
        out = {}
        for s, h in self.heads.items():
            p, f = h["proposed"], h["fixed_V5"]
            lA, mA = p(z, 1)
            lB = p.classifier(p.fusion(z, v5_mask))
            lC = f.classifier(f.fusion(z, v5_mask))
            out[s] = {"A_logits": lA.cpu().numpy(), "B_logits": lB.cpu().numpy(), "C_logits": lC.cpu().numpy(),
                      "A_lead_idx": mA.argmax(1).cpu().numpy()}
        return out
