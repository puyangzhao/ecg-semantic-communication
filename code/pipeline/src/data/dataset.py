"""PTB-XL 100Hz WFDB data loading, normalization, and PyTorch Dataset.

Produces input tensors of shape [12, 1000] per record (12 leads, 10s @ 100Hz),
batched by DataLoader to [B, 12, 1000].
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from src.data.labels import build_label_matrix, parse_scp_codes
from src.data.splits import assign_split

NUM_LEADS = 12
SIGNAL_LENGTH = 1000  # 10s @ 100Hz


def load_ptbxl_metadata(root: str) -> pd.DataFrame:
    """Loads ptbxl_database.csv + scp_statements.csv, builds superclass label
    matrix, assigns official train/val/test split. Does not touch waveform files.
    """
    db_path = os.path.join(root, "ptbxl_database.csv")
    scp_path = os.path.join(root, "scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    scp_statements = pd.read_csv(scp_path, index_col=0)

    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)
    labels = build_label_matrix(df, scp_statements)
    for i, cls in enumerate(["NORM", "MI", "STTC", "CD", "HYP"]):
        df[f"label_{cls}"] = labels[:, i]

    df["split"] = assign_split(df)
    return df


def load_waveform(root: str, filename_lr: str) -> np.ndarray:
    """filename_lr is the relative path column from ptbxl_database.csv, e.g.
    'records100/00000/00001_lr' (no extension). Returns [12, T] float32.
    """
    record_path = os.path.join(root, filename_lr)
    signal, _fields = wfdb.rdsamp(record_path)  # [T, 12]
    return signal.astype(np.float32).T  # [12, T]


@dataclass
class NormalizationStats:
    mean: np.ndarray  # [12]
    std: np.ndarray  # [12]

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "NormalizationStats":
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])


def fit_normalization(df_train: pd.DataFrame, root: str, filename_col: str = "filename_lr",
                       eps: float = 1e-8) -> NormalizationStats:
    """Fits per-lead mean/std using training-split waveforms only.

    Single pass over training records, accumulating sum and sum-of-squares per
    lead in float64 (each record's [12, T] contribution is reduced with
    vectorized numpy ops, not a per-sample Python loop, so this stays fast even
    over the full ~17.6k-record training split).
    """
    total_sum = np.zeros(NUM_LEADS, dtype=np.float64)
    total_sumsq = np.zeros(NUM_LEADS, dtype=np.float64)
    total_count = 0

    for filename in df_train[filename_col]:
        x = load_waveform(root, filename).astype(np.float64)  # [12, T]
        total_sum += x.sum(axis=1)
        total_sumsq += (x ** 2).sum(axis=1)
        total_count += x.shape[1]

    mean = total_sum / total_count
    variance = total_sumsq / total_count - mean ** 2
    std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None)) + eps
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


class PTBXLDataset(Dataset):
    def __init__(self, df_split: pd.DataFrame, root: str, norm_stats: NormalizationStats,
                 filename_col: str = "filename_lr", label_cols: list[str] | None = None):
        self.df = df_split.reset_index(drop=True)
        self.root = root
        self.filename_col = filename_col
        self.label_cols = label_cols or ["label_NORM", "label_MI", "label_STTC", "label_CD", "label_HYP"]
        self.mean = torch.from_numpy(norm_stats.mean).view(NUM_LEADS, 1)
        self.std = torch.from_numpy(norm_stats.std).view(NUM_LEADS, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x = load_waveform(self.root, row[self.filename_col])  # [12, T]
        x = torch.from_numpy(x)
        if x.shape[1] != SIGNAL_LENGTH:
            # PTB-XL 100Hz records are documented as exactly 1000 samples; guard
            # against any malformed record rather than silently truncating/padding.
            raise ValueError(
                f"Unexpected signal length {x.shape[1]} for {row[self.filename_col]} "
                f"(expected {SIGNAL_LENGTH})"
            )
        x = (x - self.mean) / self.std
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return x, y
