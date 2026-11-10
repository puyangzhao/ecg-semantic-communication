"""Data audit: record/patient counts, label prevalence, missing values, split
validation (no patient leakage across splits), tensor shape spot-check.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.data.labels import SUPERCLASSES


@dataclass
class SplitSummary:
    n_records: int
    n_patients: int
    label_prevalence: dict
    n_zero_label_records: int
    n_multilabel_records: int


def check_no_patient_leakage(df: pd.DataFrame, split_col: str = "split",
                              patient_col: str = "patient_id") -> dict:
    """Returns overlap counts between every pair of splits (should all be 0)."""
    splits = df[split_col].unique().tolist()
    patient_sets = {s: set(df.loc[df[split_col] == s, patient_col]) for s in splits}
    overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlaps[f"{a}_vs_{b}"] = len(patient_sets[a] & patient_sets[b])
    return overlaps


def summarize_split(df_split: pd.DataFrame) -> SplitSummary:
    label_cols = [f"label_{c}" for c in SUPERCLASSES]
    labels = df_split[label_cols].values
    prevalence = {
        c: {
            "count": int(labels[:, i].sum()),
            "fraction": float(labels[:, i].mean()) if len(df_split) else 0.0,
        }
        for i, c in enumerate(SUPERCLASSES)
    }
    n_positive_per_record = labels.sum(axis=1)
    return SplitSummary(
        n_records=len(df_split),
        n_patients=df_split["patient_id"].nunique(),
        label_prevalence=prevalence,
        n_zero_label_records=int((n_positive_per_record == 0).sum()),
        n_multilabel_records=int((n_positive_per_record > 1).sum()),
    )


def missing_value_report(df: pd.DataFrame) -> dict:
    cols = ["age", "height", "weight", "sex", "scp_codes", "strat_fold", "filename_lr"]
    report = {}
    for c in cols:
        if c == "scp_codes":
            n_missing = int(df["scp_codes"].apply(lambda d: len(d) == 0).sum())
        else:
            n_missing = int(df[c].isna().sum())
        report[c] = n_missing
    return report


def run_full_audit(df: pd.DataFrame, tensor_shape_check: dict | None = None) -> dict:
    audit = {
        "total_records": len(df),
        "total_patients": int(df["patient_id"].nunique()),
        "splits": {},
        "patient_leakage": check_no_patient_leakage(df),
        "missing_values": missing_value_report(df),
        "tensor_shape_check": tensor_shape_check or {},
    }
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split]
        audit["splits"][split] = asdict(summarize_split(df_split))
    return audit


def save_audit_json(audit: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)


def render_audit_markdown(audit: dict) -> str:
    lines = ["# PTB-XL Data Audit Report", ""]
    lines.append(f"- Total records: {audit['total_records']}")
    lines.append(f"- Total unique patients: {audit['total_patients']}")
    lines.append("")
    lines.append("## Split sizes and label prevalence")
    lines.append("")
    lines.append("| Split | Records | Patients | Zero-label records | Multi-label records | " +
                  " | ".join(SUPERCLASSES) + " (count / %) |")
    lines.append("|---|---|---|---|---|" + "---|" * len(SUPERCLASSES))
    for split in ["train", "val", "test"]:
        s = audit["splits"][split]
        prev_cells = " | ".join(
            f"{s['label_prevalence'][c]['count']} ({100*s['label_prevalence'][c]['fraction']:.1f}%)"
            for c in SUPERCLASSES
        )
        lines.append(
            f"| {split} | {s['n_records']} | {s['n_patients']} | {s['n_zero_label_records']} | "
            f"{s['n_multilabel_records']} | {prev_cells} |"
        )
    lines.append("")
    lines.append("## Patient leakage check (must all be 0)")
    lines.append("")
    for pair, n in audit["patient_leakage"].items():
        lines.append(f"- {pair}: {n}")
    lines.append("")
    lines.append("## Missing values")
    lines.append("")
    for col, n in audit["missing_values"].items():
        lines.append(f"- {col}: {n} missing/empty")
    lines.append("")
    if audit.get("tensor_shape_check"):
        lines.append("## Tensor shape spot-check")
        lines.append("")
        for k, v in audit["tensor_shape_check"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)
