"""PTB-XL diagnostic superclass multilabel target construction.

Convention (standard PTB-XL benchmarking approach, e.g. Strodthoff et al. 2021):
for each record, parse `scp_codes` (a stringified dict of SCP-ECG statement ->
likelihood), keep only statements marked `diagnostic == 1` in scp_statements.csv,
map each to its `diagnostic_class` (one of NORM/MI/STTC/CD/HYP), and take the
*set* of resulting superclasses as the record's label. A record can and does
carry more than one positive superclass — this is a multilabel problem, and we
do not collapse multi-label records to a single class or drop them.

A record whose scp_codes contain no diagnostic statement (e.g. only form/rhythm
statements) yields an all-zero label vector; such records are reported, not
silently discarded, in the data audit.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(raw: str) -> dict:
    """scp_codes is stored as a Python dict literal string, e.g. "{'NORM': 100.0}"."""
    if pd.isna(raw):
        return {}
    return ast.literal_eval(raw)


def build_scp_to_superclass_map(scp_statements: pd.DataFrame) -> dict:
    """Map SCP-ECG statement code -> diagnostic superclass, diagnostic statements only."""
    diag = scp_statements[scp_statements["diagnostic"] == 1.0]
    return dict(zip(diag.index, diag["diagnostic_class"]))


def record_superclasses(scp_codes: dict, scp_to_superclass: dict) -> set:
    classes = set()
    for code in scp_codes:
        sc = scp_to_superclass.get(code)
        if sc is not None:
            classes.add(sc)
    return classes


def build_label_matrix(df: pd.DataFrame, scp_statements: pd.DataFrame) -> np.ndarray:
    """df must have a 'scp_codes' column (raw string or already-parsed dict).
    Returns a [N, 5] float32 multi-hot matrix in SUPERCLASSES order.
    """
    scp_to_superclass = build_scp_to_superclass_map(scp_statements)
    class_index = {c: i for i, c in enumerate(SUPERCLASSES)}

    n = len(df)
    y = np.zeros((n, len(SUPERCLASSES)), dtype=np.float32)
    for row_idx, raw in enumerate(df["scp_codes"].values):
        codes = raw if isinstance(raw, dict) else parse_scp_codes(raw)
        for sc in record_superclasses(codes, scp_to_superclass):
            y[row_idx, class_index[sc]] = 1.0
    return y
