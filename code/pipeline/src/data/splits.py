"""Official PTB-XL stratified-fold split assignment.

Per the PTB-XL documentation: strat_fold 1-8 = train, 9 = validation, 10 = test.
These folds are pre-computed by the dataset authors to balance diagnostic
superclass distribution and patient assignment; we only consume them, never
re-derive our own split.
"""
from __future__ import annotations

import pandas as pd

TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


def assign_split(df: pd.DataFrame, fold_col: str = "strat_fold") -> pd.Series:
    def _split(fold: int) -> str:
        if fold in TRAIN_FOLDS:
            return "train"
        if fold == VAL_FOLD:
            return "val"
        if fold == TEST_FOLD:
            return "test"
        raise ValueError(f"Unexpected strat_fold value: {fold}")

    return df[fold_col].astype(int).map(_split)
