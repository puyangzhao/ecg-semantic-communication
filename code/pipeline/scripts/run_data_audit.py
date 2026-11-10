#!/usr/bin/env python
"""Loads PTB-XL metadata, fits normalization on train, runs the full data audit,
and writes docs/data_audit.md + outputs/reports/data_audit.json.

Usage: python scripts/run_data_audit.py --root data/ptbxl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.audit import render_audit_markdown, run_full_audit, save_audit_json
from src.data.dataset import NUM_LEADS, SIGNAL_LENGTH, load_ptbxl_metadata, load_waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ptbxl")
    parser.add_argument("--out-md", default="docs/data_audit.md")
    parser.add_argument("--out-json", default="outputs/reports/data_audit.json")
    args = parser.parse_args()

    df = load_ptbxl_metadata(args.root)

    # tensor-shape spot check on a few records across splits
    shape_check = {}
    for split in ["train", "val", "test"]:
        row = df[df["split"] == split].iloc[0]
        x = load_waveform(args.root, row["filename_lr"])
        shape_check[split] = {
            "expected": [NUM_LEADS, SIGNAL_LENGTH],
            "actual": list(x.shape),
            "record": row["filename_lr"],
        }

    audit = run_full_audit(df, tensor_shape_check=shape_check)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_audit_json(audit, args.out_json)

    md = render_audit_markdown(audit)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)

    print(md)
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
