#!/usr/bin/env python
"""Apply label_map_frozen.json + exclusions -> per-record outcome labels and a CONSORT-style count table.
Reads header codes only (no model, no predictions)."""
import json, os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "label_map_frozen.json")))
df = pd.read_csv(os.path.join(HERE, "results/header_audit_records.csv"), low_memory=False)
df["codes"] = df["dx"].fillna("").map(lambda s: set(c for c in str(s).split(",") if c))
flow = [("header files", len(df))]
keep = pd.Series(True, index=df.index)
steps = [
    ("unreadable", df.get("read_error").notna() if "read_error" in df else pd.Series(False, index=df.index)),
    ("not 12 standard leads", df["n_sig"].fillna(0).ne(12)),
    ("non-finite / adapter refused", ~df["adapter_ok"].fillna(False).astype(bool)),
    ("empty Dx", df["codes"].map(len).eq(0)),
    ("flat lead", df["n_flat_leads"].fillna(0).gt(0)),
    ("clipping", df["max_abs_mV"].fillna(0).ge(32.76)),
    ("exact duplicate pair", df["sig_sha1"].map(df["sig_sha1"].value_counts()).gt(1)),
    ("pacing rhythm", df["codes"].map(lambda c: "10370003" in c)),
]
for name, m in steps:
    newly = keep & m
    keep &= ~m
    flow.append((f"excluded: {name}", int(newly.sum())))
flow.append(("eligible", int(keep.sum())))
out = df.loc[keep, ["record", "rel", "age", "sex"]].copy()
codes = df.loc[keep, "codes"]
counts = {}
for oc, spec in M["outcomes"].items():
    pos = codes.map(lambda c: bool(c & set(spec["positive_codes"])))
    amb = codes.map(lambda c: bool(c & set(spec["ambiguous_codes_excluded_unless_a_positive_code_is_present"])))
    lab = np.where(pos, 1, np.where(amb, -1, 0))  # -1 = excluded for this outcome
    out[oc] = lab
    counts[oc] = {"positive": int((lab == 1).sum()), "negative": int((lab == 0).sum()),
                  "excluded_ambiguous": int((lab == -1).sum()),
                  "prevalence_among_analysed": float((lab == 1).sum() / (lab >= 0).sum())}
out.to_csv(os.path.join(HERE, "results/external_labels.csv"), index=False)
json.dump({"flow": flow, "outcome_counts": counts}, open(os.path.join(HERE, "results/label_flow.json"), "w"), indent=2)
for f in flow: print(f)
print(json.dumps(counts, indent=1))
