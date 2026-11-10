#!/usr/bin/env python
"""Header audit + input QA for PhysioNet ecg-arrhythmia 1.0.0. No model, no performance.

Per record: header fields (leads, fs, length, units/gain, age, sex, Dx SNOMED
codes), signal checks (NaN/inf, flat leads, amplitude), a hash of the decoded
signal (exact-duplicate detection), and the frozen adapter
(revision_audit/external_validation_plan/external_pipeline.py::to_training_format)
applied to produce the PTB-XL-format [12, 1000] mV array. Adapter outputs are
saved for later inference; per-lead z-scored distribution statistics are
compared with PTB-XL fold 9 (no labels, no model).

Usage: python revision_audit/external_ecgarr/input_qa.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revision_audit/external_validation_plan"))
from external_pipeline import to_training_format  # noqa: E402
from src.data.dataset import NormalizationStats  # noqa: E402

BASE = ("data/ecg_arrhythmia_1.0.0/extracted/"
        "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0")
CACHE = "cache/ecgarr"


def one(rel):
    path = os.path.join(BASE, rel)
    r = {"record": os.path.basename(rel), "rel": rel}
    try:
        h = wfdb.rdheader(path)
        r.update(n_sig=h.n_sig, fs=h.fs, sig_len=h.sig_len, units=",".join(h.units), sig_name=",".join(h.sig_name),
                 gain=",".join(str(g) for g in h.adc_gain))
        com = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in h.comments if ":" in c}
        r.update(age=com.get("Age"), sex=com.get("Sex"), dx=com.get("Dx", ""))
        sig, f = wfdb.rdsamp(path)
        r["sig_sha1"] = hashlib.sha1(np.ascontiguousarray(sig).tobytes()).hexdigest()
        r["any_nonfinite"] = bool(~np.isfinite(sig).all())
        sig0 = np.nan_to_num(sig)
        flat = (np.ptp(sig0, axis=0) < 1e-6)
        r["n_flat_leads"] = int(flat.sum())
        r["max_abs_mV"] = float(np.abs(sig0).max())
        r["median_ptp_mV"] = float(np.median(np.ptp(sig0, axis=0)))
        try:
            x = to_training_format(sig, f["sig_name"], f["units"], f["fs"])
            r["adapter_ok"], r["adapter_error"] = True, ""
        except Exception as e:  # refused by the frozen adapter
            x = None
            r["adapter_ok"], r["adapter_error"] = False, str(e)[:200]
    except Exception as e:
        r["read_error"] = str(e)[:200]
        x = None
    return r, x


def main():
    rels = []
    for top in sorted(os.listdir(os.path.join(BASE, "WFDBRecords"))):
        for sub in sorted(os.listdir(os.path.join(BASE, "WFDBRecords", top))):
            d = os.path.join(BASE, "WFDBRecords", top, sub)
            rels += [os.path.join("WFDBRecords", top, sub, f[:-4]) for f in sorted(os.listdir(d)) if f.endswith(".hea")]
    with Pool(48) as p:
        res = p.map(one, rels, chunksize=50)
    rows = [r for r, _ in res]
    df = pd.DataFrame(rows)
    os.makedirs(CACHE, exist_ok=True)
    ok = [i for i, (r, x) in enumerate(res) if x is not None]
    X = np.stack([res[i][1] for i in ok]).astype(np.float32)
    np.save(os.path.join(CACHE, "x_mv.npy"), X)
    df.iloc[ok][["record", "rel"]].to_csv(os.path.join(CACHE, "x_index.csv"), index=False)
    df.to_csv(os.path.join(HERE, "results/header_audit_records.csv"), index=False)
    # distribution vs PTB-XL fold 9 (no labels, no model)
    ns = NormalizationStats.load(os.path.join(ROOT, "outputs/norm_stats.npz"))
    Z = (X - ns.mean[None, :, None]) / ns.std[None, :, None]
    val = np.load("cache/ptbxl100/val.npz")["x"]
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    dist = {}
    for li, ln in enumerate(lead_names):
        e, v = Z[:, li, :], val[:, li, :]
        dist[ln] = {"ext_mean": float(e.mean()), "ext_std": float(e.std()), "ptbxl_val_mean": float(v.mean()),
                    "ptbxl_val_std": float(v.std()),
                    "ext_median_record_ptp_mV": float(np.median(np.ptp(X[:, li, :], axis=1))),
                    "ptbxl_val_median_record_ptp_mV": float(np.median(np.ptp(v * ns.std[li] + ns.mean[li], axis=1))),
                    "ext_frac_abs_z_gt_10": float((np.abs(e) > 10).mean())}
    codes = Counter(c for d in df["dx"].fillna("") for c in d.split(",") if c)
    summary = {
        "n_header_files": len(rels),
        "n_read_errors": int(df.get("read_error", pd.Series(dtype=str)).notna().sum()),
        "n_sig": df["n_sig"].value_counts().to_dict(), "fs": df["fs"].value_counts().to_dict(),
        "sig_len": df["sig_len"].value_counts().head(10).to_dict(), "units": df["units"].value_counts().to_dict(),
        "gain": df["gain"].value_counts().head(5).to_dict(),
        "lead_order": df["sig_name"].value_counts().to_dict(),
        "n_any_nonfinite": int(df["any_nonfinite"].sum()),
        "n_with_flat_lead": int((df["n_flat_leads"] > 0).sum()),
        "n_all_leads_flat": int((df["n_flat_leads"] == 12).sum()),
        "n_adapter_ok": int(df["adapter_ok"].sum()),
        "adapter_errors": df.loc[~df["adapter_ok"].astype(bool), "adapter_error"].value_counts().head(10).to_dict(),
        "n_empty_dx": int((df["dx"].fillna("") == "").sum()),
        "n_exact_duplicate_signal_groups": int((df["sig_sha1"].value_counts() > 1).sum()),
        "n_records_in_duplicate_groups": int(df["sig_sha1"].map(df["sig_sha1"].value_counts()).gt(1).sum()),
        "age_missing_or_nan": int(pd.to_numeric(df["age"], errors="coerce").isna().sum()),
        "sex": df["sex"].value_counts().to_dict(),
        "snomed_code_counts": dict(codes.most_common()),
        "per_lead_distribution_vs_ptbxl_val": dist,
        "max_abs_mV_quantiles": df["max_abs_mV"].quantile([0.5, 0.99, 0.999, 1.0]).to_dict(),
    }
    json.dump(summary, open(os.path.join(HERE, "results/input_qa_summary.json"), "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("snomed_code_counts", "per_lead_distribution_vs_ptbxl_val", "lead_order")}, indent=1, default=str))


if __name__ == "__main__":
    main()
