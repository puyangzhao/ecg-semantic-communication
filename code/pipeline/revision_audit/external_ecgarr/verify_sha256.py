#!/usr/bin/env python
"""Verify every file of PhysioNet ecg-arrhythmia 1.0.0 against the official SHA256SUMS.txt."""
import hashlib, json, os, sys
from multiprocessing import Pool
BASE = "data/ecg_arrhythmia_1.0.0/extracted/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
def h(item):
    digest, rel = item
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        return rel, "missing"
    return rel, "ok" if hashlib.sha256(open(p, "rb").read()).hexdigest() == digest else "mismatch"
if __name__ == "__main__":
    items = [l.strip().split(None, 1) for l in open(os.path.join(BASE, "SHA256SUMS.txt")) if l.strip()]
    official = [l.strip().split(None, 1) for l in open("data/ecg_arrhythmia_1.0.0/_meta/SHA256SUMS.txt") if l.strip()]
    same_list = sorted(map(tuple, items)) == sorted(map(tuple, official))
    with Pool(32) as p:
        res = p.map(h, items, chunksize=500)
    bad = [r for r in res if r[1] != "ok"]
    out = {"n_listed": len(items), "n_ok": len(res) - len(bad), "bad": bad[:50], "n_bad": len(bad),
           "zip_sha256sums_identical_to_online_SHA256SUMS": same_list}
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "bad"}))
