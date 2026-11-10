#!/usr/bin/env python
"""Collect E4 training histories (E4 and original 40-epoch runs), head summaries and the E4 best-lead lock into
one JSON for the figure/table code (manuscript code/data/e4_training.json). Reads only; no evaluation.

Usage: python revision_audit/e4_convergence/export_e4_data.py OUT.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KEYS = ("epoch", "train_loss", "val_auroc_K12", "val_auprc_K12", "val_auroc_K1", "val_auroc_K3", "val_auroc_K6", "composite")


def run_summary(d):
    r = json.load(open(os.path.join(d, "train_result.json")))
    c = json.load(open(os.path.join(d, "config.json")))
    h = [{k: row[k] for k in KEYS} for row in r["history"]]
    return {"epochs_max": c["epochs_max"], "patience": c["patience"], "params_encoder": c["params_encoder"],
            "params_total": c["params_total"], "epochs_run": len(h), "best_epoch": r["best_epoch"],
            "best_composite": r["best_composite"], "stopped_by_patience": r.get("stopped_by_patience", len(h) < c["epochs_max"]),
            "runtime_min": r["runtime_min"], "history": h}


out = {"e4": {}, "o40": {}, "heads_e4": {}}
for s in (0, 1, 2):
    out["e4"][s] = run_summary(os.path.join(HERE, f"runs/new_seed{s}_e120"))
    out["o40"][s] = run_summary(os.path.join(ROOT, f"revision_audit/backbone/runs/new_seed{s}"))
    n = min(40, out["e4"][s]["epochs_run"])
    out["e4"][s]["first_epochs_identical_to_o40"] = n if all(
        out["e4"][s]["history"][i]["composite"] == out["o40"][s]["history"][i]["composite"] for i in range(n)) else -1
    heads = {}
    for name in sorted(os.listdir(os.path.join(HERE, "runs"))):
        if name.startswith(f"e4_s{s}_") and os.path.exists(os.path.join(HERE, "runs", name, "result.json")):
            r = json.load(open(os.path.join(HERE, "runs", name, "result.json")))
            heads[name] = {"best_epoch": r["best_epoch"], "epochs_run": len(r["history"]), "val_K1": r["val_K1"]}
    out["heads_e4"][s] = heads
out["e4_best_fixed_lead"] = json.load(open(os.path.join(HERE, "results/e4_best_fixed_lead.json")))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print({s: (out["e4"][s]["epochs_run"], out["e4"][s]["best_epoch"], out["e4"][s]["stopped_by_patience"],
           out["e4"][s]["first_epochs_identical_to_o40"]) for s in (0, 1, 2)})
