#!/usr/bin/env python
"""E4 downstream steps on the frozen extended-training encoders (validation only).

  embed  --seed S                 per-lead embeddings (train, val) of runs/new_seed{S}_e120/best.pt,
                                  with backbone/embed.py's embed() and its receiver-vs-e2e sanity check
  head   --seed S --method M [--lead L]
                                  k1_fair/train_heads.py main() unchanged (Phase-2A recipe), only the
                                  embedding directory and the run directory are redirected to E4
  lock                            E4 best fixed lead by the k1_fair/lock_new_best_lead.py rule
                                  (max seed-mean fold-9 K=1 macro-AUROC; ties -> AUPRC -> lead order)
Fold 10 is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("E4_REPO_ROOT", os.path.dirname(os.path.dirname(HERE)))  # override only for the scratch dry run
for p in ("", "revision_audit/backbone", "revision_audit/k1_fair"):
    sys.path.insert(0, os.path.join(ROOT, p))
import train_heads  # noqa: E402
from embed import embed  # noqa: E402
from lead_resnet import LeadAwareModel  # noqa: E402
from train_backbone import CACHE, NEW_KW  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402

SEEDS = (0, 1, 2)


def emb_dir(seed):
    return os.path.join(HERE, "emb", f"seed{seed}")


def load_encoder(seed):
    m = LeadAwareModel(**NEW_KW)
    m.load_state_dict(torch.load(os.path.join(HERE, f"runs/new_seed{seed}_e120/best.pt"))["model_state_dict"])
    return m.eval()


def do_embed(seed, splits, threads):
    assert "test" not in splits or os.path.exists(os.path.join(HERE, "plan_hashes/EVAL_FROZEN.sha256")), \
        "fold 10 only after EVAL_FROZEN"
    torch.set_num_threads(threads)
    m = load_encoder(seed)
    os.makedirs(emb_dir(seed), exist_ok=True)
    for sp in splits:
        d = np.load(os.path.join(CACHE, f"{sp}.npz"))
        z = embed(m, d["x"])
        ref = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{sp}.pt"))
        assert np.array_equal(ref["ecg_id"].numpy(), d["ecg_id"])
        torch.save({"z": z, "y": torch.from_numpy(d["y"]), "ecg_id": ref["ecg_id"], "patient_id": ref["patient_id"],
                    "sex": ref["sex"], "age": ref["age"]}, os.path.join(emb_dir(seed), f"{sp}.pt"))
        with torch.no_grad():
            x0 = torch.from_numpy(d["x"][:64])
            dev = float((m(x0, torch.ones(64, 12)) - m.receiver(z[:64], torch.ones(64, 12))).abs().max())
        print(sp, tuple(z.shape), "receiver-vs-e2e max abs", dev, flush=True)


def do_head(seed, method, lead, threads):
    train_heads.HERE = HERE  # runs -> e4_convergence/runs/e4_s{seed}_...
    train_heads.emb_dir = lambda backbone: emb_dir(int(backbone.split("_s")[1]))
    sys.argv = ["train_heads.py", "--backbone", f"e4_s{seed}", "--method", method, "--seed", str(seed),
                "--threads", str(threads)] + (["--lead", lead] if lead else [])
    train_heads.main()


def do_lock():
    rows = []
    for L in LEAD_NAMES:
        r = [json.load(open(os.path.join(HERE, f"runs/e4_s{s}_fixed_lead_{L}_seed{s}/result.json")))["val_K1"] for s in SEEDS]
        rows.append({"lead": L, "auroc_mean": float(np.mean([q["macro_auroc"] for q in r])),
                     "auprc_mean": float(np.mean([q["macro_auprc"] for q in r])),
                     "auroc_per_seed": [q["macro_auroc"] for q in r]})
    rows.sort(key=lambda q: (-q["auroc_mean"], -q["auprc_mean"], LEAD_NAMES.index(q["lead"])))
    out = {"rule": "max seed-mean fold-9 K=1 macro-AUROC of separately trained fixed-lead heads; ties -> AUPRC -> lead order",
           "locked_best_lead": rows[0]["lead"], "ranking_e4": rows,
           "note": "V5 stays the pre-existing reference lead; the E4 backbone's own best is reported separately"}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results/e4_best_fixed_lead.json"), "w"), indent=2)
    print("E4 best fixed lead:", out["locked_best_lead"], [(q["lead"], round(q["auroc_mean"], 4)) for q in rows[:5]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("embed", "head", "lock"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--method")
    ap.add_argument("--lead", default=None)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.step == "embed":
        do_embed(a.seed, a.splits.split(","), a.threads)
    elif a.step == "head":
        do_head(a.seed, a.method, a.lead, a.threads)
    else:
        do_lock()


if __name__ == "__main__":
    main()
