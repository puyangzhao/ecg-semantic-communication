#!/usr/bin/env python
"""Fixed-subset enumeration (K=1: 12, K=3: 220, K=6: 924) and the K=1 label-assisted oracle,
all through ONE subset-agnostic receiver per backbone/seed, on cached embeddings (no retraining).

Subset-agnostic receiver:
  orig  : Phase 2A random_k head (outputs/runs/phase2a_random_k_seed{s}; trained on uniformly random
          K-subsets, K in {1,3,6}; masked mean pooling + linear head)
  new   : the revision backbone's own receiver (revision_audit/backbone/runs/new_seed{s}; trained
          end-to-end with random lead subsets, masked attention pooling + MLP)
Selection (fold 9, --split val): for each K the subset maximizing the seed-mean macro-AUROC; ties ->
higher seed-mean macro-AUPRC -> lexicographically smallest lead-index tuple. Written to
locked_subsets_<backbone>.json and never revised afterwards. --split test re-uses the locked file.

Policies compared through the same receiver: enumerated best fixed subset, submitted clinical subset
(K=3: II,V2,V5; K=6: limb leads; K=1: II and V5), per-record random subsets (mean metric over 20 draws),
and the per-record selector's own top-K choice (proposed selector of the same seed).

K=1 oracle: per record, the lead whose single-lead prediction has the lowest mean (unweighted) BCE over
the five TRUE labels. Uses labels, so it is not a deployable policy; primary statistic is mean BCE;
AUROC/AUPRC are descriptive only (not an upper bound for those metrics).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revision_audit/backbone"))
sys.path.insert(0, os.path.join(ROOT, "revision_audit/k1_fair"))
from lead_resnet import LeadAwareModel  # noqa: E402
from scripts.train_phase2a import build_selector  # noqa: E402
from src.data.clinical_lead_subsets import FIXED_CLINICAL_SUBSETS, LEAD_NAMES, lead_indices  # noqa: E402
from src.models.phase2a_model import Phase2ASelectionModel  # noqa: E402
from train_backbone import NEW_KW  # noqa: E402
from train_heads import build as build_head  # noqa: E402

SEEDS = (0, 1, 2)


def metrics(y, logits):
    p = 1 / (1 + np.exp(-logits))
    au = [roc_auc_score(y[:, c], p[:, c]) for c in range(5)]
    ap = [average_precision_score(y[:, c], p[:, c]) for c in range(5)]
    return {"macro_auroc": float(np.mean(au)), "macro_auprc": float(np.mean(ap)),
            **{f"auroc_{n}": float(a) for n, a in zip(["NORM", "MI", "STTC", "CD", "HYP"], au)},
            **{f"auprc_{n}": float(a) for n, a in zip(["NORM", "MI", "STTC", "CD", "HYP"], ap)}}


def bce(logits, y):
    return (np.logaddexp(0, logits) - y * logits).mean(-1)  # mean over 5 labels


class Agnostic:
    def __init__(self, backbone, seed, split):
        self.backbone = backbone
        if backbone == "orig":
            d = torch.load(os.path.join(ROOT, f"outputs/phase2a/embeddings/{split}.pt"))
            m = Phase2ASelectionModel(build_selector("random_k", 32), latent_dim=32, num_classes=5)
            m.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2a_random_k_seed{seed}/checkpoints/best.pt"),
                                         map_location="cpu")["model_state_dict"])
            self.f = lambda z, mk: m.classifier(m.fusion(z, mk))
            sel = Phase2ASelectionModel(build_selector("proposed", 32), latent_dim=32, num_classes=5)
            sel.load_state_dict(torch.load(os.path.join(ROOT, f"outputs/runs/phase2a_proposed_seed{seed}/checkpoints/best.pt"),
                                           map_location="cpu")["model_state_dict"])
        else:
            d = torch.load(os.path.join(ROOT, f"revision_audit/backbone/emb/new_seed{seed}/{split}.pt"))
            bb = LeadAwareModel(**NEW_KW)
            bb.load_state_dict(torch.load(os.path.join(ROOT, f"revision_audit/backbone/runs/new_seed{seed}/best.pt"))["model_state_dict"])
            bb.eval()
            self.f = lambda z, mk: bb.receiver(z, mk)
            sel = build_head("proposed", f"new_s{seed}", None)
            sel.load_state_dict(torch.load(os.path.join(ROOT, f"revision_audit/k1_fair/runs/new_s{seed}_proposed_seed{seed}/best.pt"))["model_state_dict"])
        sel.eval()
        self.z, self.y = d["z"], d["y"].numpy()
        with torch.no_grad():
            self.sel_scores = sel.selector.raw_scores(self.z)
        self.patient_id, self.ecg_id = d["patient_id"].numpy(), d["ecg_id"].numpy()

    @torch.no_grad()
    def logits_mask(self, mk):
        return self.f(self.z, torch.as_tensor(mk, dtype=torch.float32)).numpy().astype(np.float64)

    def logits_subset(self, S):
        mk = np.zeros((len(self.y), 12), np.float32)
        mk[:, list(S)] = 1
        return self.logits_mask(mk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=("orig", "new"), required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    a = ap.parse_args()
    if a.split == "test":
        assert os.path.exists(os.path.join(ROOT, "revision_audit/lock/LOCK_MANIFEST.json"))
    torch.set_num_threads(16)
    A = {s: Agnostic(a.backbone, s, a.split) for s in SEEDS}
    y = A[0].y
    lock_path = os.path.join(HERE, f"locked_subsets_{a.backbone}.json")
    out = {"backbone": a.backbone, "split": a.split, "n": int(len(y)), "enumeration": {}, "policies": {}, "oracle": {}}
    if a.split == "val":
        locked = {}
        for K in (1, 3, 6):
            rows = []
            for S in itertools.combinations(range(12), K):
                ms = [metrics(y, A[s].logits_subset(S)) for s in SEEDS]
                rows.append({"subset": [LEAD_NAMES[i] for i in S], "idx": list(S),
                             "macro_auroc_mean": float(np.mean([m["macro_auroc"] for m in ms])),
                             "macro_auprc_mean": float(np.mean([m["macro_auprc"] for m in ms])),
                             "per_seed_auroc": [m["macro_auroc"] for m in ms]})
            rows.sort(key=lambda r: (-r["macro_auroc_mean"], -r["macro_auprc_mean"], r["idx"]))
            out["enumeration"][K] = {"n_subsets": len(rows), "top10": rows[:10], "all": rows}
            locked[K] = rows[0]["idx"]
        json.dump({"rule": "max seed-mean fold-9 macro-AUROC; ties -> AUPRC -> lexicographic", "backbone": a.backbone,
                   "locked": {K: [LEAD_NAMES[i] for i in v] for K, v in locked.items()}, "locked_idx": locked},
                  open(lock_path, "w"), indent=2)
    lk = {int(k): v for k, v in json.load(open(lock_path))["locked_idx"].items()}
    for K in (1, 3, 6):
        pol = {}
        for s in SEEDS:
            ag = A[s]
            r = {"enumerated_best_fixed": metrics(y, ag.logits_subset(lk[K]))}
            if K == 1:
                r["fixed_II"] = metrics(y, ag.logits_subset(lead_indices(["II"])))
                r["fixed_V5"] = metrics(y, ag.logits_subset(lead_indices(["V5"])))
            else:
                r["clinical_subset"] = metrics(y, ag.logits_subset(lead_indices(FIXED_CLINICAL_SUBSETS[K])))
            rng = np.random.default_rng(1000 + s * 10 + K)
            rand = []
            for _ in range(20):
                mk = np.zeros((len(y), 12), np.float32)
                idx = np.argsort(rng.random((len(y), 12)), 1)[:, :K]
                np.put_along_axis(mk, idx, 1, 1)
                rand.append(metrics(y, ag.logits_mask(mk)))
            r["random_per_record_mean20"] = {k: float(np.mean([q[k] for q in rand])) for k in rand[0]}
            mk = np.zeros((len(y), 12), np.float32)
            np.put_along_axis(mk, ag.sel_scores.topk(K, 1).indices.numpy(), 1, 1)
            r["per_record_selector_choice"] = metrics(y, ag.logits_mask(mk))
            pol[s] = r
        out["policies"][K] = pol
    # ---- K=1 oracle ----
    for s in SEEDS:
        ag = A[s]
        per_lead = np.stack([ag.logits_subset([l]) for l in range(12)], 1)  # [N, 12, 5]
        b = bce(per_lead, y[:, None, :])  # [N, 12]
        orc = b.argmin(1)
        sel = ag.sel_scores.argmax(1).numpy()
        n = np.arange(len(y))
        best_fixed = int(b.mean(0).argmin())
        out["oracle"][s] = {
            "mean_bce": {"oracle": float(b[n, orc].mean()), "selector_choice": float(b[n, sel].mean()),
                         "V5": float(b[:, LEAD_NAMES.index("V5")].mean()), "II": float(b[:, 1].mean()),
                         f"best_fixed_by_bce_this_split({LEAD_NAMES[best_fixed]})": float(b[:, best_fixed].mean()),
                         "random_lead_expected": float(b.mean())},
            "descriptive_metrics": {"oracle": metrics(y, per_lead[n, orc]), "selector_choice": metrics(y, per_lead[n, sel]),
                                    "V5": metrics(y, per_lead[:, LEAD_NAMES.index("V5")])},
            "oracle_lead_frequency": {LEAD_NAMES[i]: float((orc == i).mean()) for i in range(12)},
            "selector_agrees_with_oracle": float((sel == orc).mean()),
            "note": "oracle uses the true labels; mean BCE is the primary statistic; AUROC/AUPRC descriptive, not a bound"}
    if a.split == "val":
        for K in out["enumeration"]:
            out["enumeration"][K].pop("all")  # full table saved separately
    json.dump(out, open(os.path.join(HERE, "results", f"subset_enum_{a.backbone}_{a.split}.json"), "w"), indent=2)
    print(json.dumps(json.load(open(lock_path))["locked"]))
    for K in (1, 3, 6):
        print(K, {k: round(np.mean([out["policies"][K][s][k]["macro_auroc"] for s in SEEDS]), 4) for k in out["policies"][K][0]})
    print("oracle", {k: round(np.mean([out["oracle"][s]["mean_bce"][k] for s in SEEDS]), 4) for k in ("oracle", "selector_choice", "V5", "random_lead_expected")})


if __name__ == "__main__":
    main()
