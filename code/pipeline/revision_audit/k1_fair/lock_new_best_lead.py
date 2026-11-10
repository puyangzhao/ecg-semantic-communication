#!/usr/bin/env python
"""Rank the 12 separately trained fixed-K=1-lead heads on the revision backbone by the seed-mean fold-9
macro-AUROC (ties -> macro-AUPRC -> lead order) and lock the best lead; also writes the fold-9 logits of
new_A (proposed selector + head), new_B (same head, V5 forced) and new_C (fixed-V5 head) for the
threshold freezing. Original backbone ranking reported alongside (descriptive)."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "revision_audit/backbone"))
from train_heads import build
from src.data.clinical_lead_subsets import LEAD_NAMES
rank = {}
for bb in ("orig", "new"):
    rows = []
    for L in LEAD_NAMES:
        r = [json.load(open(os.path.join(HERE, f"runs/{bb if bb=='orig' else 'new_s'+str(s)}_fixed_lead_{L}_seed{s}/result.json")))["val_K1"] for s in (0, 1, 2)]
        rows.append({"lead": L, "auroc_mean": float(np.mean([q["macro_auroc"] for q in r])), "auprc_mean": float(np.mean([q["macro_auprc"] for q in r])),
                     "auroc_per_seed": [q["macro_auroc"] for q in r]})
    rows.sort(key=lambda q: (-q["auroc_mean"], -q["auprc_mean"], LEAD_NAMES.index(q["lead"])))
    rank[bb] = rows
best = rank["new"][0]["lead"]
json.dump({"rule": "max seed-mean fold-9 K=1 macro-AUROC of separately trained fixed-lead heads; ties -> AUPRC -> lead order",
           "locked_best_lead": best, "ranking_new": rank["new"], "ranking_orig_descriptive": rank["orig"],
           "note": "V5 stays the pre-locked reference lead (chosen on the original backbone); this backbone's own best is reported separately"},
          open(os.path.join(HERE, "results/new_backbone_best_fixed_lead.json"), "w"), indent=2)
V5 = LEAD_NAMES.index("V5"); out = {}
for s in (0, 1, 2):
    va = torch.load(os.path.join(ROOT, f"revision_audit/backbone/emb/new_seed{s}/val.pt")); Z = va["z"]; out["y_true"] = va["y"].numpy()
    def ld(method, lead=None):
        m = build(method, f"new_s{s}", lead)
        m.load_state_dict(torch.load(os.path.join(HERE, f"runs/new_s{s}_{method}{'_' + lead if lead else ''}_seed{s}/best.pt"))["model_state_dict"]); return m.eval()
    v5 = torch.zeros(len(Z), 12); v5[:, V5] = 1
    with torch.no_grad():
        pa = ld("proposed"); out[f"new_A_seed{s}"] = pa(Z, 1)[0].numpy(); out[f"new_B_seed{s}"] = pa.logits_with_mask(Z, v5).numpy()
        out[f"new_C_seed{s}"] = ld("fixed_lead", "V5")(Z, 1)[0].numpy()
np.savez(os.path.join(ROOT, "revision_audit/thresholds/new_backbone_val_logits.npz"), **out)
print("new backbone best fixed lead:", best, [(q["lead"], round(q["auroc_mean"], 4)) for q in rank["new"][:5]])
