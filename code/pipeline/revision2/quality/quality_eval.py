#!/usr/bin/env python
"""PLAN §5: PTB-XL signal-quality annotations and K=1 lead selection (frozen models, cached predictions).

Fields: baseline_drift, static_noise, burst_noise, electrodes_problems (ptbxl_database.csv, PTB-XL 1.0.3).
Parsing: split on commas/whitespace/semicolons; lead tokens I, II, III, aVR, aVL, aVF, V1-V6 (case-insensitive,
'?' removed); ranges "A-B" in the order I, II, III, aVR, aVL, aVF, V1..V6; "alles" = all 12 leads; other words carry
no lead information. Record groups: annotated (any field non-empty) vs not annotated (not called clean).

Models (seeds 0-2): selector (orig_A), matched fixed V5 (orig_C), fixed Lead II (orig_II), cached logits in
revision_audit/formal_eval/results/k1_logits_{split}.npz; selector lead per record: comm/leads/proposed_seed{s}_{split}.npy.

Usage: quality_eval.py --split val ; quality_eval.py --split test (after the freeze)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2common as C  # noqa: E402
from src.data.clinical_lead_subsets import LEAD_NAMES  # noqa: E402

FIELDS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
LEADS_UP = [l.upper() for l in LEAD_NAMES]
V5 = LEAD_NAMES.index("V5")
MODELS = ["orig_A", "orig_C", "orig_II"]


def parse_field(txt):
    """-> (set of lead indices, has_text, unresolved_words)"""
    if not isinstance(txt, str) or not txt.strip(" ,"):
        return set(), False, []
    leads, other = set(), []
    # fold-9 implementation fix (logs/ISSUES.md #2): "I - AVF" / "I --AVL" -> "I-AVF"; "v1-6" -> "V1-V6"
    txt = re.sub(r"\s*-+\s*", "-", txt)
    txt = re.sub(r"\b[vV](\d)-(\d)\b", r"V\1-V\2", txt)
    for tok in re.split(r"[,\s;]+", txt.strip()):
        t = tok.replace("?", "").strip().upper()
        if not t:
            continue
        if t in ("ALLES", "ALLE", "ALL"):
            leads |= set(range(12))
        elif t in LEADS_UP:
            leads.add(LEADS_UP.index(t))
        elif "-" in t and all(p in LEADS_UP for p in t.split("-")) and len(t.split("-")) == 2:
            a, b = (LEADS_UP.index(p) for p in t.split("-"))
            leads |= set(range(min(a, b), max(a, b) + 1))
        else:
            other.append(tok)
    return leads, True, other


QG = {}


def _inter_draw(idx):
    """one whole-fold patient resample: (selector - V5) in annotated minus in not-annotated records, per seed"""
    y, P, annot = QG["y"], QG["P"], QG["annot"]
    ia, inn = idx[annot[idx]], idx[~annot[idx]]
    return {s: (C.metric_vec(y, P[("orig_A", s)], ia) - C.metric_vec(y, P[("orig_C", s)], ia))
            - (C.metric_vec(y, P[("orig_A", s)], inn) - C.metric_vec(y, P[("orig_C", s)], inn)) for s in C.SEEDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    if a.split == "test":
        C.require_frozen()
    d = np.load(os.path.join(C.RA, f"formal_eval/results/k1_logits_{a.split}.npz"))
    y, pid, ecg = d["y"], d["patient_id"], d["ecg_id"]
    emb = torch.load(os.path.join(C.ROOT, f"outputs/phase2a/embeddings/{a.split}.pt"))
    assert np.array_equal(emb["ecg_id"].numpy(), ecg), "record order differs from the embeddings"
    sel = {s: np.load(os.path.join(C.RA, f"comm/leads/proposed_seed{s}_{a.split}.npy")) for s in C.SEEDS}
    db = pd.read_csv(os.path.join(C.ROOT, "data/ptbxl/ptbxl_database.csv")).set_index("ecg_id").loc[ecg]
    assert (db["strat_fold"].values == (9 if a.split == "val" else 10)).all()
    n = len(ecg)
    flags = np.zeros((n, 12), bool)
    any_text = np.zeros(n, bool)
    resolved = np.zeros(n, bool)
    counts = {}
    words = {}
    for f in FIELDS:
        has, res_ = np.zeros(n, bool), np.zeros(n, bool)
        allx = 0
        for i, txt in enumerate(db[f].values):
            L, h, other = parse_field(txt)
            has[i], res_[i] = h, bool(L)
            allx += len(L) == 12
            for l in L:
                flags[i, l] = True
            for w in other:
                words[w] = words.get(w, 0) + 1
        any_text |= has
        resolved |= res_
        counts[f] = {"non_empty": int(has.sum()), "lead_resolved": int(res_.sum()), "lead_unresolved": int((has & ~res_).sum()),
                     "all_12_leads": int(allx)}
    nf = flags.sum(1)
    counts["any_field"] = {"records": int(any_text.sum()), "patients": int(len(np.unique(pid[any_text]))),
                           "lead_resolved": int(resolved.sum()), "lead_unresolved_only": int((any_text & ~resolved).sum()),
                           "all_12_flagged": int((nf == 12).sum()), "partial_1_to_11": int(((nf >= 1) & (nf <= 11)).sum())}
    counts["not_annotated"] = {"records": int((~any_text).sum()), "patients": int(len(np.unique(pid[~any_text])))}
    counts["flag_rate_per_lead_among_resolved"] = {LEAD_NAMES[l]: float(flags[resolved, l].mean()) for l in range(12)}
    counts["non_lead_words_top"] = dict(sorted(words.items(), key=lambda kv: -kv[1])[:25])
    out = {"split": a.split, "post_hoc_revision_analysis": True, "n_records": n, "counts": counts, "groups": {}}

    P = {(m, s): C.sig(d[f"{m}_seed{s}"]) for m in MODELS for s in C.SEEDS}
    # within-group analyses
    for gname, gmask in (("annotated", any_text), ("not_annotated", ~any_text)):
        idx = np.flatnonzero(gmask)
        yy, pp = y[idx], pid[idx]
        Pg = {k: v[idx] for k, v in P.items()}
        obs = {k: C.metric_vec(yy, v) for k, v in Pg.items()}
        boot = C.bootstrap(yy, Pg, C.patient_draws(pp, a.B))
        out["groups"][gname] = {"records": int(len(idx)), "patients": int(len(np.unique(pp))),
                                "positives": dict(zip(C.CLASSES, map(int, yy.sum(0)))),
                                "absolute": {m: C.absolute(obs, boot, m) for m in MODELS},
                                "paired": {f"{x}-{z}": C.paired(obs, boot, x, z) for x, z in (("orig_A", "orig_C"), ("orig_A", "orig_II"))}}
    # interaction: whole-fold patient draws, split by group inside each draw
    draws = C.patient_draws(pid, a.B)
    obs_i = {}
    for s in C.SEEDS:
        for g, msk in (("a", any_text), ("n", ~any_text)):
            for m in ("orig_A", "orig_C"):
                obs_i[(m + g, s)] = C.metric_vec(y[msk], P[(m, s)][msk])

    QG.update(y=y, P=P, annot=any_text)
    with Pool(32) as pool:
        bi = pool.map(_inter_draw, draws, chunksize=16)
    inter = {}
    for i, nm in enumerate(C.MNAMES):
        per_seed = [float((obs_i[("orig_Aa", s)][i] - obs_i[("orig_Ca", s)][i]) - (obs_i[("orig_An", s)][i] - obs_i[("orig_Cn", s)][i]))
                    for s in C.SEEDS]
        bd = np.array([np.mean([b[s][i] for s in C.SEEDS]) for b in bi])
        inter[nm] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "sd_between_seeds": float(np.std(per_seed, ddof=1)),
                     "ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))], "p_boot": C.boot_p(bd)}
    out["interaction_annotated_minus_not_of_selector_minus_V5"] = inter

    # lead level: records with 1..11 flagged leads
    part = np.flatnonzero((nf >= 1) & (nf <= 11))
    lead_out = {"records": int(len(part)), "patients": int(len(np.unique(pid[part]))), "per_seed": {}}
    O, E = {}, {}
    for s in C.SEEDS:
        p_hat = np.bincount(sel[s][~any_text], minlength=12) / (~any_text).sum()
        O[s] = flags[part, sel[s][part]].astype(float)
        E[s] = (flags[part] * p_hat[None]).sum(1)
        lead_out["per_seed"][str(s)] = {"observed_flagged_pick_rate": float(O[s].mean()), "expected_from_unannotated_freq": float(E[s].mean()),
                                        "observed_minus_expected": float((O[s] - E[s]).mean()),
                                        "selection_freq_not_annotated": dict(zip(LEAD_NAMES, map(float, p_hat)))}
    lead_out["observed_minus_expected_mean"] = float(np.mean([(O[s] - E[s]).mean() for s in C.SEEDS]))
    lead_out["v5_flag_rate"] = float(flags[part, V5].mean())
    lead_out["random_choice_rate"] = float((nf[part] / 12).mean())
    rng_draws = C.patient_draws(pid[part], a.B)
    bd = np.array([np.mean([(O[s][ix] - E[s][ix]).mean() for s in C.SEEDS]) for ix in rng_draws])
    lead_out["observed_minus_expected_ci95"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
    lead_out["p_boot"] = C.boot_p(bd)
    lead_out["per_lead"] = {LEAD_NAMES[l]: {"flagged_records": int(flags[part, l].sum()),
                                            "select_rate_when_flagged": float(np.mean([(sel[s][part][flags[part, l]] == l).mean() if flags[part, l].any() else np.nan for s in C.SEEDS])),
                                            "select_rate_not_annotated": float(np.mean([(sel[s][~any_text] == l).mean() for s in C.SEEDS]))}
                            for l in range(12)}
    out["lead_level"] = lead_out
    C.dump(out, os.path.join(C.R2, "results", "quality", f"quality_{a.split}.json"))
    print(counts["any_field"], counts["not_annotated"])
    print({g: (v["records"], round(v["paired"]["orig_A-orig_C"]["macro_auroc"]["mean"], 4)) for g, v in out["groups"].items()})
    print("lead level", lead_out["records"], lead_out["observed_minus_expected_mean"], lead_out["observed_minus_expected_ci95"])


if __name__ == "__main__":
    main()
