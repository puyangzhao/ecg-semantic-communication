# Six-seed training-stability analysis of the primary K=1 comparison (post hoc)

Written 2026-11-06T03:24:33Z, before any of the training or evaluation below.

## Status

This is a **post hoc supplementary analysis**, added after the locked fold-10 results had been seen (formal run 2026-11-04 08:07:30 UTC). It was requested to answer Reviewer 1 comment 18 (training variability). It is not part of the locked plan (`revision_audit/lock/LOCK_MANIFEST.json`) and must not be described as pre-specified.

## Fixed in advance

- **Comparison.** Original frozen backbone, K=1: selector (orig_A) − matched fixed V5 (orig_C). This is the primary comparison of the paper.
- **Seeds.** Exactly 0, 1, 2, 3, 4, 5. No other seeds, no reselection, no exclusions.
- **Selector checkpoints (reused, not retrained):**
  - seeds 0–2: `outputs/runs/phase2a_proposed_seed{s}/checkpoints/best.pt`;
  - seeds 3–5: `revision_audit/k1_gate_pilot/phase2a_extra_seeds/runs/phase2a_proposed_seed{s}/checkpoints/best.pt`. These were trained earlier with the unchanged `configs/phase2a.yaml` on cuda (best epochs 5/35/16).
- **Fixed-V5 heads:**
  - seeds 0–2 exist in `revision_audit/fixed_v5/outputs/runs/fixed_k1V5_seed{s}`;
  - seeds 3–5 are missing and are trained now with the unchanged wrapper `revision_audit/fixed_v5/run_fixed_lead.py --k1-lead V5 --seed {3,4,5}`. That wrapper uses the original `configs/phase2a.yaml`: folds 1–8 train, fold 9 early stopping (mean-over-K validation macro-AUROC, patience 10, 60 epochs), and test.pt never read during training. The checkpoint is chosen by that original fold-9 rule; nothing else is selected.
- **Pairing.** Selector seed s is paired with fixed-V5 seed s.
- **Fold-10 predictions:**
  - seeds 0–2 are reused from `revision_audit/formal_eval/results/k1_logits_test.npz` (orig_A_seed*, orig_C_seed*), not recomputed;
  - seeds 3–5 are computed once from the checkpoints above on `outputs/phase2a/embeddings/test.pt`, with the same inference path as `formal_eval/k1_eval.py` (orig_A = model(Z,1); orig_C = classifier(fusion(Z, V5 mask))).
- **Reported:**
  - per seed: macro-AUROC and macro-AUPRC of both arms, the paired differences, and the CD and STTC AUROC/AUPRC differences;
  - across the six seeds: mean, SD (ddof=1, between training runs) and range of each difference;
  - patient-level paired bootstrap for the six-seed mean difference, kept separate from the between-seed SD. This uses 2000 resamples with `stats.patient_draws` (rng 20261110, as in k1_eval.py) and, per resample, the per-seed difference followed by the mean over seeds. The interval reflects patient sampling given these six trained models.
- **Metrics.** `revision_audit/common/stats.py` auroc/auprc (the same as the formal evaluation). As a consistency check, the seed 0–2 values must reproduce the locked k1_eval_test.json per-seed primary differences.
- **Not used.** The gate's validation-fold 3-of-6 selection plays no role here.
