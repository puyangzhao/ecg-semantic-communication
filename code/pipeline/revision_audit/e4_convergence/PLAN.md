# E4 — extended training of the lead-aware backbone (post hoc revision analysis)

Written 2026-11-08 (timestamp and SHA-256 in `plan_hashes/PLAN_FROZEN.sha256`), **before** any E4 model
was trained and before any E4 output on fold 10 existed. This analysis is post hoc: it was added after
the authors had already seen fold-10 results of the 40-epoch backbone (formal run 2026-11-04 and
revision-2 run 2026-11-07). Fold 10 has therefore been used before; E4 uses it once more, after a
second freeze (`plan_hashes/EVAL_FROZEN.sha256`). It is never described as pre-specified.

## 1. Audit of the original 40-epoch runs (`revision_audit/backbone/runs/new_seed{0,1,2}`)

| seed | params (total / encoder) | epochs run | best epoch (0-based) | stopped by patience? |
|---|---|---|---|---|
| 0 | 828,934 / 824,320 | 40 | 39 | no (hit 40-epoch cap) |
| 1 | 828,934 / 824,320 | 40 | 37 | no (hit 40-epoch cap) |
| 2 | 828,934 / 824,320 | 40 | 38 | no (hit 40-epoch cap) |

* Early stopping: patience 8 on the validation composite (mean fold-9 macro-AUROC over K=12 and fixed
  random K=1/3/6 masks, rng 777). None of the three runs triggered it; all best epochs are in the
  last 3 epochs. Convergence was therefore not established.
* Checkpoints: only `best.pt` = {model_state_dict, epoch, composite}. No last checkpoint, no optimizer
  state (Adam moments), no numpy/torch RNG state. Optimizer: Adam lr 1e-3, wd 1e-4, **no scheduler**.

## 2. Decision: retrain from scratch, same seeds

Full resumption from the original last state is impossible (no last weights, no Adam moments, no RNG
state). Loading `best.pt` weights alone would not be a resumption, so it is not done. All three seeds
are retrained from scratch with `train_backbone_e4.py`, which imports the original model, forward,
mask and prediction code and changes only:
* max epochs 40 → **120**; patience 8 and the validation composite unchanged;
* the learning rate is constant, so a longer budget changes no other setting;
* a full-state `last.pt` is saved every epoch (no randomness consumed).
Data split (folds 1–8 train, 9 val), cached 100 Hz waveforms `/scratch/pz28/cache/ptbxl100/{train,val}.npz`,
5 superclass labels, architecture, loss, batch 64, lead-subset sampling, seed handling, device (CPU,
16 threads) as in the original run. Seeds 0, 1, 2 — all reported, none dropped.
Implementation check (not a selection criterion): with the same seed, code and threads the first 40
epochs are expected to reproduce the original history; the actual agreement is reported.

## 3. Convergence definition (fixed now)

* A seed is **converged under the training rule** if early stopping triggers (8 consecutive epochs
  without improvement of the validation composite) before epoch 120.
* If a seed reaches epoch 120 without triggering (equivalently best epoch ≥ 112), it is reported as
  **"convergence not confirmed"**; no further extension is made.
* No test metric is used for stopping, model choice or any setting. 0.925 is not a target.

## 4. Downstream models (validation only, before fold 10)

On each E4 encoder (best.pt, frozen): per-lead embeddings for train/val (`backbone/embed.py` logic);
heads with the unchanged Phase-2A recipe `k1_fair/train_heads.py` (K∈{1,3,6} per batch, Adam 1e-3/wd 1e-4,
batch 64, 60 epochs, patience 10, early stopping on mean fold-9 macro-AUROC over K), paired seed = backbone
seed: proposed selector (A), global learned ranking, random-K, and 12 fixed-lead heads.
* V5 (matched fixed-V5 head, C) stays the pre-existing reference.
* E4 best fixed lead: max seed-mean fold-9 K=1 macro-AUROC over the 12 fixed-lead heads; ties →
  AUPRC → lead order (rule of `k1_fair/lock_new_best_lead.py`).
Validation curves, best epochs and stopping status are reviewed before the second freeze.

## 5. Fold-10 analysis (once, after `EVAL_FROZEN.sha256`)

Statistics reuse `revision_audit/common/stats.py` and `formal_eval/k1_eval.py` helpers: one shared set
of 2000 patient-level bootstrap resamples (rng 20261110); per resample, per-seed difference then mean over
seeds; percentile 95% interval; two-sided bootstrap p. All comparisons exploratory (post hoc), no
multiplicity adjustment, interpreted by intervals.
1. E4 backbone, 12-lead input: macro-AUROC/AUPRC per seed, seed mean, seed SD, bootstrap CI of the mean;
   CD and STTC AUROC/AUPRC.
2. E4 − original 40-epoch backbone (paired by seed): 12-lead macro-AUROC/AUPRC, CD, STTC.
3. K=1 on the E4 backbone: selector (A) − matched fixed V5 (C); selector − E4 val-selected best fixed
   lead; macro and CD/STTC AUROC/AUPRC.
4. Context only: E4 K=1 contrasts next to the 40-epoch ones (no new test of their difference).
Two uncertainty sources are reported separately: the bootstrap interval (test-set sampling,
conditional on the trained models) and the between-seed SD (training-run variability).

## 6. Reference value ~0.925

Reviewer 1 cites ≈0.925 for full-lead PTB-XL models [their ref. 20]. Comparable only if dataset version,
label definition (superdiagnostic, label handling), sampling rate, folds and metric match; checked against
Strodthoff et al. 2021 and reported as a literature reference unless all match.
