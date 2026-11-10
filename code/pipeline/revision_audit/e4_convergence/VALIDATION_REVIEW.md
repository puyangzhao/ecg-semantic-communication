# E4 validation review (fold 9 only; written before the E4 fold-10 run)

Date: 2026-11-08 (CST). Sources: `results/e4_training.json`, `results/e4_eval_val.json`, `results/e4_best_fixed_lead.json`, `logs/`.

## Encoders (max 120 epochs, patience 8, constant LR)
| seed | epochs run | best epoch (1-based) | stopped by patience | best fold-9 composite: 40-ep run → E4 | fold-9 12-lead macro-AUROC of the used checkpoint: 40-ep → E4 |
|---|---|---|---|---|---|
| 0 | 52 | 44 | yes | 0.8871 → 0.8903 | 0.9284 → 0.9302 |
| 1 | 70 | 62 | yes | 0.8873 → 0.8910 | 0.9260 → 0.9304 |
| 2 | 54 | 46 | yes | 0.8880 → 0.8904 | 0.9312 → 0.9298 |

* All three seeds stopped under the original rule: 8 epochs without improvement, well before 120.
  By the definition fixed in PLAN.md §3 they are **converged under the training rule**.
* Implementation check: epochs 1–40 of every E4 run are bit-identical to the original 40-epoch run
  (train loss and all validation metrics). E4 is therefore the original run continued, not a different run.
* Gains after epoch 40 are small. The composite rose by 0.002–0.004, and the 12-lead fold-9 AUROC moved by
  −0.001 to +0.004 (mean +0.0016, 95% CI −0.0002 to 0.0034, patient bootstrap on fold 9).

## Heads (Phase-2A recipe, 60 epochs max, patience 10, threads 2 as originally)
* All 45 heads (3 seeds × {proposed, global, random, 12 fixed leads}) stopped by patience before 60 epochs.
* The proposed selector reaches its best fold-9 composite in epochs 1–7. This is the same pattern as the
  40-epoch heads (best epochs 2–4), so it is not new in E4. The rule was not changed.
* E4 best fixed lead (rule of lock_new_best_lead.py): **V6** (seed-mean fold-9 K=1 AUROC 0.8758; V5 0.8729).
  This is the same lead as for the 40-epoch encoder. V5 stays the matched reference.

## Fold-9 K=1 (not used to change anything)
selector − V5: +0.0230 [0.0162, 0.0303]; selector − V6: +0.0202 [0.0132, 0.0275]. For the 40-epoch encoder,
selector − V5 was +0.0245.

No setting, seed, lead or model was changed after this review. Next step: freeze (plan_hashes/EVAL_FROZEN.sha256),
then `eval_e4.py --split test` once.
