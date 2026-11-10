# E4 — extended training of the lead-aware encoder (post hoc)

This analysis is post hoc. The plan was written and hashed before training (`PLAN.md`; `plan_hashes/PLAN_FROZEN.sha256`, 2026-11-09T00:21:51Z), and the addendum before any encoder finished (`PLAN_ADDENDUM_1.md`, 00:35:35Z). Models and comparisons were frozen after the validation review (`VALIDATION_REVIEW.md`; `plan_hashes/EVAL_FROZEN.sha256`, 02:28:10Z). Fold 10 was then evaluated once (`logs/FOLD10_STEPS.log`, 02:28–02:33Z). Fold 10 had already been used in the original study and in revision rounds 1–2.

## Result summary (fold 10; 3 seeds, all reported)

| | seed 0 | seed 1 | seed 2 | mean (SD) |
|---|---|---|---|---|
| Epochs run / best epoch (max 120, patience 8) | 52 / 44 | 70 / 62 | 54 / 46 | all stopped by patience |
| 40-epoch runs: best epoch (cap 40) | 40 | 38 | 39 | none stopped |
| 12-lead macro-AUROC, ≤120 epochs | 0.9214 | 0.9244 | 0.9228 | 0.9229 (0.0015) |
| 12-lead macro-AUROC, 40 epochs | 0.9208 | 0.9196 | 0.9232 | 0.9212 (0.0018) |
| 12-lead macro-AUPRC, ≤120 epochs | 0.8033 | 0.8077 | 0.8110 | 0.8073 (0.0039) |

Paired differences use 2,000 shared patient-level bootstrap resamples, with per-seed differences averaged over seeds. The interval describes patient sampling conditional on the trained models; the per-seed SD describes training variability.

| Comparison | metric | mean Δ [95% CI] | per seed | SD |
|---|---|---|---|---|
| ≤120 − 40 epochs, 12-lead | macro-AUROC | +0.0016 [−0.0002, 0.0034] | +0.0006, +0.0048, −0.0005 | 0.0028 |
| | macro-AUPRC | +0.0019 [−0.0026, 0.0066] | | 0.0049 |
| | CD AUROC / AUPRC | +0.0060 [0.0020, 0.0103] / +0.0136 [0.0053, 0.0224] | | |
| | STTC AUROC / AUPRC | +0.0027 [0.0004, 0.0050] / +0.0048 [−0.0025, 0.0120] | | |
| K=1 selector − matched V5 (E4) | macro-AUROC | +0.0278 [0.0202, 0.0362] | +0.0295, +0.0326, +0.0214 | 0.0058 |
| | macro-AUPRC | +0.0467 [0.0314, 0.0627] | | 0.0185 |
| | CD AUROC / AUPRC | +0.0445 [0.0288, 0.0611] / +0.0564 [0.0269, 0.0871] | | |
| | STTC AUROC / AUPRC | −0.0164 [−0.0262, −0.0063] / −0.0175 [−0.0438, 0.0076] | | |
| K=1 selector − validation-best lead V6 (E4) | macro-AUROC | +0.0237 [0.0163, 0.0313] | | 0.0068 |
| | macro-AUPRC | +0.0432 [0.0279, 0.0599] | | |
| (reference) 40-epoch selector − V5 | macro-AUROC | +0.0306 [0.0233, 0.0385] | | 0.0015 |

- **Convergence.** All three seeds stopped under the original rule well before 120 epochs, so convergence under the training rule is confirmed. Epochs 1–40 are bit-identical to the original runs.
- **Best fixed lead.** The E4 best fixed lead on fold 9 is V6, as for the 40-epoch encoder.
- **0.925.** The ~0.925 value is not a same-protocol comparator (`REFERENCE_0925.md`). On the 2,158 labelled test records, the descriptive macro-AUROC is 0.923 for E4 and 0.922 for 40 epochs.
- **Cross-checks.** The 40-epoch values recomputed here equal `backbone/results/backbone_test.json` and `formal_eval/results/k1_eval_test.json` (o40_A−o40_C = +0.0306 [0.0233, 0.0385]).

## Files

- `train_backbone_e4.py`: the original trainer (imports `backbone/train_backbone.py`) with a 120-epoch cap and a full-state `last.pt`.
- `downstream_e4.py`: embed / head / lock steps. It reuses `backbone/embed.py` and `k1_fair/train_heads.py` unchanged, redirecting only their paths.
- `eval_e4.py`: the evaluation script. It reuses the `formal_eval/k1_eval.py` helpers and `common/stats.py`.
- `run_pipeline_val.sh`: runs the validation-only pipeline.
- `export_e4_data.py`: writes the summaries used by the figure code.
- `results/`: `e4_eval_{val,test}.json`, logits, `e4_training.json`, `e4_best_fixed_lead.json`. The test files are read-only.
- `runs/`: checkpoints (`best.pt`, full-state `last.pt`) and the 45 head runs.
- `emb/`: the embeddings.
- `manuscript/`: `base_ECG_Info_Checked/` (the author's base, read-only), `work/` (the revised documents), and `final/` (the delivery).

## Reproduce

```
PY=python  # interpreter of the environment in environment.yml; export CUDA_VISIBLE_DEVICES=""
for s in 0 1 2; do $PY revision_audit/e4_convergence/train_backbone_e4.py --seed $s --threads 16; done
revision_audit/e4_convergence/run_pipeline_val.sh            # embeddings, heads, lock, fold-9 evaluation
# fold 10 (done once; eval_e4.py refuses to overwrite results/e4_eval_test.json):
for s in 0 1 2; do $PY revision_audit/e4_convergence/downstream_e4.py embed --seed $s --splits test --threads 16; done
$PY revision_audit/e4_convergence/eval_e4.py --split test
$PY revision_audit/e4_convergence/export_e4_data.py revision_audit/e4_convergence/results/e4_training.json
```

The training ran on CPU (AMD EPYC 9354) with 16 threads per encoder and 2 threads per head, as in the original runs; thread count affects the 4th decimal of head metrics. Wall time was 72, 105, and 89 min for the three encoders, run in parallel.


Note: absolute paths inside the hashed `PLAN.md`, `run_pipeline_val.sh` and `plan_hashes/*.sha256` refer to the authors' HPC system and are left unchanged so that the frozen hashes still verify.
