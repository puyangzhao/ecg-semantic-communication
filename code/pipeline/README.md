# Training and evaluation

Run the commands below from `code/pipeline/`. Download the datasets separately; checkpoints and waveform caches are generated locally. See the repository README for installation.

Scripts retain their research-stage filenames to preserve imports and checkpoint paths. The encoder is trained first and frozen; clean-link selection and noisy-channel repetition are trained separately.

## Data

| Dataset | Location | Notes |
|---|---|---|
| PTB-XL 1.0.3 (PhysioNet) | `data/ptbxl/` (`ptbxl_database.csv`, `scp_statements.csv`, `records100/`) | Official `strat_fold`: 1–8 train, 9 validation, 10 test. |
| PhysioNet "Large-scale 12-lead ECG database for arrhythmia study" v1.0.0 (external cohort) | `data/ecg_arrhythmia_1.0.0/extracted/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/` | `SHA256SUMS.txt` in `data/ecg_arrhythmia_1.0.0/_meta/`. |

Waveform caches are written to `cache/`.

## Entry points, in order

Seeds are 0, 1 and 2 unless stated otherwise. `$PY` is the environment's python.

| Stage | Command | Output used by the paper |
|---|---|---|
| 0. Data audit (labels, folds) | `$PY scripts/run_data_audit.py` | Table I counts |
| 1. Preprocessing + shared per-lead encoder (100 Hz, per-lead z-score from training folds; `src/data/`) | `$PY scripts/train_baseline.py --config configs/baseline.yaml --run-name baseline_run1` (seed 42) | frozen encoder |
| 2. Lead embeddings | `$PY scripts/precompute_embeddings.py` | `outputs/phase2a/embeddings/{train,val,test}.pt` |
| 3. Lead selection (selector and comparators) | `$PY scripts/train_phase2a.py --method {proposed,fixed_clinical_k,global_learned_k,random_k,ceiling_k12} --seed s --config configs/phase2a.yaml --run-name phase2a_<method>_seed<s>` | Fig. 2, Table II |
| 3b. Matched fixed V5 (test fold never read) | `$PY revision_audit/fixed_v5/run_fixed_lead.py --k1-lead V5 --seed s` | primary comparator |
| 4. Ranked repetition | `$PY scripts/train_phase2b_pilot.py --config configs/phase2b_pilot.yaml --run-name phase2b_pilot_k1_awgn_seed0` (and `_seed1.yaml`, `_seed2.yaml`) | Fig. 3, Table III, Fig. S1 |
| 5. Transmission comparators | `revision_audit/comm/run_train_comm.sh`, then `$PY revision_audit/comm/lock_policies.py` | Fig. 3, Table III |
| 6. Thresholds and subsets (validation fold) | `$PY revision_audit/thresholds/freeze_thresholds.py`; `$PY revision_audit/subset_enum/enumerate_subsets.py --backbone orig --split val` | Table S4; Table S3 |
| 7. Lead-aware encoder (exploratory) | `$PY revision_audit/backbone/cache_waveforms.py --splits train,val`; `$PY revision_audit/backbone/train_backbone.py --arch new --seed s`; `$PY revision_audit/backbone/embed.py --seed s --splits train,val`; `revision_audit/k1_fair/run_new_pipeline.sh` | Table II rows 5–6, Fig. S5 |
| 8. Record-level gate (exploratory) | `revision_audit/k1_gate_pilot/` (`train_proposed_valonly.py`, `oof_inner_fold.py`, `gate_pilot.py`, `multiseed_bootstrap.py`), `revision_audit/gate_lock/gate_eval.py` | Fig. S6 |
| 9. Lock | `revision_audit/lock/make_lock.py` writes `LOCK_MANIFEST.json`. The test-fold scripts refuse to run without it; the manifest of the reported run is included. | — |
| 10. Test fold and external cohort | `revision_audit/formal_eval/run_formal.sh` runs, among others, `formal_eval/k1_eval.py --split test`, `comm/eval_comm.py --split test --seed s`, `comm/comm_report.py --split test`, `external_ecgarr/external_eval.py` | Tables II–IV, Figs. 2–4, Tables S2–S6 |
| 11. Six-seed stability (post hoc) | `$PY revision_audit/fixed_v5/run_fixed_lead.py --k1-lead V5 --seed {3,4,5}`; `$PY revision_audit/stability6/stability6_eval.py` | Table S10 |

Steps 10–11 also need checkpoints at the paths in the scripts:

- `outputs/runs/...`;
- `revision_audit/fixed_v5/outputs/runs/...`;
- `revision_audit/k1_gate_pilot/phase2a_extra_seeds/runs/...` for selector seeds 3–5, which are produced by `train_proposed_valonly.py --seed s`.

External-cohort inputs are prepared by `external_ecgarr/verify_sha256.py`, `input_qa.py` and `build_labels.py`. The frozen SNOMED CT map is in `label_map_frozen.json`. The frozen K=1 receiver pipeline is in `revision_audit/external_validation_plan/external_pipeline.py`.


## Checks

After preparing the training embeddings, run:

```bash
PY=python OUT=/tmp/ecg_code_check bash checks/run_code_check.sh
```

This short software check uses patient-disjoint subsets of training folds only and does not contribute to paper results.

## Additional experiments

- `revision2/`: channel-budget sweeps, Rician/time-varying fading, signal-quality and noiseless checks.
- `revision_audit/e4_convergence/`: extended training of the lead-aware encoder. Run `bash revision_audit/e4_convergence/run_pipeline_val.sh` from this directory after the preceding stages.
- `revision_audit/stability6/`: post hoc six-seed selector-versus-fixed-lead analysis.
- `revision_audit/gate_lock/`: exploratory switching, including validation-selected runs and the all-six sensitivity analysis.

Frozen manifests describe the reported research runs, including artifacts that are generated locally. Reproducing a new run requires generating its artifacts and recording a new manifest; the supplied hashes are provenance, not hashes of newly trained models.
