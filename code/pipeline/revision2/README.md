# revision2 — additional revision analyses (J-BHI, second round)

Read `PLAN.md` (fixed before any result), then `SUMMARY.md`. Failures and fixes: `logs/ISSUES.md`.
Python: the interpreter of the environment in `environment.yml` (`$PY`), `CUDA_VISIBLE_DEVICES=""`. Logit caches: `$CACHE_DIR/revision2/`.
Absolute paths inside `PLAN.md`, `*.sh` and `plan_hashes/*.sha256` refer to the authors' HPC system; they are left unchanged so that the frozen sha256 hashes still verify.
(sha256 in `results/cache_sha256.txt`). Nothing outside `revision2/` was modified; `formal_eval/run_formal.sh` was not re-run.

| Step | Command | Output |
|---|---|---|
| 1 reproduce locked numbers | `python revision2/repro/reproduce_key.py`; `python revision2/channels/lock_and_report.py --check-lock --check-report test` | `results/repro_key_numbers.json`, `results/checks/` |
| 2 fold 9 | `revision2/channels/run_val.sh`; `revision2/budget/check_codecs.py`; `revision2/budget/run_train.sh`; `budget/eval_n.py --split val --seed s --n {8,32}` (and `--n 16 --check16`); `noiseless/noiseless_check.py --split val`; `quality/quality_eval.py --split val`; `channels/lock_and_report.py --lock --report val`; `budget/lock_and_report.py --lock --report val` | `results/*/…val…`, locked policies |
| 3 freeze | `revision2/make_freeze.sh` | `plan_hashes/FROZEN.sha256` |
| 4 fold 10 (once) | `revision2/run_fold10.sh` (refuses on hash mismatch or second start) | `results/*/…test…`, `logs/fold10/` |
| 5 manuscript | `cd revision2/manuscript/code && python make_all.py --out ../source`; tectonic `main.tex`, `supplement.tex`, `response_letter/Response2.tex`; `marked/build_marked.sh` | `manuscript/` |
| 6 ZIP | `revision2/manuscript/tools/build_zip.sh` | `JBHI_revision2_submission_2026-11-07.zip` |

Code: `r2common.py` (paths, metrics, k1_eval bootstrap), `r2comm.py` (policy rule and comm_report statistics),
`repro/`, `noiseless/`, `channels/` (Rician, time variation), `budget/` (N = 8/32 models, SSCC codecs, training, evaluation),
`quality/` (PTB-XL annotations). Trained N = 8/32 models: `budget/runs/` (24 runs).
