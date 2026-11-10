# E4 plan addendum 1 (written during E4 training, before any E4 encoder had finished and before any E4 fold-10 output)

1. **Thread count.** Heads are trained with `--threads 2 -P 12`, as in the original `k1_fair/run_new_pipeline.sh`.
   A scratch dry run showed that 3 threads changes the 4th decimal of a head's fold-9 AUROC, while 2 threads
   reproduces the original head exactly (seed 1, fixed V5: 0.86737 vs 0.86737). The dry run used the 40-epoch
   encoder as a stand-in and nothing from fold 10.
2. **Protocol-alignment sensitivity for the ~0.925 reference.** This is descriptive and not a comparison test.
   Strodthoff et al. 2021 (superdiagnostic task) drop the records that have no superclass label. The present
   protocol keeps them as all-negative: 40 of 2198 fold-10 records. In addition to the primary fold-10 numbers,
   the 12-lead macro-AUROC/AUPRC of E4 and of the 40-epoch backbone will be reported on fold 10 restricted to
   records with at least one superclass label. Other differences remain and are reported, not removed:
   - PTB-XL 1.0.3 here vs the version used in that paper;
   - training also includes the unlabelled records;
   - different architectures;
   - single models per seed here.
3. **Repo-root override.** `downstream_e4.py` and `eval_e4.py` read `E4_REPO_ROOT` only for the scratch dry run.
   It is unset in every real run.
