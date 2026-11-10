# Is the reviewer's ~0.925 a same-protocol comparison?

Reviewer 1, comment 4: "roughly 0.925 reported for full-lead PTB-XL models [20]". The reviewers' reference list
is not available, so [20] cannot be identified with certainty. The closest match is Strodthoff et al.,
IEEE JBHI 25(5):1519-1528, 2021 (`strodthoff2021benchmarks` in references.bib).

Checked 2026-11-08 against the public benchmark code (github.com/helme/ecg_ptbxl_benchmarking: README
leaderboard, code/utils/utils.py):

| Item | Benchmark | This study | Same? |
|---|---|---|---|
| Task | 5 diagnostic superclasses, multilabel | same | yes |
| Label rule | any diagnostic SCP key, likelihood ignored | same | yes |
| Records without superclass | **dropped** (`superdiagnostic_len > 0`), train and test | **kept as all-negative** (334 train / 37 val / 40 test) | **no** |
| Folds | 1–8 train / 9 val / 10 test | same | yes |
| PTB-XL version | released with the 2020 paper (pre-1.0.3) | 1.0.3 | **no / not verifiable** |
| Sampling rate | code supports 100 and 500 Hz; rate behind the leaderboard numbers not stated on the page | 100 Hz | not verified |
| Metric | macro-AUROC on fold 10, single number per model (bootstrap uncertainty in parentheses) | macro-AUROC, seed mean of 3 runs | partly |
| Superdiagnostic leaderboard | resnet1d_wang 0.930, xresnet1d101 0.928, lstm 0.927, fcn_wang 0.925, inception1d 0.921, lstm_bidir 0.921, Wavelet+NN 0.874 | – | – |

Conclusion: the ~0.925 is a literature reference and not a same-condition comparison. 0.925 is one value
(fcn_wang) in a 0.921–0.930 range of deep models. The test-side difference (unlabelled records) is
quantified descriptively by the labelled-only sensitivity in PLAN_ADDENDUM_1. The training-side and
version differences remain.
