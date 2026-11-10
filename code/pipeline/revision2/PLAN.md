# Revision 2: additional analyses for the J-BHI revision (plan, fixed before any new result)

Written 2026-11-07 (UTC; exact time and sha256 in `plan_hashes/PLAN_v1.sha256`).

## 0. Status and what was already seen

- **Post hoc.** Every analysis below was added after the locked formal evaluation (lock
  `revision_audit/lock/LOCK_MANIFEST.json`, sha256 29dd5a19…, formal fold-10 run 2026-11-04 08:07:30 UTC). PTB-XL
  fold 10 had been used once by the original submission and once by the locked revision run. These analyses are
  **additional revision analyses**. They are never called pre-specified, confirmatory or a first independent test.
- **Seen before this plan** (from `revision_audit/additional_channel_checks/`, run 2026-11-07 22:56–23:10 UTC; fold 9 only):
  - implementation checks: K_R=0 reproduces the cached Rayleigh LS-2 logits (max |Δ| 0); the policy-lock code
    reproduces `locked_comm_policies.json`; the report code reproduces `comm_report_val.json` (after the B=200 issue in
    `revision_audit/additional_channel_checks/logs/ISSUES.md`);
  - the fold-9 joint-JSCC noiseless check (Experiment 1 below), whose numbers were seen: bypass = AWGN noise off
    = 0.9286 macro-AUROC, 20 dB 0.9285, clean 12-lead reference 0.9215;
  - the fold-9 Rician LS-2 (3 and 10 dB) logits were computed; progress lines with a few AUROC values were printed.
    No policy, configuration or choice below was changed after that, and none depends on it.
  Nothing on fold 10 has been computed for any new analysis.
- **No tuning.** No hyperparameter, seed, SNR, configuration or rule below is changed after a result is seen. Seeds are
  0, 1, 2 (the primary-comparison seeds), all reported. Training and selection use folds 1–8 (training) and fold 9
  (early stopping, policy locking, implementation checks) only.
- **Order.** (1) cached-prediction reproduction of existing key numbers; (2) implementation and fold-9 runs;
  (3) freeze: `plan_hashes/FROZEN.sha256` lists this plan, all code, all locked fold-9 policies and fold-9 results;
  (4) fold-10 evaluation, once, by `run_fold10.sh`, which refuses to start unless every frozen hash matches.
- **Compute.** CPU (`CUDA_VISIBLE_DEVICES=""`, env `ecg_xhosp`), as for the original communication evaluation, so that
  new numbers are bit-comparable with the cached ones. GPUs were idle but are not needed for these model sizes.
- **Never overwritten.** Original models, locked results, cached logits under `/scratch/pz28/cache/comm_eval/`,
  `revision_audit/`, `final_submission/` and the old ZIPs. `formal_eval/run_formal.sh` is not re-run.
- **Notation.** K_R = Rician K-factor = power of the line-of-sight (direct) component divided by the power of the scattered
  component. K = number of transmitted leads (always 1 in the transmission analyses). N = complex channel uses for data.

## 1. Reproduction of existing key numbers from cached predictions (no retraining)

Recompute and compare with the locked JSONs (tolerance 1e-9 unless stated):
- Primary: selector − matched V5, macro-AUROC +0.0175 [0.0086, 0.0268] from `formal_eval/results/k1_logits_test.npz`
  with `stats.patient_draws(pid, 2000, seed=20261110)` vs `k1_eval_test.json`.
- Transmission: `rician_report.py --check --split test --B 1000` on the existing cache vs `comm_report_test.json`
  (policy means 0.769 / 0.864, +0.095 [0.089, 0.102], all contrasts).
- Clean 12-lead reference 0.914 (fold 10) from `outputs/runs/phase2a_ceiling_k12_seed*/test_raw_K12.npz` vs
  `phase2a_clinical_metrics.json`; joint JSCC at AWGN 20 dB 0.922 from the cache vs `comm_report_test.json`.

## 2. Experiment 1 — the same joint-JSCC checkpoint without noise vs 20 dB AWGN

- Checkpoints: `revision_audit/comm/runs/jscc_e2e_seed{0,1,2}/best.pt`, each analysed separately.
- Conditions, identical input (concatenated frozen 12-lead embeddings, 384 values), encoder, per-record power
  normalization (energy 16), 16-symbol bottleneck, decoder and five-logit output; only the channel step differs:
  - A: channel bypassed (`net.rx(net.tx(src))`);
  - B: full evaluation path with the AWGN channel and noise variance 0 (`channel_sim.channel(x, +inf dB, …, "awgn")`);
    also B′ (training path, `awgn_torch` with noise 0) and the cached "noiseless" logits;
  - C: AWGN 20 dB with the existing 30 paired realizations (`CRNBank`), which must equal the cached logits.
- Equivalence: max |Δlogit| ≤ 1e-5 between A, B, B′, cached noiseless; if not met, locate before interpreting.
- Clean 12-lead reference (a different model): `phase2a_ceiling_k12_seed{s}` = mean pooling of the 12 embeddings +
  linear 32→5 head, trained on clean input. Reported side by side with its architecture.
- Folds 9 and 10. Reported: per seed and seed mean macro-AUROC/AUPRC; paired differences C − A (same checkpoint =
  noise effect), A − reference and C − reference (labelled between-model differences, never noise effects);
  95% patient bootstrap (2000 draws, seed 20261110), per-seed difference then mean over seeds; C uses all 30
  realizations. Between-seed SD reported separately.
- Architecture ablation only if A ≠ B (not expected).

## 3. Experiment 2 — Rician block fading, perfect and LS-2 CSI

- Frozen methods (no retraining or fine-tuning for the new channels): original repetition `rep_orig_r{1,2,4}`;
  same-lead repetition `rep_ctrl_r{1,2,4}`; SSCC (5 configurations); single-lead JSCC `jscc_ctrl`; joint JSCC `jscc_e2e`.
- K_R ∈ {3 dB, 10 dB}. Unit total average power: h = sqrt(K_R/(K_R+1))·e^{jθ} + sqrt(1/(K_R+1))·g, E|h|² = 1;
  g = existing CRNBank Rayleigh draw; θ ~ U[0, 2π) per (realization, record) from a separate generator (seed 20261301
  fold 9, 20261302 fold 10). h constant within the record (pilots and data).
- CSI:
  - perfect: 16 data uses, energy 16, x̂ = y/h, SSCC LLR variance σ²/|h|²;
  - LS-2: 2 pilots x_p = 1 then 16 data uses = 18 uses, energy 18 per record for every method; ĥ = mean of the two
    pilot observations; equalization x̂ = y/ĥ, repetition combining after equalization, SSCC LLR variance σ²/|ĥ|². The
    receiver never uses the true h in this condition (checked in code: the LS path receives only y and y_p).
- Noise and pairing: existing CRNBank payload/pilot noise; same draws for every method, SNR, K_R and CSI.
- SNR: the seven policy SNRs {−15, −10, −5, 0, 5, 10, 20} dB (40 dB and noiseless also stored, reported only); J = 30.
- Checks (fold 9): K_R = 0 must reproduce the cached `rayleigh_perfect` and `rayleigh_ls2` logits (max |Δ| ≤ 1e-5);
  E|h|² within 0.02 of 1; LOS power fraction within 0.02 of K_R/(K_R+1).
- Policies: the existing fold-9 rule (`comm/lock_policies.py`) applied unchanged to each of the four Rician conditions
  (repetition: threshold r=4 if SNR ≤ τ else r=2; SSCC: per-SNR lookup; best fixed option; ties → AUPRC → smaller τ /
  list order). No submitted policy exists for Rician, so the original scheme uses its rule-selected τ.
- Report: as `comm_report.py`: per (method, seed, SNR) the realization mean; policy level = equal-weight mean over the
  seven SNRs; contrasts per seed then mean over seeds; 1000 patient resamples (seed 20261120) shared by everything,
  first 10 realizations inside the bootstrap; both macro-AUROC and macro-AUPRC. Contrasts: `comm_report.CONTRASTS` plus
  jscc_ctrl − rep_orig[policy] and sscc[lookup] − rep_orig[policy]. Every fixed option is also reported.

### 3b. Time-variation sensitivity (fixed K_R = 3 dB, LS-2 only)

- Channel trajectory over the 18 uses of a record, in transmission order: pilot, pilot, data 1…16.
  LOS term constant (as above); scattered term Gauss–Markov per use: g_1 = g (existing draw),
  g_t = ρ g_{t−1} + sqrt(1−ρ²) e_t, e_t ~ CN(0,1) from a separate generator (seed 20261311 fold 9, 20261312 fold 10),
  so E|h_t|² = 1 for every t. ρ ∈ {1, 0.95, 0} (correlation between consecutive uses; ρ=1 = block fading).
- Receiver: ĥ = mean of the two pilot observations (uses 1–2), applied to all 16 data symbols; no per-symbol true CSI.
- Methods: joint JSCC and same-lead repetition (r = 1, 2, 4 and the policy locked for Rician 3 dB LS-2). Frozen.
- Check: ρ = 1 must reproduce the block Rician 3 dB LS-2 logits exactly.
- Report: macro-AUROC/AUPRC per SNR and policy mean, per seed; contrasts ρ=0.95 − ρ=1 and ρ=0 − ρ=1 within each
  method, and jscc_e2e − rep_ctrl[policy] at each ρ, with the bootstrap of §3.

## 4. Experiment 3 — channel-use budget N ∈ {8, 16, 32}

- Name in the paper: "comparison at different channel-use budgets". Unit average symbol power is kept, so the total
  energy per record is N (8, 16, 32); gains cannot be attributed to bandwidth alone.
- Channels: AWGN (primary) and block Rayleigh, perfect CSI; seven SNRs; J = 30; seeds 0–2.
- Paired draws: N = 16 uses the existing CRNBank. For N = 8 the first 8 payload noise draws of the same bank; for N = 32
  the 16 existing draws followed by 16 new unit-variance draws (seed 20261321 fold 9, 20261322 fold 10). h = existing draw.
- N = 16: existing models and cached logits are reused (no recomputation beyond the reproduction check).
- Methods at N = 8, 32:
  - joint JSCC and single-lead JSCC: retrained; encoder output 2N reals, decoder input 2N reals, hidden widths
    unchanged (256/128 encoder, 128 decoder), per-record power normalization to energy N;
  - same-lead repetition: retrained; (m, r) with m·r = N, m ≤ 16 (the latent has 16 complex symbols):
    N = 8: (8,1), (4,2), (2,4); N = 32: (16,2), (8,4), (4,8); r = 1 at N = 32 is **not applicable**;
  - original repetition: the submitted checkpoint is **frozen and transferred** (its symbol ranking and head applied to
    the feasible (m, r)); labelled frozen transfer, not a matched retrained comparison;
  - SSCC: quantizer and head refitted per configuration with the existing recipe; codes restricted to the existing
    RM(1,5)/RM(2,5) family and uncoded; Gray QPSK/16-QAM with the existing average-power convention (one fixed
    16-QAM scale so that the training-fold mean energy is N; codewords are not renormalized per record):
    - N = 8: S6_RM1_16QAM, S16_RM2_16QAM, S16_UNC_QPSK, S32_UNC_16QAM (RM + QPSK needs a length-16 code → not applicable);
    - N = 16: the existing five configurations;
    - N = 32: S12_RM1x2_QPSK, S32_RM2x2_QPSK, S64_RM2x4_16QAM, S64_UNC_QPSK, S128_UNC_16QAM
      (multi-codeword 16-QAM: codeword pairs interleaved with the existing 2-codeword interleaver).
- Training recipe (identical to `comm/train_comm.py`): folds 1–8, AWGN, SNR per batch from {−5,0,5,10,15,20} dB,
  repetition (m, r) resampled per batch from the N-specific set, Adam 1e-3, wd 1e-4, batch 64, class-weighted BCE,
  max 40 epochs, patience 8, early stopping on fold-9 mean macro-AUROC over the training SNRs (and (m, r)). SSCC heads:
  linear 32→5 on noiseless dequantized embeddings, 60 epochs, patience 10.
- Policies per N and channel: the same fold-9 rules as §3.
- Checks (fold 9): JSCC tx output shape [n, N] complex and energy N per record; repetition energy N; SSCC energy (mean N,
  range reported); N=16 code path with the existing checkpoints reproduces the cached logits; the multi-codeword SSCC
  encoders decode noiselessly to the quantizer indices (bit-exact).
- Report: per N the policy-mean macro-AUROC/AUPRC of each method (curve), per seed; within-N contrasts jscc_e2e −
  rep_orig[policy], jscc_e2e − rep_ctrl[policy], jscc_ctrl − rep_ctrl[policy], sscc[lookup] − rep_ctrl[policy], with
  the §3 bootstrap; a table per method: data uses, pilots (0), mean total energy, information bits (SSCC), whether
  retrained or frozen transfer, not-applicable configurations.

## 5. Experiment 4 — PTB-XL signal-quality annotations and lead selection

- Fields: `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems` of `ptbxl_database.csv`
  (PTB-XL 1.0.3). `extra_beats` and `pacemaker` are not signal-quality fields and are not used.
- Parsing (rule fixed now): split on commas/whitespace; lead tokens I, II, III, aVR, aVL, aVF, V1–V6 (case-insensitive,
  trailing "?" removed); ranges "A-B" expand in the standard order I, II, III, aVR, aVL, aVF, V1…V6; "alles" = all 12;
  other words (leicht, mittel, stark, noisy recording, …) carry no lead information. A field is lead-resolved if at least
  one lead was parsed; otherwise lead-unresolved. Counts of each case are reported per split.
- Record groups: **annotated** (any of the four fields non-empty) vs **not annotated**. Not annotated is not called
  clean.
- Models (frozen, cached predictions): selector (orig_A), matched fixed V5 (orig_C), fixed Lead II (orig_II); seeds 0–2.
- Report per group: records, patients, positives per class; macro and per-class AUROC/AUPRC per model (seed mean and
  per seed); paired differences selector − V5 and selector − II, and the interaction (annotated − not annotated) of
  selector − V5; 2000 patient resamples (seed 20261110) drawn within each group (interaction: draws of the whole fold,
  then split by group); per-seed difference then mean. Exploratory, no multiplicity claim.
- Lead level (records with at least one lead-resolved field; fold-specific): does the selector avoid flagged leads?
  For each record i with flagged set F_i (1 ≤ |F_i| ≤ 11), observed O_i = 1[selected lead ∈ F_i];
  expected E_i = Σ_{ℓ∈F_i} p̂_s(ℓ), where p̂_s is seed s's selection frequency among the not-annotated records of the
  same fold (controls for the selector's general lead preferences). Statistic: mean(O − E) per seed, then mean over
  seeds; patient bootstrap. Also the V5 flag rate (fixed baseline) and |F|/12 (random choice), descriptive. Records
  with all 12 leads flagged are counted but excluded from this statistic. Diagnosis also drives the choice, so this is
  descriptive; no claim that the model localizes noise.
- If only record-level information existed, no lead-level analysis would be done; here lead-level text exists for part of
  the annotations (counts reported).
- Fold 9 first (implementation), then fold 10.

## 6. Not done in this round

- No full parameter grids; no tuning toward any published AUROC.
- Lead-aware backbone: training completed its fixed 40-epoch schedule for all three seeds without early stopping
  (best epochs 39, 37, 38; `revision_audit/backbone/runs/new_seed*/train_result.json`). No longer training is run in this
  round; the Limitations keep the possible undertraining.
- MACs are not converted into device energy. BLE is used only for an explanation of units with official sources
  (analog complex symbols vs modulated bits vs information bits vs protocol overhead); no BLE link is claimed.

## 7. Outputs

`revision2/`: this plan, `plan_hashes/`, code by experiment (`repro/`, `noiseless/`, `channels/`, `budget/`, `quality/`),
`logs/` (with `ISSUES.md`: every failure and fix), `results/` (JSON summaries), `figures/`, `SUMMARY.md`. Logit caches
in `/scratch/pz28/cache/revision2/` with a sha256 list in `results/`. No git commit, no push.
