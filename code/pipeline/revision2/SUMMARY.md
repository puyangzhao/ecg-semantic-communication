# Revision 2: additional revision analyses — summary

All analyses are post hoc revision analyses: planned in `PLAN.md` (2026-11-07 23:10 UTC), checked on folds 1–9, frozen
(`plan_hashes/FROZEN.sha256`, 58 files, 23:42:32 UTC) and evaluated once on fold 10 (`run_fold10.sh`, 23:42–00:06 UTC).
Seeds 0–2, all reported. CPU only. Issues and fixes: `logs/ISSUES.md` (3 entries, none changed a protocol).

## Checks (all passed)

| Check | Result | File |
|---|---|---|
| Key locked numbers from cached predictions | primary +0.0175 [0.0086, 0.0268] dev 0; 12-lead 0.914 dev 1.8e-10; joint JSCC 20 dB 0.922 dev 0 | `results/repro_key_numbers.json` |
| Locked transmission report re-derived (fold 10, B=1000; fold 9, B=200) | max dev 2.2e-16 | `results/checks/report_reproduction_test.json` |
| Policy rule reproduces `locked_comm_policies.json` | identical | `results/checks/lock_rule.json` |
| Rician code with K_R=0 vs cached Rayleigh (perfect + LS-2), fold 9 | max abs dlogit 0 | `results/checks/channels_k0_val_seed*.json` |
| Time variation rho=1 vs block Rician 3 dB LS-2 (folds 9, 10) | 0 | `results/checks/timevar_rho1_*.json` |
| Budget code at N=16 vs cached logits (fold 9) | 0 | `results/checks/budget_n16_val_seed*.json` |
| SSCC at N=16 bit-identical; new configs decode noiselessly; QPSK energy exactly N | pass | `results/checks/budget_codecs.json` |
| E|h|^2, LOS fraction, lag-1 correlation | 1.000; 0.666 / 0.909; 0.950 | `results/channels/*_eval_*.json` |
| Joint JSCC: bypass = AWGN noise off = training path = cached noiseless (folds 9, 10, 3 seeds) | max abs dlogit 0 | `results/noiseless/` |

## Results (fold 10, seed mean; 95% patient-level CI)

1. **Joint JSCC without noise.** Same checkpoint: bypass / noise off 0.9217, AWGN 20 dB 0.9216; noise effect −0.0002
   [−0.0002, −0.0001] AUROC, −0.0006 AUPRC. Clean 12-lead mean-pool + linear model 0.9142. Joint JSCC (noise-free) −
   12-lead model +0.0076 [0.0041, 0.0111] AUROC, +0.029 AUPRC: a between-model (architecture) difference, not noise.
   Fold 9: 0.9286 / 0.9285 / 0.9215.
2. **Rician fading** (frozen models). Joint JSCC best at every SNR in all four conditions. Joint JSCC − original
   repetition: 3 dB perfect +0.089, LS +0.074 [0.068, 0.078]; 10 dB perfect +0.093, LS +0.077 [0.072, 0.082].
   JSCC 1 lead − same-lead repetition +0.012/+0.015/+0.013/+0.016; SSCC − same-lead repetition −0.037/−0.014/−0.039/−0.012.
   Adaptive vs best fixed r ≤ 0.0017. Rician > Rayleigh (LS: joint JSCC 0.772/0.783 vs 0.762).
   Time variation (3 dB, LS): rho 0.95 / 0 cost joint JSCC 0.015 / 0.022, repetition 0.013 / 0.020; gap 0.059 → 0.057.
3. **Budget N = 8 / 16 / 32** (energy N). Same method order at every N in AWGN and Rayleigh. AWGN: joint JSCC − original
   repetition +0.091 / +0.095 / +0.105; joint JSCC − same-lead repetition +0.069 / +0.072 / +0.086; JSCC 1 lead − same-lead
   repetition +0.015 / +0.013 / +0.023; SSCC − same-lead repetition −0.065 / −0.038 / −0.043. 16→32: joint JSCC +0.019,
   same-lead repetition +0.005. Original repetition at N≠16 is a frozen transfer.
4. **Signal-quality annotations.** Test: 528 records (490 patients) annotated, 1,670 not; 502 lead-resolved.
   Selector − V5 macro-AUROC +0.0186 [−0.0006, 0.0367] annotated vs +0.0176 [0.0070, 0.0278] not; interaction
   +0.0010 [−0.021, 0.023]; CD gain and STTC loss in both groups. Lead level (409 records with 1–11 named leads): selector
   sent a named lead 15.1% vs 26.6% expected from its unannotated lead frequencies (−11.6 points [−14.4, −8.4]); V5
   named 11.2%. Association only (lead choice also depends on diagnosis).

## Not done (by plan)
- Lead-aware backbone: completed its fixed 40-epoch schedule (best epochs 39/37/38); not trained longer.
- No BLE link, no device-energy conversion of MACs.

## Manuscript (`manuscript/`)
- Main 9 pp (was 8), supplement 12 pp (was 7), letter 17 pp, marked 10 pp (base b229b7b, provisional).
- New: Methods II-D paragraph "Additional channels and budgets", II-G "Signal-Quality Annotations", II-H revision-analysis
  statement; Results III-B quality paragraph, III-C noise-free explanation, Rician and budget paragraphs; Discussion BLE
  sentence; Limitations updated; refs [4] [7] [8] [9] [16]. Supplement S1 units/BLE paragraph, Section S6, Tables S11–S15,
  Figs. S7–S10. "Phase 2A/2B" now named once (II-B) as the original names.
