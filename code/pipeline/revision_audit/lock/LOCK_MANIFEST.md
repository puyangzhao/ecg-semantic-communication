# Revision lock manifest

Created (UTC): 2026-11-04T08:07:23.123298+00:00  
git HEAD: 51637d62575bc01f589f3a7585d695f17c9c01d2 (revision_audit/ is untracked at lock time; every script and artifact below is pinned by sha256)

PTB-XL fold 10 was used once by the original submission. This is an evaluation of the locked revision protocol on the original held-out test fold.

## Seeds
```
{
 "selector_backbone_comm": [
  0,
  1,
  2
 ],
 "gate_locked": [
  0,
  3,
  5
 ],
 "gate_selection_rule": "fold-9 (gate_eq - margin) at 50% rank-matched > 0 for both CD and STTC AUPRC (6 -> 3); see revision_audit/gate_lock/gate_lock.json",
 "gate_sensitivity_all_six": "seeds 0-5 also evaluated on fold 10, reported as a pre-specified sensitivity analysis of the selection"
}
```

## Frozen choices
```
{
 "new_backbone_best_fixed_lead": {
  "rule": "max seed-mean fold-9 K=1 macro-AUROC of separately trained fixed-lead heads; ties -> AUPRC -> lead order",
  "locked_best_lead": "V6",
  "ranking_new": [
   {
    "lead": "V6",
    "auroc_mean": 0.8744255926575747,
    "auprc_mean": 0.704408335266499,
    "auroc_per_seed": [
     0.8752644877122554,
     0.8722224288771487,
     0.8757898613833198
    ]
   },
   {
    "lead": "V5",
    "auroc_mean": 0.8707600124380964,
    "auprc_mean": 0.7094297911940778,
    "auroc_per_seed": [
     0.873922479291941,
     0.8673700066580343,
     0.8709875513643137
    ]
   },
   {
    "lead": "aVR",
    "auroc_mean": 0.8578925474461929,
    "auprc_mean": 0.6629988822380222,
    "auroc_per_seed": [
     0.8588827490499327,
     0.8590414660938016,
     0.8557534271948445
    ]
   },
   {
    "lead": "II",
    "auroc_mean": 0.8558974518969144,
    "auprc_mean": 0.6726420564189565,
    "auroc_per_seed": [
     0.8591106426855845,
     0.8515533006084695,
     0.8570284123966893
    ]
   },
   {
    "lead": "V4",
    "auroc_mean": 0.8403204745697561,
    "auprc_mean": 0.6427006015012678,
    "auroc_per_seed": [
     0.8435553755025385,
     0.8364667314757448,
     0.8409393167309849
    ]
   },
   {
    "lead": "I",
    "auroc_mean": 0.8273423847834174,
    "auprc_mean": 0.6252981184197187,
    "auroc_per_seed": [
     0.8255243749142711,
     0.8259270045626896,
     0.8305757748732912
    ]
   },
   {
    "lead": "aVF",
    "auroc_mean": 0.8172751467107758,
    "auprc_mean": 0.6086419091989111,
    "auroc_per_seed": [
     0.8172657919151824,
     0.8141638555635928,
     0.8203957926535521
    ]
   },
   {
    "lead": "V3",
    "auroc_mean": 0.8080090939964935,
    "auprc_mean": 0.6029678018444277,
    "auroc_per_seed": [
     0.8069271839851344,
     0.8054455438398257,
     0.8116545541645201
    ]
   },
   {
    "lead": "V1",
    "auroc_mean": 0.8004060977540145,
    "auprc_mean": 0.5913191261435958,
    "auroc_per_seed": [
     0.8013997999311402,
     0.797949779020069,
     0.8018687143108343
    ]
   },
   {
    "lead": "V2",
    "auroc_mean": 0.7893531624498816,
    "auprc_mean": 0.5772890960215541,
    "auroc_per_seed": [
     0.788170115922247,
     0.7875765029654209,
     0.7923128684619771
    ]
   },
   {
    "lead": "aVL",
    "auroc_mean": 0.7880825038476541,
    "auprc_mean": 0.5665719869111198,
    "auroc_per_seed": [
     0.7872215172886964,
     0.7847569128416911,
     0.792269081412575
    ]
   },
   {
    "lead": "III",
    "auroc_mean": 0.7793615051031901,
    "auprc_mean": 0.5557982913556468,
    "auroc_per_seed": [
     0.7773831328682075,
     0.7786863938301456,
     0.7820149886112171
    ]
   }
  ],
  "ranking_orig_descriptive": [
   {
    "lead": "V5",
    "auroc_mean": 0.8409418731769064,
    "auprc_mean": 0.662552495845513,
    "auroc_per_seed": [
     0.8407794068697468,
     0.8414239352195441,
     0.8406222774414284
    ]
   },
   {
    "lead": "V6",
    "auroc_mean": 0.82965745026236,
    "auprc_mean": 0.6377012198236949,
    "auroc_per_seed": [
     0.82911650061664,
     0.8307986731843799,
     0.8290571769860602
    ]
   },
   {
    "lead": "II",
    "auroc_mean": 0.8274969634125653,
    "auprc_mean": 0.6260763309961351,
    "auroc_per_seed": [
     0.8277443278614133,
     0.8271198707298554,
     0.8276266916464271
    ]
   },
   {
    "lead": "V4",
    "auroc_mean": 0.8147674071196261,
    "auprc_mean": 0.5990755915930107,
    "auroc_per_seed": [
     0.8148705346984325,
     0.8153674300231961,
     0.8140642566372499
    ]
   },
   {
    "lead": "I",
    "auroc_mean": 0.8011065943458515,
    "auprc_mean": 0.5914283141440282,
    "auroc_per_seed": [
     0.8015063252978699,
     0.8009268227090247,
     0.8008866350306599
    ]
   },
   {
    "lead": "aVF",
    "auroc_mean": 0.7810787518313558,
    "auprc_mean": 0.5611964627846101,
    "auroc_per_seed": [
     0.7809965951898814,
     0.7816117384011683,
     0.7806279219030177
    ]
   },
   {
    "lead": "V3",
    "auroc_mean": 0.7629299783176444,
    "auprc_mean": 0.5411546517491942,
    "auroc_per_seed": [
     0.7612124041190366,
     0.7611885564237767,
     0.7663889744101201
    ]
   },
   {
    "lead": "V2",
    "auroc_mean": 0.7475808048907419,
    "auprc_mean": 0.5082277753487131,
    "auroc_per_seed": [
     0.7470486642556876,
     0.7481134323127593,
     0.7475803181037787
    ]
   },
   {
    "lead": "aVR",
    "auroc_mean": 0.7343473233763285,
    "auprc_mean": 0.47123769770308166,
    "auroc_per_seed": [
     0.7370880362737646,
     0.73383021648506,
     0.7321237173701608
    ]
   },
   {
    "lead": "aVL",
    "auroc_mean": 0.7191814634662407,
    "auprc_mean": 0.4628836485340703,
    "auroc_per_seed": [
     0.7201824348963397,
     0.7198961036021799,
     0.7174658519002023
    ]
   },
   {
    "lead": "V1",
    "auroc_mean": 0.713595848907716,
    "auprc_mean": 0.4781572097722863,
    "auroc_per_seed": [
     0.7140363752302242,
     0.7158959088612885,
     0.7108552626316351
    ]
   },
   {
    "lead": "III",
    "auroc_mean": 0.6793520505825912,
    "auprc_mean": 0.43186476278430935,
    "auroc_per_seed": [
     0.6802143177951995,
     0.6761493581671421,
     0.6816924757854321
    ]
   }
  ],
  "note": "V5 stays the pre-locked reference lead (chosen on the original backbone); this backbone's own best is reported separately"
 },
 "subsets": {
  "orig": {
   "1": [
    "V5"
   ],
   "3": [
    "aVF",
    "V1",
    "V5"
   ],
   "6": [
    "I",
    "aVR",
    "aVF",
    "V1",
    "V2",
    "V5"
   ]
  },
  "new": {
   "1": [
    "V6"
   ],
   "3": [
    "aVF",
    "V2",
    "V5"
   ],
   "6": [
    "I",
    "II",
    "III",
    "V1",
    "V2",
    "V5"
   ]
  }
 },
 "thresholds": "revision_audit/thresholds/frozen_thresholds.json (Youden J on fold 9, ties -> lowest threshold)",
 "comm_policies": "revision_audit/comm/locked_comm_policies.json (fold 9; submitted rep_orig policies unchanged)",
 "external_label_map": "revision_audit/external_ecgarr/label_map_frozen.json",
 "temperature_scaling": "one scalar T per (model, seed) fitted on fold 9 by BCE; per-class Platt exploratory"
}
```

## Primary / key secondary / multiplicity
```
{
 "primary": "PTB-XL fold 10, original backbone, K=1: proposed selector + head (orig_A) minus matched fixed-V5 head (orig_C), macro-AUROC; seeds 0-2; shared patient-level paired bootstrap (2000, rng 20261110), per-seed difference then mean",
 "key_secondary_holm": [
  "orig_A - orig_C macro-AUPRC",
  "CD AUROC",
  "CD AUPRC",
  "STTC AUROC",
  "STTC AUPRC"
 ],
 "multiplicity": "Holm-Bonferroni over the 5 key secondary endpoints using two-sided bootstrap p-values; every other comparison is exploratory with unadjusted 95% intervals and no significance claim",
 "external_primary": "ecg-arrhythmia v1.0.0: orig_A - orig_C for CD and STTC-partial AUROC/AUPRC, record-level bootstrap (no patient identifier); reported separately from PTB-XL"
}
```

## Exploratory

- orig_A - orig_B (same head, lead swapped); orig_B - orig_C; orig_A vs fixed II and global ranking
- revision backbone: 12-lead performance vs original architecture; new_A vs new_C / new_BEST / new_II / new_GL; new_A - orig_A (whole-model comparison)
- K=3/K=6 enumerated best fixed subset vs clinical subset vs random per-record vs selector choice (same subset-agnostic receiver), both backbones
- K=1 label-assisted oracle (mean BCE primary statistic)
- frozen development-set operating points: TP/FP/TN/FN, sensitivity, specificity with intervals
- temperature scaling and Platt (exploratory) calibration; descriptive subgroups (sex, age bins)
- gate (seeds 0,3,5): 50% rank-matched quota vs margin and vs matched random; frozen-threshold execution and actual switch rate; all-six-seed sensitivity
- communication: rep_orig (submitted), rep_ctrl, SSCC (5 configs + lookup), JSCC ctrl, JSCC e2e; AWGN, block Rayleigh perfect CSI, block Rayleigh LS-2 pilots; locked policies vs best fixed
- external: orig_A - orig_B and orig_B - orig_C. The gate is NOT evaluated externally: its training target (mean BCE gain over all five PTB-XL classes) is not interpretable when only CD and STTC-partial are mapped (decided before any external model output)

## Software-error rule

if a software error that affects a conclusion is found after this lock, the pre-correction result is kept and reported next to the corrected one with the fixing change; nothing is tuned on fold 10 or external results

Artifact hashes: 238 files, see LOCK_MANIFEST.json.
