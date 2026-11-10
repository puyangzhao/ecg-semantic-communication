#!/usr/bin/env python
"""PLAN §3b check: rho = 1 logits must equal the block Rician 3 dB LS-2 logits (same split, seed, method, SNR)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2common as C  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "val"
worst = 0.0
for s in C.SEEDS:
    base = os.path.join(C.CACHE, "channels", split, f"seed{s}")
    for f in os.listdir(os.path.join(base, "rician3dB_ls2_rho1.0")):
        a = np.load(os.path.join(base, "rician3dB_ls2_rho1.0", f))
        b = np.load(os.path.join(base, "rician3dB_ls2", f))
        worst = max(worst, float(np.abs(a - b).max()))
C.dump({"split": split, "worst_abs_dlogit": worst, "pass": worst <= 1e-5}, os.path.join(C.R2, "results", "checks", f"timevar_rho1_{split}.json"))
print("rho=1 vs block Rician 3 dB LS-2: worst", worst)
