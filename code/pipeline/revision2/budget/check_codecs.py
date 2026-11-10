#!/usr/bin/env python
"""PLAN §4 checks (fold 9 + training folds only):
  1. N = 16: SSCCN with the existing seed-0 codebooks reproduces comm/sscc.SSCC.tx and .rx bit for bit;
  2. every N = 8 / 32 configuration: noiseless transmit -> receive returns exactly the quantizer reconstruction
     (on a random quantizer fitted to 2000 training embeddings; bit-exact indices), and QPSK energy is exactly N;
  3. RepSchemeN(n=16) equals eval_comm.RepScheme; JSCCN(384, 16) loads the existing checkpoint and equals JSCC.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import r2common as C  # noqa: E402
import eval_comm as EC  # noqa: E402
import sscc as S  # noqa: E402
from comm_models import JSCC  # noqa: E402
from models_n import JSCCN, RepSchemeN  # noqa: E402
from sscc_n import CONFIGS_N, SSCCN  # noqa: E402
from train_comm import EMB, proposed_leads  # noqa: E402


def main():
    out = {}
    va = torch.load(os.path.join(EMB, "val.pt"))
    lead = proposed_leads(0, "val")
    z1 = va["z"][torch.arange(len(lead)), torch.from_numpy(lead)].numpy()
    rng = np.random.default_rng(0)
    for name in S.CONFIGS:
        st = pickle.load(open(os.path.join(C.RA, f"comm/runs/sscc_heads_seed0/codec_{name}.pkl"), "rb"))
        old, new = S.SSCC(name), SSCCN(name, 16)
        old.pq.books = new.pq.books = st["books"]
        old.scale = new.scale = st["scale"]
        xo, xn = old.tx(z1), new.tx(z1)
        y = (xo[None] + 0.5 * (rng.standard_normal(xo.shape) + 1j * rng.standard_normal(xo.shape)).astype(np.complex64))
        nv = np.full((1, len(z1), 1), 0.5, np.float32)
        out[f"n16_{name}"] = bool(np.array_equal(xo, xn) and np.array_equal(old.rx(y, nv), new.rx(y, nv)))
    tr = torch.load(os.path.join(EMB, "train.pt"))
    ltr = proposed_leads(0, "train")
    z1tr = tr["z"][torch.arange(len(ltr)), torch.from_numpy(ltr)].numpy()[:2000]
    for n in (8, 32):
        for name in CONFIGS_N[n]:
            sc = SSCCN(name, n, seed=0).fit(z1tr)
            x = sc.tx(z1[:500])
            idx = sc.pq.encode(z1[:500])
            zh = sc.rx(x[None], np.full((1, 500, 1), 1e-6, np.float32))[0]
            e = (np.abs(x) ** 2).sum(1)
            ok = np.array_equal(zh, sc.pq.decode(idx)) and x.shape == (500, n)
            if sc.cfg["mod"] == "QPSK":
                ok = ok and np.allclose(e, n, atol=1e-4)
            out[f"n{n}_{name}"] = bool(ok)
            out[f"n{n}_{name}_energy_mean"] = float(e.mean())
    # repetition and JSCC at N = 16
    M = EC.load_all(0, "val")[1]
    sch, zz = M["rep_ctrl"]
    gen = RepSchemeN(sch.scores, sch.head, 16)
    ok = True
    for r in (1, 2, 4):
        xo, io = sch.tx(zz, r)
        xn, i_n = gen.tx(zz, r)
        xh = xo[None] + 0.1 * (rng.standard_normal(xo.shape) + 1j * rng.standard_normal(xo.shape)).astype(np.complex64)
        ok = ok and np.array_equal(xo, xn) and np.array_equal(sch.rx(xh, io, r), gen.rx(xh, i_n, r))
    out["rep_n16_equal"] = bool(ok)
    sd = torch.load(os.path.join(C.RA, "comm/runs/jscc_e2e_seed0/best.pt"))
    a, b = JSCC(384), JSCCN(384, 16)
    a.load_state_dict(sd)
    b.load_state_dict(sd)
    a.eval(), b.eval()
    src = va["z"].reshape(len(lead), -1)
    with torch.no_grad():
        out["jscc_n16_equal"] = bool(torch.equal(a.rx(a.tx(src)), b.rx(b.tx(src))))
    out["pass"] = all(v for k, v in out.items() if not k.endswith("energy_mean"))
    C.dump(out, os.path.join(C.R2, "results", "checks", "budget_codecs.json"))
    print(out)


if __name__ == "__main__":
    main()
