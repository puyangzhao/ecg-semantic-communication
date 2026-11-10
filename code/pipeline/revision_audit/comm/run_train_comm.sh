#!/bin/bash
cd "$(dirname "$0")/../.."
PY=${PY:-python}
# leads first (deterministic, cached), then models in parallel
for s in 0 1 2; do CUDA_VISIBLE_DEVICES="" $PY -c "
import sys; sys.path.insert(0,'revision_audit/comm'); sys.path.insert(0,'.')
from train_comm import proposed_leads
for sp in ('train','val'): proposed_leads($s, sp)"; done
for m in sscc_heads rep_ctrl jscc_ctrl jscc_e2e; do for s in 0 1 2; do echo "$m $s"; done; done | \
  CUDA_VISIBLE_DEVICES="" xargs -P 6 -n 2 sh -c "$PY revision_audit/comm/train_comm.py --method \$0 --seed \$1 --threads 3 > revision_audit/comm/logs/train_\$0_seed\$1.log 2>&1"
echo exit=$? > revision_audit/comm/logs/run_train_comm.exit
