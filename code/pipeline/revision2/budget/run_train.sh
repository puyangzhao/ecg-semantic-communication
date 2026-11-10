#!/bin/bash
# PLAN §4: train every retrained model at N = 8 and 32 (3 seeds), 24 jobs in parallel, CPU
cd "$(dirname "$0")/../.."
PY=${PY:-python}
for m in sscc_heads rep_ctrl jscc_ctrl jscc_e2e; do for n in 8 32; do for s in 0 1 2; do echo "$m $n $s"; done; done; done | \
  CUDA_VISIBLE_DEVICES="" xargs -P 24 -n 3 sh -c "$PY revision2/budget/train_n.py --method \$0 --n \$1 --seed \$2 --threads 2 > revision2/logs/budget/train_\$0_n\$1_seed\$2.log 2>&1"
echo exit=$? > revision2/logs/budget/run_train.exit
