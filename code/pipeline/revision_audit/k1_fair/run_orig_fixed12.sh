#!/bin/bash
# 12 separately trained fixed-K=1-lead heads x seeds 0-2 on the submitted (orig) backbone
cd "$(dirname "$0")/../.."
PY=${PY:-python}
for s in 0 1 2; do for L in I II III aVR aVL aVF V1 V2 V3 V4 V5 V6; do
  [ -f revision_audit/k1_fair/runs/orig_fixed_lead_${L}_seed${s}/DONE ] || echo "$L $s"
done; done | CUDA_VISIBLE_DEVICES="" xargs -P 8 -n 2 sh -c "$PY revision_audit/k1_fair/train_heads.py --backbone orig --method fixed_lead --lead \$0 --seed \$1 --threads 2 > revision_audit/k1_fair/logs/orig_fixed_\$0_seed\$1.log 2>&1"
echo exit=$? > revision_audit/k1_fair/logs/run_orig_fixed12.exit
