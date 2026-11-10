#!/bin/bash
# fold-9 runs of PLAN §3/§3b (checks, block Rician, time variation), 3 seeds in parallel
cd "$(dirname "$0")/../.."
PY=${PY:-python}
L=revision2/logs/channels
for s in 0 1 2; do
  for mode in "--check-k0" "" "--timevar"; do
    tag=$(echo "val_seed${s}${mode}" | tr -d ' -')
    CUDA_VISIBLE_DEVICES="" $PY revision2/channels/eval_channels.py --split val --seed $s --threads 6 $mode > $L/$tag.log 2>&1 &
  done
done
wait
echo done
