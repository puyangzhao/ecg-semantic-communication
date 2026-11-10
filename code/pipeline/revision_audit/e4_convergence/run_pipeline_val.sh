#!/bin/bash
# E4 validation-only pipeline: waits for the three 120-epoch encoders, then embeddings (train, val) ->
# heads (proposed, global, random, 12 fixed leads) per seed -> E4 best-lead lock -> fold-9 evaluation.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=""
E=revision_audit/e4_convergence
for s in 0 1 2; do while [ ! -f $E/runs/new_seed${s}_e120/DONE ]; do sleep 60; done; done
echo "$(date -Is) backbones done"
for s in 0 1 2; do $PY $E/downstream_e4.py embed --seed $s --threads 16 > $E/logs/embed_seed$s.log 2>&1 || exit 1; done
echo "$(date -Is) embeddings done"
for s in 0 1 2; do
  echo "$s proposed -"; echo "$s global_learned_k -"; echo "$s random_k -"
  for L in I II III aVR aVL aVF V1 V2 V3 V4 V5 V6; do echo "$s fixed_lead $L"; done
done | xargs -P 12 -n 3 sh -c 'if [ "$2" = "-" ]; then A=""; else A="--lead $2"; fi; '"$PY"' '"$E"'/downstream_e4.py head --seed $0 --method $1 $A --threads 2 > '"$E"'/logs/head_s$0_$1_$2.log 2>&1'
echo "$(date -Is) heads done"
$PY $E/downstream_e4.py lock > $E/logs/lock.log 2>&1 || exit 1
$PY $E/eval_e4.py --split val > $E/logs/eval_val.log 2>&1 || exit 1
echo "$(date -Is) val pipeline done"; echo exit=0 > $E/logs/run_pipeline_val.exit
