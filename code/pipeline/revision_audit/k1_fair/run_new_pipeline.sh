#!/bin/bash
# Waits for the three revision-backbone trainings, then: embeddings (train, val) -> Phase-2A-recipe heads
# (proposed, global, random, 12 fixed leads) per backbone seed -> best-lead lock -> subset enumeration -> thresholds.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=""
for s in 0 1 2; do while [ ! -f revision_audit/backbone/runs/new_seed$s/DONE ]; do sleep 30; done; done
echo "$(date -Is) backbones done"
for s in 0 1 2; do $PY revision_audit/backbone/embed.py --seed $s --splits train,val --threads 16 || exit 1; done
echo "$(date -Is) embeddings done"
for s in 0 1 2; do
  echo "new_s$s proposed - $s"; echo "new_s$s global_learned_k - $s"; echo "new_s$s random_k - $s"
  for L in I II III aVR aVL aVF V1 V2 V3 V4 V5 V6; do echo "new_s$s fixed_lead $L $s"; done
done | xargs -P 12 -n 4 sh -c 'if [ "$2" = "-" ]; then A=""; else A="--lead $2"; fi; '"$PY"' revision_audit/k1_fair/train_heads.py --backbone $0 --method $1 $A --seed $3 --threads 2 > revision_audit/k1_fair/logs/$0_$1_$2_seed$3.log 2>&1'
echo "$(date -Is) heads done"
$PY revision_audit/k1_fair/lock_new_best_lead.py || exit 1
$PY revision_audit/subset_enum/enumerate_subsets.py --backbone new --split val > revision_audit/subset_enum/results/new_val.log 2>&1 || exit 1
$PY revision_audit/thresholds/freeze_thresholds.py || exit 1
$PY revision_audit/backbone/eval_backbone.py --split val > revision_audit/backbone/results/backbone_val.log 2>&1 || exit 1
echo "$(date -Is) pipeline done"; echo exit=0 > revision_audit/k1_fair/logs/run_new_pipeline.exit
