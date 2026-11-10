#!/bin/bash
# PLAN §0 step 4: the single fold-10 evaluation of every revision2 analysis. Refuses to run unless FROZEN.sha256 matches.
cd "$(dirname "$0")/.."
R=revision2; L=$R/logs/fold10; mkdir -p $L
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=""
(cd $R && sha256sum -c --quiet plan_hashes/FROZEN.sha256) || { echo "frozen hash mismatch, refusing"; exit 1; }
[ -e $L/STARTED ] && { echo "fold-10 run already started once ($L/STARTED); refusing"; exit 1; }
date -u +%FT%TZ > $L/STARTED
step() { n=$1; shift; echo "$(date -u +%FT%TZ) START $n" >> $L/STEPS.log; "$@" > $L/$n.log 2>&1; echo "$(date -u +%FT%TZ) END $n exit=$?" >> $L/STEPS.log; }
for s in 0 1 2; do
  step channels_s$s $PY $R/channels/eval_channels.py --split test --seed $s --threads 6 &
  step timevar_s$s $PY $R/channels/eval_channels.py --split test --seed $s --threads 4 --timevar &
  for n in 8 32; do step budget_n${n}_s$s $PY $R/budget/eval_n.py --split test --seed $s --n $n --threads 4 & done
done
step noiseless env OMP_NUM_THREADS=1 $PY $R/noiseless/noiseless_check.py --split test &
wait
step quality env OMP_NUM_THREADS=1 $PY $R/quality/quality_eval.py --split test
step channels_report env OMP_NUM_THREADS=1 $PY $R/channels/lock_and_report.py --report test
step budget_report env OMP_NUM_THREADS=1 $PY $R/budget/lock_and_report.py --report test
date -u +%FT%TZ > $L/FINISHED
