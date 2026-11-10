#!/bin/bash
# Formal evaluation of the locked revision protocol. Run ONCE, after revision_audit/lock/make_lock.py.
# Every step logs to revision_audit/formal_eval/logs/<step>.log and records its exit code in STEPS.log.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=""
L=revision_audit/formal_eval/logs
[ -f revision_audit/lock/LOCK_MANIFEST.json ] || { echo "no lock"; exit 1; }
step() { name=$1; shift; echo "$(date -Is) START $name" >> $L/STEPS.log; "$@" > $L/$name.log 2>&1; rc=$?; echo "$(date -Is) END $name exit=$rc" >> $L/STEPS.log; return 0; }
step cache_test $PY revision_audit/backbone/cache_waveforms.py --splits test --allow-test
for s in 0 1 2; do step embed_new_test_s$s $PY revision_audit/backbone/embed.py --seed $s --splits test; done
# independent CPU steps in parallel
( step k1_eval $PY revision_audit/formal_eval/k1_eval.py --split test ) &
( step subset_orig $PY revision_audit/subset_enum/enumerate_subsets.py --backbone orig --split test; step subset_new $PY revision_audit/subset_enum/enumerate_subsets.py --backbone new --split test ) &
( step gate_eval $PY revision_audit/gate_lock/gate_eval.py --split test --seeds 0,1,2,3,4,5; \
  step gate_boot_locked3 $PY revision_audit/k1_gate_pilot/multiseed_bootstrap.py --root revision_audit/gate_lock/results/test --seeds 0,3,5 --out locked3; \
  step gate_boot_all6 $PY revision_audit/k1_gate_pilot/multiseed_bootstrap.py --root revision_audit/gate_lock/results/test --seeds 0,1,2,3,4,5 --out all6 ) &
( for s in 0 1 2; do step comm_eval_s$s $PY revision_audit/comm/eval_comm.py --split test --seed $s --threads 6; done; step comm_report $PY revision_audit/comm/comm_report.py --split test --B 1000 --workers 24 ) &
( step heatmap $PY revision_audit/clinical/selection_heatmap.py --split test; step backbone $PY revision_audit/backbone/eval_backbone.py --split test; step external $PY revision_audit/external_ecgarr/external_eval.py ) &
wait
step figures $PY revision_audit/deliverables/make_figures.py --split test
echo "$(date -Is) ALL DONE" >> $L/STEPS.log; echo done > $L/ALL.DONE
