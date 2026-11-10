#!/bin/bash
# Code check of the training and evaluation entry points on a small patient-disjoint subset of PTB-XL
# folds 1-8 (fold 9 and fold 10 are not read). Everything is written under $OUT; no project file is
# written. The output is a software check only and is not used by any reported result.
#   PY=python OUT=/tmp/ecg_code_check checks/run_code_check.sh      (run from the pipeline folder)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${OUT:-/tmp/ecg_code_check}
mkdir -p "$OUT"
$PY checks/make_check_split.py --out "$OUT"
sed -e "s#^embeddings_dir: .*#embeddings_dir: $OUT/emb#" -e 's/num_epochs: [0-9]*/num_epochs: 2/' \
    -e 's/device: cuda/device: cpu/' -e 's/n_bootstrap: [0-9]*/n_bootstrap: 50/' configs/phase2a.yaml > "$OUT/phase2a_check.yaml"
for m in proposed fixed_clinical_k; do
  CUDA_VISIBLE_DEVICES="" $PY scripts/train_phase2a.py --method $m --seed 0 --config "$OUT/phase2a_check.yaml" \
      --run-name check_$m --outputs-dir "$OUT/outputs" > "$OUT/check_$m.log" 2>&1
  echo "$m: $(grep -c '^\[epoch' "$OUT/check_$m.log") epochs; $(grep '^Test K=1' "$OUT/check_$m.log")"
done
echo "outputs in $OUT"
