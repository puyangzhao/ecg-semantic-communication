#!/bin/bash
# PLAN §0 step 3: hash the plan, all revision2 code, the fold-9 locked policies, fold-9 results and checks.
set -e
cd "$(dirname "$0")"
{ ls PLAN.md r2common.py r2comm.py make_freeze.sh run_fold10.sh
  find repro noiseless channels budget quality -name "*.py" -o -name "*.sh" | grep -v __pycache__ | sort
  find results -name "*val*.json" | sort
  ls results/channels/locked_policies_rician.json results/budget/locked_policies_budget.json
  find results/checks -name "*.json" | sort
  ls results/repro_key_numbers.json
} > plan_hashes/frozen_files.txt
xargs sha256sum < plan_hashes/frozen_files.txt > plan_hashes/FROZEN.sha256
date -u +%FT%TZ > plan_hashes/FROZEN.time
echo "frozen $(wc -l < plan_hashes/FROZEN.sha256) files at $(cat plan_hashes/FROZEN.time)"
