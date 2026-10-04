#!/usr/bin/env bash
# Wait for the chains, build the quick table, pack the results, leave time to pull the archive,
# then stop the pod from inside (its own env has RUNPOD_API_KEY / RUNPOD_POD_ID).
source /workspace/aamas2_env.sh
for m in ${MARKERS:-CHAIN_D1_DONE CHAIN_D2_DONE}; do until [ -f results/$m ]; do sleep 60; done; done
mkdir -p results/summary
$PY_DREAM scripts/exp_d_quick.py results/dream/bfcl_tolerant*.jsonl results/*/agents_d.jsonl \
    results/*/agents_pos.jsonl > results/summary/exp_d_quick.md 2> results/log_exp_d_quick.txt
echo "exp_d_quick exit=$? $(date -u +%T)" >> results/CHAIN_D_TIMES
tar czf /workspace/exp_d_results.tar.gz --exclude=results/release_2026-10-03 results && touch /workspace/RESULTS_PACKED
sleep 2700   # pull /workspace/exp_d_results.tar.gz within 45 minutes of RESULTS_PACKED
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs)
runpodctl stop pod $RUNPOD_POD_ID
