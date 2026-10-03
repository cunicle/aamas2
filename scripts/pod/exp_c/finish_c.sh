#!/usr/bin/env bash
# Wait for the three chains, build experiment C's tables, pack the results, leave time to pull the
# archive, then stop the pod from inside (its own env has RUNPOD_API_KEY / RUNPOD_POD_ID).
source /workspace/aamas2_env.sh
for m in CHAIN_C1_DONE CHAIN_C2_DONE CHAIN_C3_DONE; do until [ -f results/$m ]; do sleep 60; done; done
bash scripts/run_minimal.sh summary_c > results/log_summary_c.txt 2>&1
echo "summary_c exit=$? $(date -u +%T)" >> results/CHAIN_C_TIMES
tar czf /workspace/exp_c_results.tar.gz --exclude=results/release_2026-10-03 results && touch /workspace/RESULTS_PACKED
sleep 2700   # pull /workspace/exp_c_results.tar.gz within 45 minutes of RESULTS_PACKED
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs)
runpodctl stop pod $RUNPOD_POD_ID
