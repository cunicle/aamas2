# Template (not used on 10-03; that pod idled ~7 h after the last chain): wait for every
# chain, build the summary tables, pack the results, give the laptop time to pull the
# archive, then stop the pod from inside (the pod's own env has RUNPOD_API_KEY / RUNPOD_POD_ID).
source /workspace/aamas2_env.sh; cd /workspace/aamas2
for m in CHAIN_D_DONE CHAIN_L_DONE; do until [ -f results/$m ]; do sleep 60; done; done   # list every marker
bash scripts/run_minimal.sh summary_length > results/log_summary_length.txt 2>&1
tar czf /workspace/results_final.tar.gz results && touch /workspace/RESULTS_PACKED
sleep 1800   # pull /workspace/results_final.tar.gz within 30 minutes of RESULTS_PACKED
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs)
runpodctl stop pod $RUNPOD_POD_ID
