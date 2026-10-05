#!/usr/bin/env bash
# Experiment E: wait for both chains, gate 4 (old records untouched, exact new line counts, no
# error field), the quick read-out, then pack (the release archive with exactly the runbook's files,
# and a full copy of results/ as a backup), leave time to pull, and stop the pod from inside.
source /workspace/aamas2_env.sh
for m in CHAIN_E_DREAM_DONE CHAIN_E_LLADA2_DONE; do until [ -f results/$m ]; do sleep 30; done; done
L=results/log_e_times.txt
$PY_DREAM scripts/pod/exp_e/gate4_check_e.py > /tmp/gate_e/gate4_check.txt 2>&1
echo "gate4_check exit=$? $(date -u +%T)" >> $L
mkdir -p results/summary
$PY_DREAM scripts/exp_e_quick.py --results results > results/summary/exp_e_quick.md 2> results/log_e_quick_stderr.txt
echo "exp_e_quick exit=$? $(date -u +%T)" >> $L
rm -rf results/gate_e && cp -r /tmp/gate_e results/gate_e
D=$(TZ=Asia/Shanghai date +%F)
mkdir -p release
tar czf release/exp_e_$D.tar.gz results/dream/bfcl_closer.jsonl \
    results/dream/length_prior_closer_s4.jsonl results/dream/length_prior_closer_s4.txt \
    results/qwen/bfcl_surplus.jsonl results/llada2/bfcl_surplus.jsonl \
    results/llada2/length_prior_s2.jsonl results/llada2/length_prior_s2.txt \
    results/log_e_*.txt results/gate_e
cp release/exp_e_$D.tar.gz /workspace/exp_e_release.tar.gz
tar czf /workspace/exp_e_results_full.tar.gz results
echo "packed exp_e_$D.tar.gz $(date -u +%T)" >> $L
touch /workspace/RESULTS_PACKED
sleep 2700   # the laptop pulls and stops within 45 minutes of RESULTS_PACKED; otherwise stop here
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs)
runpodctl stop pod $RUNPOD_POD_ID
