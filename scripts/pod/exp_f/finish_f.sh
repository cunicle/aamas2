#!/usr/bin/env bash
# Experiment F: wait for the chain, gate 4 (complete, no error field, the right requests and
# texts), the quick read-out, then pack (the release archive with exactly the runbook's files, and
# a full copy of results/ as a backup), leave time to pull, and stop the pod from inside.
source /workspace/aamas2_env.sh
until [ -f results/CHAIN_F_DONE ]; do sleep 30; done
L=results/log_f_times.txt
$PY_DREAM scripts/pod/exp_f/gate4_check_f.py > /tmp/gate_f/gate4_check.txt 2>&1
echo "gate4_check exit=$? $(date -u +%T)" >> $L
mkdir -p results/summary
$PY_DREAM scripts/exp_f_quick.py --results results > results/summary/exp_f_quick.md 2> results/log_f_quick_stderr.txt
echo "exp_f_quick exit=$? $(date -u +%T)" >> $L
rm -rf results/gate_f && cp -r /tmp/gate_f results/gate_f
D=$(TZ=Asia/Shanghai date +%F)
mkdir -p release
tar czf release/exp_f_$D.tar.gz results/qwen/agents_f1.jsonl results/qwen/agents_f_choose.jsonl \
    results/dream/agents_f1.jsonl results/dream/agents_f_choose.jsonl \
    results/summary/exp_f_quick.md results/log_f_*.txt results/gate_f
cp release/exp_f_$D.tar.gz /workspace/exp_f_release.tar.gz
tar czf /workspace/exp_f_results_full.tar.gz results
echo "packed exp_f_$D.tar.gz $(date -u +%T)" >> $L
touch /workspace/RESULTS_PACKED
sleep 2700   # the laptop pulls and stops within 45 minutes of RESULTS_PACKED; otherwise stop here
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs)
runpodctl stop pod $RUNPOD_POD_ID
