#!/usr/bin/env bash
# Gate 4: C1 sim-anon on all 104 team-symmetric requests, both models, into the main C1 files
# (the C1 phases skip these teams later). Sibling agents must give identical outputs.
source /workspace/aamas2_env.sh
G=results/gate_c; mkdir -p $G results/qwen results/dream
DREAM=Dream-org/Dream-v0-Instruct-7B; QWEN=Qwen/Qwen2.5-7B-Instruct
t() { echo "$1 $(date -u +%T)" >> $G/times.txt; }
t sanity_start
$PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset sym --protocols sim-anon \
    --out results/dream/agents_c1.jsonl > $G/log_sanity_dream.txt 2>&1 &
$PY_DREAM scripts/run_agents_bfcl.py --model $QWEN --backend ar --subset sym --protocols sim-anon \
    --out results/qwen/agents_c1.jsonl > $G/log_sanity_qwen.txt 2>&1 &
wait
t sanity_done
touch $G/SANITY_DONE
