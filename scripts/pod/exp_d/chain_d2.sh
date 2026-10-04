#!/usr/bin/env bash
# Experiment D, A: format-tolerant slots (Dream, one canvas). EST: the 10-03 length estimates.
source /workspace/aamas2_env.sh
L=results/CHAIN_D_TIMES
export EST=results/release_2026-10-03/results/dream/length_estimate.jsonl
echo "D2 start $(date -u +%T)" >> $L
bash scripts/run_minimal.sh tolerant_dream > results/log_tolerant_dream.txt 2>&1
echo "D2 tolerant_dream exit=$? $(date -u +%T)" >> $L
touch results/CHAIN_D2_DONE
