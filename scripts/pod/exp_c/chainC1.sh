#!/usr/bin/env bash
# Dream agent teams: C1 (team-symmetric requests, five protocols), C2 with oracle lengths, C2 swapped.
source /workspace/aamas2_env.sh
L=results/CHAIN_C_TIMES
echo "C1 start $(date -u +%T)" >> $L
for ph in agents_bfcl_dream agents_bfcl_swap; do
  bash scripts/run_minimal.sh $ph > results/log_$ph.txt 2>&1; echo "C1 $ph exit=$? $(date -u +%T)" >> $L
done
touch results/CHAIN_C1_DONE
