#!/usr/bin/env bash
# Experiment D, B: position agents (Qwen then Dream; BFCL sym 104 and choose-N 105).
source /workspace/aamas2_env.sh
L=results/CHAIN_D_TIMES
echo "D1 start $(date -u +%T)" >> $L
bash scripts/run_minimal.sh position_agents > results/log_position_agents.txt 2>&1
echo "D1 position_agents exit=$? $(date -u +%T)" >> $L
touch results/CHAIN_D1_DONE
