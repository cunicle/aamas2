#!/usr/bin/env bash
# Experiment F main run: waits for gates 1-2 (gate_f.sh) and runs only if they passed. Qwen2.5
# first (fast), then Dream; one process at a time (one 7B process already saturates the card).
source /workspace/aamas2_env.sh
L=results/log_f_times.txt
until [ -f /tmp/gate_f/GATE_DONE ]; do sleep 20; done
if [ -f /tmp/gate_f/GATE_PASS ]; then
  for p in ar dream; do
    echo "wording_$p start $(date -u +%T)" >> $L
    bash scripts/run_minimal.sh wording_$p 2>&1 | tee results/log_f_$p.txt > /dev/null
    echo "wording_$p exit=${PIPESTATUS[0]} $(date -u +%T)" >> $L
  done
else
  mkdir -p results; echo "gates failed, not run $(date -u +%T)" >> $L
fi
touch results/CHAIN_F_DONE
