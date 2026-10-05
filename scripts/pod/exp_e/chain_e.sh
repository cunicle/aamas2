#!/usr/bin/env bash
# Experiment E main run, one phase per chain: bash chain_e.sh dream | llada2. Waits for gates 1-2
# (gate_e.sh) and runs only if they passed. Two chains run side by side on one 80GB card.
source /workspace/aamas2_env.sh
P=$1; L=results/log_e_times.txt
until [ -f /tmp/gate_e/GATE_DONE ]; do sleep 20; done
[ -f /tmp/gate_e/GATE_PASS ] || { echo "$P: gates failed, not run $(date -u +%T)" >> $L; exit 1; }
echo "fill_fig3_$P start $(date -u +%T)" >> $L
bash scripts/run_minimal.sh fill_fig3_$P 2>&1 | tee results/log_e_$P.txt > /dev/null
echo "fill_fig3_$P exit=${PIPESTATUS[0]} $(date -u +%T)" >> $L
touch results/CHAIN_E_${P^^}_DONE
