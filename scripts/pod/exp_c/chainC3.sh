#!/usr/bin/env bash
# Qwen agent teams (C1, C2), the original-interface one-sided run of C3 (independent of the
# variant), the variant probe (teacher-forced, unconstrained P(closer)), the choose-N sim-rule
# control and C4. The variant decoding runs (closer_dream_main / _side) wait for the user's call.
source /workspace/aamas2_env.sh
L=results/CHAIN_C_TIMES
DREAM=Dream-org/Dream-v0-Instruct-7B; BFCL=bfcl:parallel,parallel_multiple
echo "C3 start $(date -u +%T)" >> $L
bash scripts/run_minimal.sh agents_bfcl_ar > results/log_agents_bfcl_ar.txt 2>&1
echo "C3 agents_bfcl_ar exit=$? $(date -u +%T)" >> $L
# the same command as in closer_dream_side (which later skips these records)
$PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --length-mode onesided \
    --k 1 --order confidence --out results/dream/bfcl_onesided.jsonl > results/log_onesided_original.txt 2>&1
echo "C3 onesided_original exit=$? $(date -u +%T)" >> $L
for ph in closer_probe agents_rule choose_tau_ltr; do
  bash scripts/run_minimal.sh $ph > results/log_$ph.txt 2>&1; echo "C3 $ph exit=$? $(date -u +%T)" >> $L
done
touch results/CHAIN_C3_DONE
