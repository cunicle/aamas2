#!/usr/bin/env bash
# Experiment D, A, part 2: the swap, one-sided and length-estimate commands of the tolerant_dream
# phase, copied from scripts/run_minimal.sh (EST = the 10-03 length estimates).
source /workspace/aamas2_env.sh
L=results/CHAIN_D_TIMES; D=results/dream
DREAM=Dream-org/Dream-v0-Instruct-7B; BFCL=bfcl:parallel,parallel_multiple
EST=results/release_2026-10-03/results/dream/length_estimate.jsonl
echo "D3 start $(date -u +%T)" >> $L
$PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1 --tolerant \
    --length-mode swap --out $D/bfcl_tolerant_swap.jsonl > results/log_tolerant_swap.txt 2>&1
echo "D3 swap exit=$? $(date -u +%T)" >> $L
$PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1 --tolerant \
    --length-mode onesided --out $D/bfcl_tolerant_onesided.jsonl > results/log_tolerant_onesided.txt 2>&1
echo "D3 onesided exit=$? $(date -u +%T)" >> $L
$PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1,16 --tolerant \
    --length-mode $EST --out $D/bfcl_tolerant_estimate.jsonl > results/log_tolerant_estimate.txt 2>&1
echo "D3 estimate exit=$? $(date -u +%T)" >> $L
touch results/CHAIN_D3_DONE
