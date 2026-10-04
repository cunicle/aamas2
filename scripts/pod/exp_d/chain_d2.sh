#!/usr/bin/env bash
# Experiment D, A, part 1: the bfcl_tolerant.jsonl commands of the tolerant_dream phase (format-
# tolerant slots, Dream, one canvas: exact/+1/+2/+8 at k=1, exact/+1 at k=16), copied from
# scripts/run_minimal.sh. Part 2 (chain_d3.sh) runs the phase's other three files alongside;
# finish_d.sh then re-runs the whole phase, which must find everything done.
source /workspace/aamas2_env.sh
L=results/CHAIN_D_TIMES; D=results/dream
DREAM=Dream-org/Dream-v0-Instruct-7B; BFCL=bfcl:parallel,parallel_multiple
echo "D2 start $(date -u +%T)" >> $L
for s in 0 1 2 8; do
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1 --tolerant \
      --surplus $s --out $D/bfcl_tolerant.jsonl > results/log_tolerant_k1_s$s.txt 2>&1
  echo "D2 k1 s=$s exit=$? $(date -u +%T)" >> $L
done
for s in 0 1; do
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 16 --tolerant \
      --surplus $s --out $D/bfcl_tolerant.jsonl > results/log_tolerant_k16_s$s.txt 2>&1
  echo "D2 k16 s=$s exit=$? $(date -u +%T)" >> $L
done
touch results/CHAIN_D2_DONE
