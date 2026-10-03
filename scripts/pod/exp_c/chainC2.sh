#!/usr/bin/env bash
# C3 variant decoding (started only after the user's decision on the merged closers): all 400 BFCL
# items (exact/+1/+2/+8 at k=1, exact/+1 at k=16), then on the 165 swap items the variant with
# swapped lengths and the variant one-sided run (the original one-sided run is skipped as done).
source /workspace/aamas2_env.sh
L=results/CHAIN_C_TIMES
echo "C2 start $(date -u +%T)" >> $L
for ph in closer_dream_main closer_dream_side; do
  bash scripts/run_minimal.sh $ph > results/log_$ph.txt 2>&1; echo "C2 $ph exit=$? $(date -u +%T)" >> $L
done
touch results/CHAIN_C2_DONE
