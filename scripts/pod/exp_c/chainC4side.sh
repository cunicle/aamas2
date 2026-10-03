#!/usr/bin/env bash
# After chainC3 (whose last Dream run is the original one-sided file), run closer_dream_side
# (variant swap + variant one-sided; the original one-sided records are skipped as done) while
# chainC2 is still in closer_dream_main; chainC2 later finds them done.
source /workspace/aamas2_env.sh
L=results/CHAIN_C_TIMES
until [ -f results/CHAIN_C3_DONE ]; do sleep 30; done
echo "C4side start $(date -u +%T)" >> $L
bash scripts/run_minimal.sh closer_dream_side > results/log_closer_dream_side_early.txt 2>&1
echo "C4side closer_dream_side exit=$? $(date -u +%T)" >> $L
touch results/CHAIN_C4SIDE_DONE
