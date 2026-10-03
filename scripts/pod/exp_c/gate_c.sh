#!/usr/bin/env bash
# Experiment C gates 1-3 on the pod: tests, the default path (5 items, k=1/16, compared with the
# 10-03 records afterwards), smoke teams (both models, both subsets, 3 items; Dream swap too) and
# the closer-in-slot variant (exact on 20 items, +1 on 5). Two sub-chains: Dream / Qwen.
source /workspace/aamas2_env.sh
G=results/gate_c; mkdir -p $G
DREAM=Dream-org/Dream-v0-Instruct-7B; QWEN=Qwen/Qwen2.5-7B-Instruct; BFCL=bfcl:parallel,parallel_multiple
P5=sim-anon,sim-label,sim-rule,turn-anon,turn-label; P4=sim-anon,sim-label,turn-anon,turn-label
t() { echo "$1 $(date -u +%T)" >> $G/times.txt; }
t start
$PY_DREAM -c "from ptcdiag.data import bfcl, choose, save_jsonl; bfcl.download('data/bfcl'); save_jsonl(choose.generate(), 'data/choose.jsonl')"
$PY_DREAM -m pytest tests -q -p no:cacheprovider > $G/pytest.txt 2>&1; t "pytest exit=$?"
(
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1,16 --limit 5 \
      --out $G/default5.jsonl > $G/log_default5.txt 2>&1; t "default5 exit=$?"
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --closer-in-slot --surplus 0 \
      --k 1 --limit 20 --out $G/closer.jsonl > $G/log_closer0.txt 2>&1; t "closer0 exit=$?"
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --closer-in-slot --surplus 1 \
      --k 1 --limit 5 --out $G/closer.jsonl > $G/log_closer1.txt 2>&1; t "closer1 exit=$?"
  $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset sym --protocols $P5 --limit 3 \
      --out $G/dream_sym.jsonl > $G/log_dream_sym.txt 2>&1; t "dream_sym exit=$?"
  $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset swap --protocols $P4 --limit 3 \
      --out $G/dream_swap.jsonl > $G/log_dream_swap.txt 2>&1; t "dream_swap exit=$?"
  $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset swap --length-mode swap \
      --protocols $P4 --limit 3 --out $G/dream_swap_swap.jsonl > $G/log_dream_swap_swap.txt 2>&1
  t "dream_swap_swap exit=$?"
) &
(
  $PY_DREAM scripts/run_agents_bfcl.py --model $QWEN --backend ar --subset sym --protocols $P5 --limit 3 \
      --out $G/qwen_sym.jsonl > $G/log_qwen_sym.txt 2>&1; t "qwen_sym exit=$?"
  $PY_DREAM scripts/run_agents_bfcl.py --model $QWEN --backend ar --subset swap --protocols $P4 --limit 3 \
      --out $G/qwen_swap.jsonl > $G/log_qwen_swap.txt 2>&1; t "qwen_swap exit=$?"
) &
wait
t done
touch $G/GATE_DONE
