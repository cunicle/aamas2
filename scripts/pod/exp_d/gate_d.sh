#!/usr/bin/env bash
# Experiment D gates 1-3 on the pod (EXP_D_PROMPT.md section 3): data, tests and dry runs; the
# default path (Dream, 5 BFCL items, k=1/16, compared with the 10-03 records afterwards); format-
# tolerant slots (exact and +1 on the first 20 items, k=1); position agents (both models, BFCL sym
# and choose-N, 3 items each). Two sub-chains: Dream / Qwen. Everything goes to results/gate_d/.
source /workspace/aamas2_env.sh
G=results/gate_d; mkdir -p $G
DREAM=Dream-org/Dream-v0-Instruct-7B; QWEN=Qwen/Qwen2.5-7B-Instruct; BFCL=bfcl:parallel,parallel_multiple
t() { echo "$1 $(date -u +%T)" >> $G/times.txt; }
t start
$PY_DREAM scripts/prepare_data.py > $G/prepare_data.txt 2>&1; t "prepare_data exit=$?"
md5sum data/choose.jsonl data/bfcl/*.json data/bfcl/possible_answer/*.json > $G/data_md5.txt
$PY_DREAM -m pytest tests -q -p no:cacheprovider > $G/pytest.txt 2>&1; t "pytest exit=$?"
$PY_DREAM scripts/run_agents_bfcl.py --subset sym --protocols pos-anon --dry-run --limit 2 > $G/dryrun_bfcl.txt 2>&1
$PY_DREAM scripts/run_agents.py --protocols pos-anon --variant list --dry-run --limit 2 > $G/dryrun_choose.txt 2>&1
t dryrun
(
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1,16 --limit 5 \
      --out $G/default5.jsonl > $G/log_default5.txt 2>&1; t "default5 exit=$?"
  for s in 0 1; do
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1 --tolerant --surplus $s \
        --limit 20 --out $G/tolerant20.jsonl > $G/log_tolerant20_s$s.txt 2>&1; t "tolerant20 s=$s exit=$?"
  done
  $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset sym --protocols pos-anon --limit 3 \
      --out $G/dream_pos_bfcl.jsonl > $G/log_dream_pos_bfcl.txt 2>&1; t "dream_pos_bfcl exit=$?"
  $PY_DREAM scripts/run_agents.py --model $DREAM --backend dllm --data probe:data/choose.jsonl --protocols pos-anon \
      --limit 3 --out $G/dream_pos_choose.jsonl > $G/log_dream_pos_choose.txt 2>&1; t "dream_pos_choose exit=$?"
) &
(
  $PY_DREAM scripts/run_agents_bfcl.py --model $QWEN --backend ar --subset sym --protocols pos-anon --limit 3 \
      --out $G/qwen_pos_bfcl.jsonl > $G/log_qwen_pos_bfcl.txt 2>&1; t "qwen_pos_bfcl exit=$?"
  $PY_DREAM scripts/run_agents.py --model $QWEN --backend ar --data probe:data/choose.jsonl --protocols pos-anon \
      --limit 3 --out $G/qwen_pos_choose.jsonl > $G/log_qwen_pos_choose.txt 2>&1; t "qwen_pos_choose exit=$?"
) &
wait
t done
touch $G/GATE_DONE
