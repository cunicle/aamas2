#!/usr/bin/env bash
# Experiment E gates 1-2 on the pod (EXP_E_PROMPT.md sections 2-3). Gate 1 (CPU): data, tests,
# bash -n. Old records: unpack the two archives into results/, count the lines of the three files
# that get appended and keep verbatim copies in /tmp/old for gate 4. Gate 2 (GPU): a few items of
# the neighbouring conditions that already ran, into /tmp/gate_e, then gate_check_e.py compares them
# with the old records. Writes /tmp/gate_e/GATE_PASS or GATE_FAIL, then GATE_DONE.
source /workspace/aamas2_env.sh
T=/tmp/gate_e; mkdir -p $T /tmp/old
DREAM=Dream-org/Dream-v0-Instruct-7B; QWEN=Qwen/Qwen2.5-7B-Instruct; LLADA2=inclusionAI/LLaDA2.0-mini
t() { echo "$1 $(date -u +%T)" >> $T/times.txt; }
t start
until [ -f /workspace/aamas2_logs/setup2_done ]; do sleep 20; done
t "setup done"
cat /workspace/aamas2_logs/venv2_*.txt /workspace/aamas2_logs/dl_*.txt | grep -E "^(dream|llada2) |exit=" > $T/setup.txt

# gate 1 (CPU)
$PY_DREAM scripts/prepare_data.py > $T/prepare_data.txt 2>&1; t "prepare_data exit=$?"
md5sum data/choose.jsonl data/bfcl/*.json data/bfcl/possible_answer/*.json > $T/data_md5.txt
tar xzf release/exp_d_2026-10-05.tar.gz -O results/gate_d/data_md5.txt > $T/data_md5_exp_d.txt
diff $T/data_md5_exp_d.txt $T/data_md5.txt > $T/data_md5_diff.txt; t "data md5 same as exp D: exit=$?"
$PY_DREAM -m pytest tests -q -p no:cacheprovider > $T/pytest.txt 2>&1; t "pytest exit=$?"
bash -n scripts/run_minimal.sh > $T/bash_n.txt 2>&1; t "bash -n exit=$? bytes=$(stat -c %s $T/bash_n.txt)"

# the old records
tar xzf release/results_2026-10-03.tar.gz
tar xzf release/exp_c_2026-10-04.tar.gz
for f in results/dream/bfcl_closer.jsonl results/qwen/bfcl_surplus.jsonl results/llada2/bfcl_surplus.jsonl; do
  wc -l $f; cp $f /tmp/old/$(echo $f | tr / _)
done > $T/old_lines.txt
md5sum /tmp/old/* >> $T/old_lines.txt
t "old records unpacked"

# gate 2 (GPU): the runbook's four commands; Dream then Qwen alongside LLaDA2.0
(
  $PY_DREAM scripts/run_dllm.py --model $DREAM --data bfcl:parallel,parallel_multiple \
      --mode skeleton --closer-in-slot --surplus 2 --k 1 --order confidence --limit 5 \
      --out $T/dream_closer_s2.jsonl > $T/log_dream_closer_s2.txt 2>&1; t "dream_closer_s2 exit=$?"
  $PY_DREAM scripts/run_ar.py --model $QWEN --data bfcl:parallel,parallel_multiple \
      --mode skeleton --surplus 8 --limit 5 --out $T/qwen_s8.jsonl > $T/log_qwen_s8.txt 2>&1; t "qwen_s8 exit=$?"
) &
(
  $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data bfcl:parallel --data bfcl:parallel_multiple \
      --per-data 50 --mode skeleton --block-length 32 --surplus 2 --k 4 --order confidence --limit 5 \
      --out $T/llada2_s2_k4.jsonl > $T/log_llada2_s2_k4.txt 2>&1; t "llada2_s2_k4 exit=$?"
  $PY_LLADA2 scripts/length_prior.py --model $LLADA2 --data bfcl:parallel,parallel_multiple \
      --surplus 1 --block-length 32 --limit 3 --out $T/llada2_probe_s1.jsonl > $T/log_llada2_probe_s1.txt 2>&1
  t "llada2_probe_s1 exit=$?"
) &
wait
$PY_DREAM scripts/pod/exp_e/gate_check_e.py > $T/gate_check.txt 2>&1; t "gate_check exit=$?"
if grep -q "^ALL GATES PASS" $T/gate_check.txt; then touch $T/GATE_PASS; else touch $T/GATE_FAIL; fi
touch $T/GATE_DONE
