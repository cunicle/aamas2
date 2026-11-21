#!/usr/bin/env bash
# Experiment F gates 1-2 on the pod (EXP_F_PROMPT.md section 3). Gate 1 (CPU): data, tests,
# bash -n, and the texts of the new wordings printed without a model. Old records: unpack the
# archives of experiments B, C and D into results/. Gate 2 (GPU): the original wordings of
# sim-label and sim-rule on the first 5 requests, both models, BFCL and choose-N, into
# /tmp/gate_f; gate_check_f.py compares them with the old records. Writes /tmp/gate_f/GATE_PASS
# or GATE_FAIL, then GATE_DONE.
source /workspace/aamas2_env.sh
T=/tmp/gate_f; mkdir -p $T
DREAM=Dream-org/Dream-v0-Instruct-7B; QWEN=Qwen/Qwen2.5-7B-Instruct
W=sim-label@para,sim-label@ordinal,sim-label@first,sim-rule@para,sim-rule@direct,sim-rule@first
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
$PY_DREAM scripts/run_agents_bfcl.py --subset sym --protocols $W --dry-run --limit 2 \
    > $T/texts_bfcl.txt 2> $T/texts_bfcl_stderr.txt; t "texts_bfcl exit=$?"
$PY_DREAM scripts/run_agents.py --data probe:data/choose.jsonl --variant list --protocols $W --dry-run --limit 2 \
    > $T/texts_choose.txt 2> $T/texts_choose_stderr.txt; t "texts_choose exit=$?"

# the old records: only the team files (the archives also hold summary files that are in git)
tar xzf release/agents_2026-10-03.tar.gz results/qwen/agents.jsonl results/dream/agents.jsonl
tar xzf release/exp_c_2026-10-04.tar.gz results/qwen/agents_c1.jsonl results/dream/agents_c1.jsonl \
    results/qwen/agents_rule.jsonl results/dream/agents_rule.jsonl
tar xzf release/exp_d_2026-10-05.tar.gz results/qwen/agents_d.jsonl results/dream/agents_d.jsonl \
    results/qwen/agents_pos.jsonl results/dream/agents_pos.jsonl
t "old records unpacked exit=$?"
wc -l results/{qwen,dream}/{agents_c1,agents_d,agents,agents_rule,agents_pos}.jsonl > $T/old_lines.txt 2>&1

# gate 2 (GPU): the original wordings again, Qwen then Dream
for tag in qwen dream; do
  if [ $tag = qwen ]; then m=$QWEN; b=ar; else m=$DREAM; b=dllm; fi
  $PY_DREAM scripts/run_agents_bfcl.py --model $m --backend $b --subset sym --protocols sim-label,sim-rule \
      --limit 5 --out $T/${tag}_bfcl.jsonl > $T/log_${tag}_bfcl.txt 2>&1; t "${tag}_bfcl exit=$?"
  $PY_DREAM scripts/run_agents.py --model $m --backend $b --data probe:data/choose.jsonl --variant list \
      --protocols sim-label,sim-rule --limit 5 --out $T/${tag}_choose.jsonl > $T/log_${tag}_choose.txt 2>&1
  t "${tag}_choose exit=$?"
done
$PY_DREAM scripts/pod/exp_f/gate_check_f.py > $T/gate_check.txt 2>&1; t "gate_check exit=$?"
if grep -q "^ALL GATES PASS" $T/gate_check.txt; then touch $T/GATE_PASS; else touch $T/GATE_FAIL; fi
touch $T/GATE_DONE
