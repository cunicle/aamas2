#!/usr/bin/env bash
# Build the anonymized supplementary zip (at most 25 MB) from the repository and the record
# archives in release/:
#
#   bash scripts/build_supplement.sh <bfcl-dir> <out.zip>
#
# <bfcl-dir> holds BFCL_v4_parallel*.json and possible_answer/ (scripts/prepare_data.py).
# Left out: the pod and diagnostic scripts (paths of the machines the runs used), the GPU
# cost estimate, notes and logs. The build stops if an identifying string remains.

set -euo pipefail
cd "$(dirname "$0")/.."
BFCL=${1:?bfcl dir}; OUT=$(realpath -m "${2:?output zip}")
TMP=$(mktemp -d); SUP=$TMP/supplement; mkdir -p $SUP
trap 'rm -rf $TMP' EXIT

cp supplement/README.md supplement/AI_USE.md requirements.txt $SUP/
# the prompts of the sessions that executed the experiments (AI_USE.md), without repository addresses
mkdir -p $SUP/prompts
for f in LOCAL_CLAUDE_PROMPT.md EXP_B_PROMPT.md EXP_C_PROMPT.md EXP_D_PROMPT.md; do
  sed -E 's#https://github\.com/[^ `]*#<repository>#g; s#claude/[a-z]+-[a-z]+-[0-9a-z]+#<branch>#g' $f > $SUP/prompts/$f
done
cp -r ptcdiag tests scripts $SUP/
rm -rf $SUP/scripts/pod $SUP/scripts/diag $SUP/scripts/estimate_cost.py $SUP/scripts/build_supplement.sh
find $SUP -name __pycache__ -type d -prune -exec rm -rf {} +
find $SUP -name '*.pyc' -delete
python3 scripts/print_prompts.py > $SUP/PROMPTS.md
mkdir -p $SUP/data/bfcl/possible_answer $SUP/results $SUP/expected/tables $SUP/expected/figures
cp data/choose.jsonl $SUP/data/
for c in parallel parallel_multiple; do
  cp $BFCL/BFCL_v4_$c.json $SUP/data/bfcl/
  cp $BFCL/possible_answer/BFCL_v4_$c.json $SUP/data/bfcl/possible_answer/
done
cp -r results/summary $SUP/results/
cp paper/tables/*.tex paper/tables/*.md $SUP/expected/tables/
cp paper/figures/surplus.pdf paper/figures/surplus.png $SUP/expected/figures/

# comments that point to the authors' internal notes
python3 - $SUP <<'EOF'
import pathlib, re, sys
subs = [(r" ?\(proposal §[^)]*\)", ""), (r"\(proposal §6, README 算力估计\)", ""),
        (r" ?\(see LOCAL_CLAUDE_PROMPT\.md\)", ""), (r" ?\(EXP_B_PROMPT\.md\)", ""),
        (r' ?\(slots longer than their value; README "已知限制" 4\)', " (slots longer than their value)")]
root = pathlib.Path(sys.argv[1])
for p in [*root.rglob("*.py"), *root.rglob("*.sh")]:
    s = p.read_text(encoding="utf-8")
    t = s
    for a, b in subs:
        t = re.sub(a, b, t)
    if t != s:
        p.write_text(t, encoding="utf-8")
EOF

# the raw records, without logs, smoke tests and gate runs
mkdir -p $TMP/rec
for a in results_2026-10-03 agents_2026-10-03 agents_b2_2026-10-03 exp_c_2026-10-04; do
  tar xzf release/$a.tar.gz -C $TMP/rec
done
(cd $TMP/rec && rm -rf results/smoke* results/gate_c results/CHAIN_* results/summary results/log_* \
    results/pilot.md results/b2_verify_ar.txt && tar cf - results | xz -9e -T0 > $SUP/records.tar.xz)

if grep -rEn --exclude=records.tar.xz \
    --exclude=AI_USE.md --exclude-dir=prompts \
    "proposal §|CLAUDE|Claude|EXP_[ABC]_PROMPT|HANDOFF|aamas2|/workspace|/root/|runpod|my-code|185\.216|cunicle" $SUP \
    || grep -rEn "aamas2|my-code|185\.216|cunicle|github\.com" $SUP/prompts $SUP/AI_USE.md; then
  echo "identifying strings remain (above)"; exit 1
fi
(cd $TMP && rm -f "$OUT" && zip -qr -9 "$OUT" supplement)
ls -l "$OUT"
