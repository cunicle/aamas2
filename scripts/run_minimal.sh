#!/usr/bin/env bash
# The "minimal" plan (proposal §6, README 算力估计): Dream-7B + LLaDA2.0-mini,
# AR controls Qwen2.5-7B-Instruct + Ling-mini-2.0. ~31 A100-hours estimated.
#
# Usage:  bash scripts/run_minimal.sh <phase>
#   setup     data + CPU tests
#   smoke     smoke tests for both dLLMs (must pass before anything else)
#   timing    20-example timing run per dLLM, then re-estimate the budget
#   dream     all Dream runs            (one GPU; set CUDA_VISIBLE_DEVICES)
#   llada2    all LLaDA2.0-mini runs    (one 80GB GPU)
#   ar        both AR baselines
#   attr      counterfactual attribution + DVS for both dLLMs (after dream/llada2)
#   summary   tables into results/summary/
#
# Every run appends to its JSONL and skips finished items, so any phase can be
# re-run after an interruption. Environments: PY_DREAM / PY_LLADA2 point to the
# python of the env for each model family (Dream needs transformers 4.46.x,
# LLaDA2.0 / Ling 4.57.x); both default to `python`.

set -euo pipefail
cd "$(dirname "$0")/.."

PY_DREAM=${PY_DREAM:-python}
PY_LLADA2=${PY_LLADA2:-python}
BFCL=bfcl:parallel,parallel_multiple
PROBE=probe:data/paraprobe.jsonl
DREAM=Dream-org/Dream-v0-Instruct-7B
LLADA2=inclusionAI/LLaDA2.0-mini
QWEN=Qwen/Qwen2.5-7B-Instruct
LING=inclusionAI/Ling-mini-2.0

# k=4 first: attribution and DVS use it, so it is available earliest
K_BFCL=4,1,2,8,16
K_PROBE=4,1,2,8

dllm_runs() {  # $1 python  $2 model  $3 tag  $4 extra args (block length)
  local py=$1 m=$2 t=$3 extra=$4
  mkdir -p results/$t
  $py scripts/run_dllm.py --model $m --data $BFCL --mode skeleton $extra \
      --k $K_BFCL --order confidence --out results/$t/bfcl_skel_k.jsonl
  $py scripts/run_dllm.py --model $m --data $BFCL --mode skeleton $extra \
      --k 1 --order left_to_right --out results/$t/bfcl_skel_ltr.jsonl
  $py scripts/run_dllm.py --model $m --data $BFCL --mode skeleton $extra \
      --k 1 --threshold 0.9 --out results/$t/bfcl_skel_tau.jsonl
  $py scripts/run_dllm.py --model $m --data $PROBE --mode skeleton $extra \
      --k $K_PROBE --order confidence --out results/$t/probe_skel_k.jsonl
}

case "${1:-}" in
  setup)
    # environments are created beforehand (see LOCAL_CLAUDE_PROMPT.md); do not pip install
    # requirements.txt here, it would replace the pinned transformers version
    $PY_DREAM scripts/prepare_data.py --per-cell 5
    $PY_DREAM -m pytest tests -q
    ;;
  smoke)
    mkdir -p results
    $PY_DREAM scripts/smoke_test.py --model $DREAM 2>&1 | tee results/smoke_dream.txt
    $PY_LLADA2 scripts/smoke_test.py --model $LLADA2 2>&1 | tee results/smoke_llada2.txt
    ;;
  timing)
    mkdir -p results/timing
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --k 1 --limit 20 \
        --out results/timing/dream.jsonl
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data $BFCL --mode skeleton --k 1 \
        --block-length 32 --limit 20 --out results/timing/llada2.jsonl
    $PY_DREAM - <<'EOF'
import json
for t in ("dream", "llada2"):
    rs = [json.loads(l) for l in open(f"results/timing/{t}.jsonl") if "nfe" in l]
    ms = [1000 * r["seconds"] / r["nfe"] for r in rs if r.get("nfe")]
    L = [r["prompt_len"] + r["cfg"]["gen_length"] for r in rs if r.get("nfe")]
    f, l = sum(ms) / len(ms), sum(L) / len(L)
    print(f"{t}: {len(rs)} runs, {f:.0f} ms per forward at mean canvas {l:.0f} tokens"
          f"  ->  --timing {t}={max(f - 15, 1) * 1000 / l:.0f}:15")
EOF
    echo "Re-estimate with the --timing values above:"
    echo "  \$PY_LLADA2 scripts/estimate_cost.py --plan minimal --timing dream=..:15 --timing llada2=..:15"
    ;;
  dream)
    dllm_runs "$PY_DREAM" $DREAM dream ""
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode free --gen-length 256 \
        --k 2 --out results/dream/bfcl_free.jsonl
    ;;
  llada2)
    dllm_runs "$PY_LLADA2" $LLADA2 llada2 "--block-length 32"
    # free mode with the model's own default decoding (block 32, threshold 0.95)
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data $BFCL --mode free --gen-length 256 \
        --block-length 32 --k 1 --threshold 0.95 --out results/llada2/bfcl_free.jsonl
    ;;
  ar)
    mkdir -p results/qwen results/ling
    for mode in skeleton free; do
      $PY_DREAM scripts/run_ar.py --model $QWEN --data $BFCL --mode $mode --out results/qwen/bfcl_$mode.jsonl
      $PY_LLADA2 scripts/run_ar.py --model $LING --data $BFCL --mode $mode --out results/ling/bfcl_$mode.jsonl
    done
    $PY_DREAM scripts/run_ar.py --model $QWEN --data $PROBE --mode skeleton --out results/qwen/probe_skeleton.jsonl
    $PY_LLADA2 scripts/run_ar.py --model $LING --data $PROBE --mode skeleton --out results/ling/probe_skeleton.jsonl
    ;;
  attr)
    $PY_DREAM scripts/attribute.py --model $DREAM --results results/dream/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/dream/attr_k4.jsonl | tee results/dream/attr_k4.txt
    $PY_DREAM scripts/dvs.py --model $DREAM --results results/dream/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_bfull_T0.0 --max-records 150 --out results/dream/dvs_k4.jsonl \
        | tee results/dream/dvs_k4.txt
    $PY_LLADA2 scripts/attribute.py --model $LLADA2 --results results/llada2/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_b32_T0.0 --out results/llada2/attr_k4.jsonl | tee results/llada2/attr_k4.txt
    $PY_LLADA2 scripts/dvs.py --model $LLADA2 --results results/llada2/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_b32_T0.0 --max-records 150 --out results/llada2/dvs_k4.jsonl \
        | tee results/llada2/dvs_k4.txt
    ;;
  summary)
    mkdir -p results/summary
    for t in dream llada2; do
      ar=$([ $t = dream ] && echo qwen || echo ling)
      $PY_DREAM scripts/summarize.py results/$t/bfcl_*.jsonl results/$ar/bfcl_*.jsonl --by category \
          --csv results/summary/bfcl_$t.csv | tee results/summary/bfcl_$t.md
      for f in n entities ambiguity shared mix; do
        $PY_DREAM scripts/summarize.py results/$t/probe_*.jsonl results/$ar/probe_*.jsonl --by meta.$f \
            --csv results/summary/probe_${t}_$f.csv | tee results/summary/probe_${t}_$f.md
      done
    done
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
