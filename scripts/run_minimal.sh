#!/usr/bin/env bash
# The "minimal" plan (proposal §6, README 算力估计): Dream-7B + LLaDA2.0-mini,
# AR controls Qwen2.5-7B-Instruct + Ling-mini-2.0. ~35 A100-hours estimated.
#
# Usage:  bash scripts/run_minimal.sh <phase>
#   setup     data + CPU tests
#   smoke     smoke tests for both dLLMs (must pass before anything else)
#   pilot     80 BFCL items x k in {1,4,16} per dLLM (LLaDA2.0 in both block settings):
#             accuracy, cross-call errors and timing before the budget is spent
#   dream     all Dream runs            (one GPU; set CUDA_VISIBLE_DEVICES)
#   llada2    all LLaDA2.0-mini runs, 32-token blocks and one-block contrast (80GB GPU)
#   ar        both AR baselines (= ar_qwen + ar_ling)
#   attr      counterfactual attribution + DVS for both dLLMs (= attr_dream + attr_llada2;
#             each after that model's runs)
#
# The sub-phases let one 80GB GPU run two chains side by side (LLaDA2.0 / Ling leave
# most of the GPU idle: their MoE layers loop over experts in Python), e.g.
#   llada2 -> attr_llada2   alongside   dream -> ar_qwen -> ar_ling -> attr_dream
#   summary   tables into results/summary/
#
# Length-prior study (slots longer than their value; README "已知限制" 4):
#   lenprior_dream / lenprior_llada2   teacher-forced: what the model wants right after the
#                                      gold value when the slot has 4 more masks
#   lensweep_dream / lensweep_llada2   the same with 1, 2, 8 more masks (Dream) / 1, 8 (LLaDA2.0)
#   surplus_dream     BFCL, every slot s in {1,2,4,8} masks too long, k in {4,1,16}
#   surplus_llada2    100 BFCL items, s in {2,8}, k in {4,1}   (s=0 is in bfcl_skel_k.jsonl)
#   surplus_ar        Qwen2.5 skeleton with s=8 (AR stops at the closer; finishes s=0 first)
#   endbias_dream     logit bonus b in {2,4,8} for padding/closers, s in {2,8}, k=4
#   surplus1_llada2   the 100 LLaDA2.0 items with s=1, k in {4,1,16}
#   swap_dream / swap_llada2   length swap: sibling slots get each other's lengths (items
#                     where that changes a slot; LLaDA2.0 on its 100 items), k in {4,1,16} / {4,1}
#   estimate_dream / estimate_llada2   slot lengths from one forward pass (no gold lengths),
#                     then decoding with them, k in {4,1,16} / {4,1}
#   choose_dream / choose_llada2 / choose_ar   choose-N probes (data/choose.jsonl): requests
#                     that leave open which N cities to call; duplicates vs k
#   summary_length    tables of the length study into results/summary/ (CPU)
# e.g. lenprior_llada2 -> surplus_llada2  alongside  lenprior_dream -> surplus_dream -> surplus_ar -> endbias_dream
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

choose_data() {  # the choose-N probes, generated once (deterministic)
  [ -f data/choose.jsonl ] ||     $PY_DREAM -c "from ptcdiag.data import choose, save_jsonl; save_jsonl(choose.generate(), 'data/choose.jsonl')"
}

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
  pilot)
    # 40 evenly spaced items per BFCL category. Written into the main result files, so the
    # dream / llada2 phases skip these runs later and the pilot costs nothing extra.
    PILOT="--data bfcl:parallel --data bfcl:parallel_multiple --per-data 40 --mode skeleton --k 1,4,16"
    mkdir -p results/dream results/llada2
    $PY_DREAM scripts/run_dllm.py --model $DREAM $PILOT --out results/dream/bfcl_skel_k.jsonl
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 $PILOT --block-length 32 --out results/llada2/bfcl_skel_k.jsonl
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 $PILOT --block-length none --out results/llada2/bfcl_skel_bfull.jsonl
    $PY_DREAM scripts/summarize.py results/dream/bfcl_skel_k.jsonl results/llada2/bfcl_skel_k.jsonl \
        results/llada2/bfcl_skel_bfull.jsonl | tee results/pilot.md
    $PY_DREAM - <<'EOF' | tee -a results/pilot.md
import json
# timing from the k=1 runs at each model's main setting; LLaDA2.0 with 32-token blocks
# forwards only up to the current block, on average prompt + half the generation region
for t, tag, frac in (("dream", "confidence_k1_tnone_bfull_T0.0", 1.0),
                     ("llada2", "confidence_k1_tnone_b32_T0.0", 0.5)):
    rs = [json.loads(l) for l in open(f"results/{t}/bfcl_skel_k.jsonl")]
    rs = [r for r in rs if r.get("cfg_tag") == tag and r.get("nfe")]
    ms = [1000 * r["seconds"] / r["nfe"] for r in rs]
    L = [r["prompt_len"] + frac * r["cfg"]["gen_length"] for r in rs]
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
    # block-length contrast (proposal §6.4): with 32-token blocks, same-parameter slots of
    # different calls share a block in <1% of pairs and so are never committed together;
    # one block over the whole canvas removes that barrier for the same weights
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data $BFCL --mode skeleton --block-length none \
        --k 4,1,16 --order confidence --out results/llada2/bfcl_skel_bfull.jsonl
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data $PROBE --mode skeleton --block-length none \
        --k 4,1 --order confidence --out results/llada2/probe_skel_bfull.jsonl
    # free mode with the model's own default decoding (block 32, threshold 0.95)
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data $BFCL --mode free --gen-length 256 \
        --block-length 32 --k 1 --threshold 0.95 --out results/llada2/bfcl_free.jsonl
    ;;
  ar)
    bash "$0" ar_qwen
    bash "$0" ar_ling
    ;;
  ar_qwen)
    mkdir -p results/qwen
    for mode in skeleton free; do
      $PY_DREAM scripts/run_ar.py --model $QWEN --data $BFCL --mode $mode --out results/qwen/bfcl_$mode.jsonl
    done
    $PY_DREAM scripts/run_ar.py --model $QWEN --data $PROBE --mode skeleton --out results/qwen/probe_skeleton.jsonl
    ;;
  ar_ling)
    mkdir -p results/ling
    for mode in skeleton free; do
      $PY_LLADA2 scripts/run_ar.py --model $LING --data $BFCL --mode $mode --out results/ling/bfcl_$mode.jsonl
    done
    $PY_LLADA2 scripts/run_ar.py --model $LING --data $PROBE --mode skeleton --out results/ling/probe_skeleton.jsonl
    ;;
  attr)
    bash "$0" attr_dream
    bash "$0" attr_llada2
    ;;
  attr_dream)
    $PY_DREAM scripts/attribute.py --model $DREAM --results results/dream/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/dream/attr_k4.jsonl | tee results/dream/attr_k4.txt
    $PY_DREAM scripts/dvs.py --model $DREAM --results results/dream/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_bfull_T0.0 --max-records 150 --out results/dream/dvs_k4.jsonl \
        | tee results/dream/dvs_k4.txt
    ;;
  attr_llada2)
    $PY_LLADA2 scripts/attribute.py --model $LLADA2 --results results/llada2/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_b32_T0.0 --out results/llada2/attr_k4.jsonl | tee results/llada2/attr_k4.txt
    $PY_LLADA2 scripts/dvs.py --model $LLADA2 --results results/llada2/bfcl_skel_k.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_b32_T0.0 --max-records 150 --out results/llada2/dvs_k4.jsonl \
        | tee results/llada2/dvs_k4.txt
    $PY_LLADA2 scripts/attribute.py --model $LLADA2 --results results/llada2/bfcl_skel_bfull.jsonl --data $BFCL \
        --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/llada2/attr_k4_bfull.jsonl \
        | tee results/llada2/attr_k4_bfull.txt
    ;;
  lenprior_dream)
    $PY_DREAM scripts/length_prior.py --model $DREAM --data $BFCL --surplus 4         --out results/dream/length_prior.jsonl | tee results/dream/length_prior.txt
    ;;
  lenprior_llada2)
    $PY_LLADA2 scripts/length_prior.py --model $LLADA2 --data $BFCL --surplus 4 --block-length 32         --out results/llada2/length_prior.jsonl | tee results/llada2/length_prior.txt
    ;;
  lensweep_dream)
    for s in 1 2 8; do
      $PY_DREAM scripts/length_prior.py --model $DREAM --data $BFCL --surplus $s           --out results/dream/length_prior_s$s.jsonl | tee results/dream/length_prior_s$s.txt
    done
    ;;
  lensweep_llada2)
    for s in 1 8; do
      $PY_LLADA2 scripts/length_prior.py --model $LLADA2 --data $BFCL --surplus $s --block-length 32           --out results/llada2/length_prior_s$s.jsonl | tee results/llada2/length_prior_s$s.txt
    done
    ;;
  surplus_dream)
    for s in 1 2 4 8; do
      $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --surplus $s           --k 4,1,16 --order confidence --out results/dream/bfcl_surplus.jsonl
    done
    ;;
  surplus_llada2)
    for s in 2 8; do
      $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data bfcl:parallel --data bfcl:parallel_multiple           --per-data 50 --mode skeleton --block-length 32 --surplus $s --k 4,1 --order confidence           --out results/llada2/bfcl_surplus.jsonl
    done
    ;;
  surplus_ar)
    mkdir -p results/qwen
    $PY_DREAM scripts/run_ar.py --model $QWEN --data $BFCL --mode skeleton --out results/qwen/bfcl_skeleton.jsonl
    $PY_DREAM scripts/run_ar.py --model $QWEN --data $BFCL --mode skeleton --surplus 8         --out results/qwen/bfcl_surplus.jsonl
    ;;
  endbias_dream)
    for b in 2 4 8; do
      for s in 2 8; do
        $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --surplus $s --end-bias $b             --k 4 --order confidence --out results/dream/bfcl_endbias.jsonl
      done
    done
    ;;
  surplus1_llada2)
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data bfcl:parallel --data bfcl:parallel_multiple         --per-data 50 --mode skeleton --block-length 32 --surplus 1 --k 4,1,16 --order confidence         --out results/llada2/bfcl_surplus.jsonl
    ;;
  swap_dream)
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --lengths swap         --k 4,1,16 --order confidence --out results/dream/bfcl_swap.jsonl
    ;;
  swap_llada2)
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data bfcl:parallel --data bfcl:parallel_multiple         --per-data 50 --mode skeleton --block-length 32 --lengths swap --k 4,1 --order confidence         --out results/llada2/bfcl_swap.jsonl
    ;;
  estimate_dream)
    $PY_DREAM scripts/length_estimate.py --model $DREAM --data $BFCL         --out results/dream/length_estimate.jsonl | tee results/dream/length_estimate.txt
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton         --lengths results/dream/length_estimate.jsonl --k 4,1,16 --order confidence         --out results/dream/bfcl_estimate.jsonl
    ;;
  estimate_llada2)
    $PY_LLADA2 scripts/length_estimate.py --model $LLADA2 --data bfcl:parallel --data bfcl:parallel_multiple         --per-data 50 --block-length 32 --out results/llada2/length_estimate.jsonl         | tee results/llada2/length_estimate.txt
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data bfcl:parallel --data bfcl:parallel_multiple         --per-data 50 --mode skeleton --block-length 32 --lengths results/llada2/length_estimate.jsonl         --k 4,1 --order confidence --out results/llada2/bfcl_estimate.jsonl
    ;;
  choose_dream)
    choose_data
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data probe:data/choose.jsonl --mode skeleton         --k 1,2,4,16 --order confidence --out results/dream/choose.jsonl
    ;;
  choose_llada2)
    choose_data
    $PY_LLADA2 scripts/run_dllm.py --model $LLADA2 --data probe:data/choose.jsonl --mode skeleton         --block-length 32 --k 1,4,16 --order confidence --out results/llada2/choose.jsonl
    ;;
  choose_ar)
    choose_data
    $PY_DREAM scripts/run_ar.py --model $QWEN --data probe:data/choose.jsonl --mode skeleton         --out results/qwen/choose.jsonl
    ;;
  summary_length)
    # tables of the length-prior study (all CPU): results/summary/*.md and .csv
    mkdir -p results/summary
    D=results/dream; L=results/llada2; Q=results/qwen
    $PY_DREAM scripts/length_analysis.py $D/bfcl_skel_k.jsonl $D/bfcl_surplus.jsonl $D/bfcl_endbias.jsonl \
        $D/bfcl_estimate.jsonl $L/bfcl_skel_k.jsonl $L/bfcl_surplus.jsonl $L/bfcl_estimate.jsonl \
        $Q/bfcl_skeleton.jsonl $Q/bfcl_surplus.jsonl --csv results/summary/length.csv | tee results/summary/length.md
    for t in dream llada2; do
      $PY_DREAM scripts/slot_errors.py results/$t/bfcl_skel_k.jsonl results/$t/bfcl_surplus.jsonl \
          results/$t/bfcl_endbias.jsonl results/$t/bfcl_estimate.jsonl --common \
          --csv results/summary/slots_$t.csv | tee results/summary/slots_$t.md
      $PY_DREAM scripts/slot_errors.py results/$t/bfcl_skel_k.jsonl results/$t/bfcl_swap.jsonl --swap-slots \
          --csv results/summary/swap_$t.csv | tee results/summary/swap_$t.md
    done
    $PY_DREAM scripts/symmetry.py $DREAM $D/bfcl_skel_k.jsonl $D/bfcl_skel_ltr.jsonl $D/bfcl_skel_tau.jsonl \
        | tee results/summary/symmetry_dream.md
    $PY_DREAM scripts/symmetry.py $LLADA2 $L/bfcl_skel_k.jsonl | tee results/summary/symmetry_llada2.md
    $PY_DREAM scripts/choose_analysis.py $D/choose.jsonl $L/choose.jsonl $Q/choose.jsonl \
        --csv results/summary/choose.csv | tee results/summary/choose.md
    $PY_DREAM scripts/masquerade.py $D/bfcl_skel_k.jsonl $D/bfcl_surplus.jsonl $D/bfcl_swap.jsonl \
        $D/bfcl_estimate.jsonl $L/bfcl_skel_k.jsonl $L/bfcl_surplus.jsonl $L/bfcl_swap.jsonl $L/bfcl_estimate.jsonl \
        --csv results/summary/masquerade.csv | tee results/summary/masquerade.md
    $PY_DREAM scripts/choose_pairs.py $D/choose.jsonl $L/choose.jsonl | tee results/summary/choose_pairs.md
    $PY_DREAM scripts/block_share.py $L/bfcl_skel_k.jsonl $L/choose.jsonl | tee results/summary/block_share.md
    $PY_DREAM scripts/consequences.py $D/bfcl_skel_k.jsonl $D/bfcl_surplus.jsonl $D/bfcl_swap.jsonl \
        $D/bfcl_estimate.jsonl $L/bfcl_skel_k.jsonl $L/bfcl_surplus.jsonl $L/bfcl_swap.jsonl $L/bfcl_estimate.jsonl \
        --csv results/summary/consequences.csv | tee results/summary/consequences.md
    $PY_DREAM scripts/paper_tables.py --results results --summary results/summary --out paper/tables
    $PY_DREAM scripts/paper_figures.py --summary results/summary --out paper/figures
    for f in $D/length_prior*.txt $L/length_prior*.txt $D/length_estimate.txt $L/length_estimate.txt; do
      if [ -f $f ]; then echo "== $f"; cat $f; fi
    done > results/summary/length_prior.md
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
  # Experiment B (EXP_B_PROMPT.md): teams of n agents answer the choose-N items, one call
  # each, under four protocols (simultaneous / turn-taking x anonymous / indexed)
  agents_ar)
    choose_data
    mkdir -p results/qwen
    $PY_DREAM scripts/run_agents.py --model $QWEN --backend ar --data probe:data/choose.jsonl \
        --protocols sim-anon,sim-label,turn-anon,turn-label --out results/qwen/agents.jsonl
    ;;
  agents_dream)
    choose_data
    mkdir -p results/dream
    $PY_DREAM scripts/run_agents.py --model $DREAM --backend dllm --data probe:data/choose.jsonl \
        --protocols sim-anon,sim-label,turn-anon,turn-label --out results/dream/agents.jsonl
    ;;
  agents_sample)
    # B2: simultaneous agents that sample (T=0.7, seeds 0-4)
    choose_data
    mkdir -p results/qwen results/dream
    $PY_DREAM scripts/run_agents.py --model $QWEN --backend ar --data probe:data/choose.jsonl \
        --protocols sim-anon,sim-label --temperature 0.7 --seeds 0,1,2,3,4 --out results/qwen/agents_sample.jsonl
    $PY_DREAM scripts/run_agents.py --model $DREAM --backend dllm --data probe:data/choose.jsonl \
        --protocols sim-anon,sim-label --temperature 0.7 --seeds 0,1,2,3,4 --out results/dream/agents_sample.jsonl
    ;;
  choose_sample_dream)
    # B2 one-canvas contrast: Dream writes all n calls with sampling (a new file, not choose.jsonl)
    choose_data
    mkdir -p results/dream
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data probe:data/choose.jsonl --mode skeleton \
        --k 1,16 --order confidence --temperature 0.7 --seeds 0,1,2,3,4 --out results/dream/choose_sample.jsonl
    ;;
  summary_agents)
    # tables of experiment B (CPU): results/summary/agents.md, .csv, agents_examples.md
    mkdir -p results/summary
    $PY_DREAM scripts/agents_analysis.py results/qwen/agents.jsonl results/dream/agents.jsonl \
        results/qwen/agents_sample.jsonl results/dream/agents_sample.jsonl \
        --canvas results/summary/choose.csv --csv results/summary/agents.csv \
        --examples results/summary/agents_examples.md | tee results/summary/agents.md
    if [ -f results/dream/choose_sample.jsonl ]; then
      $PY_DREAM scripts/choose_analysis.py results/dream/choose_sample.jsonl \
          --csv results/summary/choose_sample.csv | tee results/summary/choose_sample.md
    fi
    ;;
  # Experiment C: C1 / C2 teams of agents on the BFCL parallel requests (order cue: the 104
  # team-symmetric requests; length cue: the 165 swap requests) and the choose-N sim-rule
  # control; C3 the closer-in-slot skeleton variant and one-sided lengthening (Dream, one
  # canvas); C4 confidence threshold / left-to-right on choose-N. Sub-phases closer_dream_main /
  # closer_dream_side write different files, so two chains can run them side by side.
  agents_bfcl_ar)
    mkdir -p results/qwen
    $PY_DREAM scripts/run_agents_bfcl.py --model $QWEN --backend ar --subset sym \
        --protocols sim-anon,sim-label,sim-rule,turn-anon,turn-label --out results/qwen/agents_c1.jsonl
    $PY_DREAM scripts/run_agents_bfcl.py --model $QWEN --backend ar --subset swap \
        --protocols sim-anon,sim-label,turn-anon,turn-label --out results/qwen/agents_c2.jsonl
    ;;
  agents_bfcl_dream)
    mkdir -p results/dream
    $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset sym \
        --protocols sim-anon,sim-label,sim-rule,turn-anon,turn-label --out results/dream/agents_c1.jsonl
    $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset swap \
        --protocols sim-anon,sim-label,turn-anon,turn-label --out results/dream/agents_c2.jsonl
    ;;
  agents_bfcl_swap)
    mkdir -p results/dream
    $PY_DREAM scripts/run_agents_bfcl.py --model $DREAM --backend dllm --subset swap --length-mode swap \
        --protocols sim-anon,sim-label,turn-anon,turn-label --out results/dream/agents_c2_swap.jsonl
    ;;
  agents_rule)
    # choose-N list items, simultaneous agents told the convention (new files, not agents.jsonl)
    choose_data
    mkdir -p results/qwen results/dream
    $PY_DREAM scripts/run_agents.py --model $QWEN --backend ar --data probe:data/choose.jsonl --variant list \
        --protocols sim-rule --out results/qwen/agents_rule.jsonl
    $PY_DREAM scripts/run_agents.py --model $DREAM --backend dllm --data probe:data/choose.jsonl --variant list \
        --protocols sim-rule --out results/dream/agents_rule.jsonl
    ;;
  closer_dream)
    bash "$0" closer_dream_main
    bash "$0" closer_dream_side
    ;;
  closer_dream_main)
    # the closer-in-slot variant on every BFCL item: s = 0 (exact), 1, 2, 8 at k=1; s = 0, 1 at k=16
    mkdir -p results/dream
    for s in 0 1 2 8; do
      $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --closer-in-slot --surplus $s \
          --k 1 --order confidence --out results/dream/bfcl_closer.jsonl
    done
    for s in 0 1; do
      $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --closer-in-slot --surplus $s \
          --k 16 --order confidence --out results/dream/bfcl_closer.jsonl
    done
    ;;
  closer_dream_side)
    # the 165 swap items, k=1: the variant with swapped lengths, one-sided lengthening in both interfaces
    mkdir -p results/dream
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --closer-in-slot --length-mode swap \
        --k 1 --order confidence --out results/dream/bfcl_closer_swap.jsonl
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --length-mode onesided \
        --k 1 --order confidence --out results/dream/bfcl_onesided.jsonl
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data $BFCL --mode skeleton --closer-in-slot --length-mode onesided \
        --k 1 --order confidence --out results/dream/bfcl_closer_onesided.jsonl
    ;;
  closer_probe)
    # teacher-forced P(closer) right after the gold value in the variant (1 + s positions left)
    mkdir -p results/dream
    for s in 0 1 2 8; do
      $PY_DREAM scripts/length_prior.py --model $DREAM --data $BFCL --closer-in-slot --surplus $s \
          --out results/dream/length_prior_closer_s$s.jsonl | tee results/dream/length_prior_closer_s$s.txt
    done
    ;;
  choose_tau_ltr)
    # one canvas on choose-N: confidence threshold 0.9 and left-to-right (a new file, not choose.jsonl)
    choose_data
    mkdir -p results/dream
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data probe:data/choose.jsonl --mode skeleton --k 1 \
        --threshold 0.9 --out results/dream/choose_tau_ltr.jsonl
    $PY_DREAM scripts/run_dllm.py --model $DREAM --data probe:data/choose.jsonl --mode skeleton --k 1 \
        --order left_to_right --out results/dream/choose_tau_ltr.jsonl
    ;;
  summary_c)
    # tables of experiment C (CPU); the one-canvas rows come from the 10-03 raw results
    mkdir -p results/summary
    REL=results/release_2026-10-03
    [ -d $REL/results ] || { mkdir -p $REL && tar xzf release/results_2026-10-03.tar.gz -C $REL; }
    D=results/dream; Q=results/qwen
    $PY_DREAM scripts/agents_bfcl_analysis.py --teams $Q/agents_c1.jsonl $D/agents_c1.jsonl \
        $Q/agents_c2.jsonl $D/agents_c2.jsonl $D/agents_c2_swap.jsonl \
        --rule $Q/agents_rule.jsonl $D/agents_rule.jsonl --canvas-root $REL/results \
        --agents-csv results/summary/agents.csv --csv results/summary/agents_bfcl.csv \
        --examples results/summary/agents_bfcl_examples.md | tee results/summary/agents_bfcl.md
    $PY_DREAM scripts/closer_analysis.py --results results --original $REL/results \
        --csv results/summary/closer.csv | tee results/summary/closer.md
    $PY_DREAM scripts/choose_analysis.py $D/choose_tau_ltr.jsonl $REL/results/dream/choose.jsonl \
        --csv results/summary/choose_tau_ltr.csv | tee results/summary/choose_tau_ltr.md
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
