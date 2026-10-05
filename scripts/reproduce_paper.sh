#!/usr/bin/env bash
# Rebuild every summary, table and figure of the paper, and its statistics, from the raw records.
# CPU only; the tokenizers of the three models are fetched from the Hugging Face Hub.
#
#   tar xJf records.tar.xz             # -> results/{dream,llada2,qwen}/*.jsonl
#   bash scripts/reproduce_paper.sh    # -> results/summary/, out/tables/, out/figures/
#
# The GPU runs that wrote the records are the phases of scripts/run_minimal.sh.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DREAM=Dream-org/Dream-v0-Instruct-7B
LLADA2=inclusionAI/LLaDA2.0-mini
D=results/dream; L=results/llada2; Q=results/qwen; S=results/summary
mkdir -p $S out/tables out/figures

# the diagnoses stored with the records, recomputed with the current taxonomy (ptcdiag/eval/taxonomy.py)
$PY scripts/rediagnose.py $D/*.jsonl $L/*.jsonl $Q/*.jsonl

# one canvas: k sweep, surplus, swap, estimated lengths, closing bias, choose-N (Tables 1, 4, 5)
$PY scripts/length_analysis.py $D/bfcl_skel_k.jsonl $D/bfcl_surplus.jsonl $D/bfcl_endbias.jsonl \
    $D/bfcl_estimate.jsonl $L/bfcl_skel_k.jsonl $L/bfcl_surplus.jsonl $L/bfcl_estimate.jsonl \
    $Q/bfcl_skeleton.jsonl $Q/bfcl_surplus.jsonl --csv $S/length.csv > $S/length.md
for t in dream llada2; do
  $PY scripts/slot_errors.py results/$t/bfcl_skel_k.jsonl results/$t/bfcl_surplus.jsonl \
      results/$t/bfcl_endbias.jsonl results/$t/bfcl_estimate.jsonl --common \
      --csv $S/slots_$t.csv > $S/slots_$t.md
  $PY scripts/slot_errors.py results/$t/bfcl_skel_k.jsonl results/$t/bfcl_swap.jsonl --swap-slots \
      --csv $S/swap_$t.csv > $S/swap_$t.md
done
$PY scripts/symmetry.py $DREAM $D/bfcl_skel_k.jsonl $D/bfcl_skel_ltr.jsonl $D/bfcl_skel_tau.jsonl \
    > $S/symmetry_dream.md
$PY scripts/symmetry.py $LLADA2 $L/bfcl_skel_k.jsonl > $S/symmetry_llada2.md
$PY scripts/choose_analysis.py $D/choose.jsonl $L/choose.jsonl $Q/choose.jsonl --csv $S/choose.csv > $S/choose.md
$PY scripts/masquerade.py $D/bfcl_skel_k.jsonl $D/bfcl_surplus.jsonl $D/bfcl_swap.jsonl $D/bfcl_estimate.jsonl \
    $L/bfcl_skel_k.jsonl $L/bfcl_surplus.jsonl $L/bfcl_swap.jsonl $L/bfcl_estimate.jsonl \
    --csv $S/masquerade.csv > $S/masquerade.md
$PY scripts/choose_pairs.py $D/choose.jsonl $L/choose.jsonl > $S/choose_pairs.md
$PY scripts/block_share.py $L/bfcl_skel_k.jsonl $L/choose.jsonl > $S/block_share.md
$PY scripts/co_commit.py $D/bfcl_skel_k.jsonl $D/bfcl_skel_tau.jsonl $L/bfcl_skel_k.jsonl > $S/co_commit.md
$PY scripts/consequences.py $D/bfcl_skel_k.jsonl $D/bfcl_surplus.jsonl $D/bfcl_swap.jsonl $D/bfcl_estimate.jsonl \
    $L/bfcl_skel_k.jsonl $L/bfcl_surplus.jsonl $L/bfcl_swap.jsonl $L/bfcl_estimate.jsonl \
    --csv $S/consequences.csv > $S/consequences.md
# teacher-forced closing probe (Figure 3c)
for f in $D/length_prior.txt $D/length_prior_s1.txt $D/length_prior_s2.txt $D/length_prior_s8.txt \
         $L/length_prior.txt $L/length_prior_s1.txt $L/length_prior_s2.txt $L/length_prior_s8.txt \
         $D/length_estimate.txt $L/length_estimate.txt; do
  echo "== $f"; cat $f
done > $S/length_prior.md
# what the type mask forbids in surplus masks (Section 6)
$PY scripts/typemask_probe.py --results results > $S/typemask_probe.md

# teams of agents on choose-N (Table 2, right)
$PY scripts/agents_analysis.py $Q/agents.jsonl $D/agents.jsonl $Q/agents_sample.jsonl $D/agents_sample.jsonl \
    $Q/agents_pos.jsonl $D/agents_pos.jsonl \
    --canvas $S/choose.csv --csv $S/agents.csv --examples $S/agents_examples.md > $S/agents.md
$PY scripts/choose_analysis.py $D/choose_sample.jsonl --csv $S/choose_sample.csv > $S/choose_sample.md
$PY scripts/choose_analysis.py $D/choose_tau_ltr.jsonl $D/choose.jsonl --csv $S/choose_tau_ltr.csv \
    > $S/choose_tau_ltr.md
$PY scripts/choose_focal.py $Q/agents.jsonl $D/agents.jsonl $Q/agents_sample.jsonl $D/agents_sample.jsonl \
    $Q/agents_rule.jsonl $D/agents_rule.jsonl > $S/choose_focal.md

# teams of agents on BFCL (Table 2, left; Table 3, bottom) and the closing token in the slot (Table 3)
$PY scripts/agents_bfcl_analysis.py --teams $Q/agents_c1.jsonl $D/agents_c1.jsonl \
    $Q/agents_c2.jsonl $D/agents_c2.jsonl $D/agents_c2_swap.jsonl $Q/agents_d.jsonl $D/agents_d.jsonl \
    --rule $Q/agents_rule.jsonl $D/agents_rule.jsonl --canvas-root results \
    --agents-csv $S/agents.csv --csv $S/agents_bfcl.csv --examples $S/agents_bfcl_examples.md > $S/agents_bfcl.md
$PY scripts/closer_analysis.py --results results --original results --csv $S/closer.csv > $S/closer.md
$PY scripts/swap_direction.py --results results > $S/swap_direction.md
$PY scripts/exp_d_analysis.py --results results > $S/exp_d.md

# the paper's tables, Figure 3, statistics and other quoted numbers
$PY scripts/paper_tables.py --results results --summary $S --out out/tables --c-results results
$PY scripts/paper_figures.py --summary $S --out out/figures
$PY scripts/stats.py --results results --agents-results results --c-results results --out $S/stats.md
$PY scripts/review_numbers.py --results results > $S/review_numbers.md
echo "done: results/summary/, out/tables/, out/figures/"
