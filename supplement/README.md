# Supplementary material

Code, prompts and raw records for *Acting in Parallel Without Talking: Focal Points, Collisions, and Length Signals in Diffusion-LM Agents and LLM Agent Teams*.

## Contents

| Path | What it holds |
|---|---|
| `PROMPTS.md` | Every prompt text: the system message, the choose-N request templates, and the five protocol notes of the teams of agents |
| `records.tar.xz` | The raw records of every run: one JSON line per request and condition, with the output text, the parsed calls, the BFCL verdict and, for the dLLMs, the decoding trace |
| `results/summary/` | The summaries computed from the records, which the tables, the figure and the quoted numbers are read from |
| `expected/` | The paper's tables and Figure 2 as built from these summaries |
| `ptcdiag/` | The library: data loading, the skeleton constraint, the traced masked-diffusion sampler, the AR reference, the set-level matching and error taxonomy |
| `scripts/` | Run scripts (`run_dllm.py`, `run_ar.py`, `run_agents.py`, `run_agents_bfcl.py`; the phases in `run_minimal.sh`), analysis scripts, and `reproduce_paper.sh` |
| `data/` | The BFCL v4 `parallel` and `parallel_multiple` categories with their possible answers (Apache-2.0, from the Gorilla repository), and the 105 choose-N requests |
| `tests/` | Unit tests (`python -m pytest tests -q`) |
| `AI_USE.md` | How AI tools were used in this work |

## Reproducing the tables, the figure and the statistics (CPU)

Python 3.11 with `numpy`, `scipy`, `matplotlib` and `transformers` (the analysis loads only the tokenizers of the three models from the Hugging Face Hub).

```bash
tar xJf records.tar.xz                  # -> results/{dream,llada2,qwen}/*.jsonl
bash scripts/reproduce_paper.sh         # -> results/summary/, out/tables/, out/figures/
diff -r out/tables expected/tables      # no output
```

| Paper | Built from |
|---|---|
| Table 1 | the `bfcl_skel_*` records, `symmetry_dream.md`, `symmetry_llada2.md` |
| Table 2 | `agents.csv`, `agents_bfcl.csv` |
| Table 3 | `closer.csv`, `agents_bfcl.csv` |
| Table 4 | `masquerade.csv` |
| Section 8 (closing bias, estimated lengths) | the `bfcl_skel_k`, `bfcl_surplus`, `bfcl_endbias` and `bfcl_estimate` records (`mitigation.tex`, not in the paper) |
| Figure 2 | `length.csv`, `length_prior.md`, `closer.csv` |
| Intervals and tests | `stats.md` |
| Other numbers | `review_numbers.md`, `co_commit.md`, `choose_focal.md`, `choose_pairs.md`, `choose_tau_ltr.md`, `block_share.md`, `consequences.md`, `swap_*.md`, `slots_*.md` |

`results/summary/agents_examples.md` and `agents_bfcl_examples.md` show complete prompts and outputs of example teams; `closer.md` shows filled canvases with the slots marked.

## The GPU runs

`scripts/run_minimal.sh <phase>` runs each condition and appends to its record file, skipping finished requests. Dream-v0-Instruct-7B and Qwen2.5-7B-Instruct ran with transformers 4.46.2 and torch 2.5.1, LLaDA2.0-mini with transformers 4.57.1 and torch 2.8.0. Decoding is greedy except where a temperature is given; sampled runs use the seeds recorded in each record. `scripts/smoke_test.py` checks each model's adapter against its reference sampler before any run.
