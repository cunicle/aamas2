# Experiment D: gates and report diagnostics

Run 2026-10-04 on one A100-SXM4-80GB (RunPod lufv6jfv2pbr6p): first ssh 17:48 UTC, main runs 17:58-19:47 UTC,
pod stopped 19:50 UTC (about 2.0 GPU hours). Raw records: release/exp_d_2026-10-05.tar.gz. The pre-registered quick
table is results/summary/exp_d_quick.md; this file adds the gate record and diagnostics computed from the same raw
records for the run report. Sections 3-6 are NOT in the plan (EXP_D_PROMPT.md); they are read-outs, not new runs.

## 1. Gates (EXP_D_PROMPT.md section 3)

| gate | result |
|---|---|
| 1 CPU | pytest 121 passed on the laptop and on the pod (the plan says 120: eeb27cf, after the experiment D code, added one test to tests/test_taxonomy.py; tests/test_exp_d.py has its 15). Dry runs: each position agent's user text is the original request only, every BFCL agent gets the whole turn's slot lengths. |
| 2 default path | Dream, 5 BFCL items, k=1 and k=16, no --tolerant: 10/10 records identical to the 10-03 bfcl_skel_k.jsonl in text and gen_ids. |
| 3 tolerant smoke | first 20 items, k=1: exact 95.0% vs the original interface 95.0% on the same items (gate: no more than 10 points lower). +1 slots hold filler (escaped newlines, i/f suffixes, '_000'), removed in text. |
| 3 position-agent smoke | both models, BFCL sym and choose-N, 3 teams each: other calls are placeholders (Dream: frozen masks decoded to ... / null), the own call is decoded by the model. Qwen's agent 2 wrote the placeholder '...' as its own city in all 3 choose-N teams (model output; see section 6). |
| 4 sanity | Dream pos-anon choose-N cities = Dream one canvas k=16 (10-03 choose.jsonl, confidence_k16_tnone_bfull_T0.0): 105/105 identical. |

The tolerant_dream phase ran as two chains side by side (its commands copied from run_minimal.sh), next to
position_agents; re-running the whole phase afterwards found all 3,530 records done.

## 2. Format-tolerant slots next to the original interface (same items, Dream)

Original rows: the 10-03 records (bfcl_skel_k, bfcl_surplus, bfcl_swap, bfcl_estimate) and experiment C's
bfcl_onesided.jsonl. Identical canvases: same gen_ids.

| k | slot lengths | n | tolerant set acc % | original set acc % (same items) | difference (points) | identical canvases | wrong->right | right->wrong | tolerant parses % | original parses % |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exact | 400 | 90.0 | 89.8 | +0.3 | 397/400 | 1 | 0 | 99.2 | 99.0 |
| 1 | +1 | 400 | 28.2 | 15.5 | +12.7 | 180/400 | 66 | 15 | 84.8 | 82.2 |
| 1 | +2 | 400 | 20.0 | 3.2 | +16.8 | 112/400 | 69 | 2 | 79.8 | 75.2 |
| 1 | +8 | 400 | 67.2 | 60.0 | +7.3 | 190/400 | 43 | 14 | 94.5 | 89.5 |
| 16 | exact | 400 | 84.8 | 84.5 | +0.3 | 396/400 | 1 | 0 | 95.0 | 94.8 |
| 16 | +1 | 400 | 9.0 | 5.0 | +4.0 | 163/400 | 22 | 6 | 81.2 | 74.8 |
| 1 | swap | 165 | 67.3 | 67.3 | +0.0 | 164/165 | 0 | 0 | 99.4 | 98.8 |
| 1 | onesided | 165 | 18.2 | 17.6 | +0.6 | 164/165 | 1 | 0 | 98.8 | 98.8 |
| 1 | model estimate | 400 | 47.5 | 42.2 | +5.2 | 264/400 | 32 | 11 | 96.8 | 89.5 |
| 16 | model estimate | 400 | 38.2 | 32.2 | +6.0 | 231/400 | 32 | 8 | 93.5 | 82.0 |

## 3. Slot classes under surplus masks, and what the normalization removed (not in the plan)

scripts/pod/exp_d/slot_classes.py: every slot classified by scripts/closer_analysis.py's classify_value, tolerant
slots read as decoded (normalized), original slots from their own tokens.

Slot classes of every slot, Dream k=1 (% of slots):

| surplus | interface | n slots | own | overfill | truncated | sibling_fit | sibling | other |
|---|---|---|---|---|---|---|---|---|
| +1 | original | 3063 | 37.5 | 44.4 | 0.4 | 0.2 | 0.3 | 17.2 |
| +1 | tolerant | 3063 | 56.7 | 24.8 | 1.5 | 0.2 | 0.3 | 16.5 |
| +2 | original | 3063 | 14.5 | 55.7 | 0.1 | 0.1 | 0.0 | 29.6 |
| +2 | tolerant | 3063 | 42.5 | 25.9 | 0.3 | 0.0 | 0.5 | 30.8 |
| +8 | original | 3063 | 78.6 | 13.0 | 0.0 | 0.0 | 0.7 | 7.6 |
| +8 | tolerant | 3063 | 84.2 | 7.5 | 0.3 | 0.0 | 0.7 | 7.3 |

What the tolerant normalization removed (string / number / boolean slots, k=1):

| surplus | slots | with filler removed | decimal point (+ zeros) | digit group '_000' | escaped newline | letter suffix (i, f, L, y, ...) | other change | other suffix |
|---|---|---|---|---|---|---|---|---|
| +1 | 2862 | 1105 (38.6%) | 4 | 38 | 808 | 212 | 13 | 30 |
| +2 | 2862 | 1039 (36.3%) | 916 | 6 | 12 | 2 | 56 | 47 |
| +8 | 2862 | 165 (5.8%) | 66 | 6 | 0 | 0 | 32 | 61 |

## 4. Swap and one-sided lengthening: slot classes (scripts/pod/exp_d/swap_tolerant.py)

Computed as scripts/swap_direction.py (whose original rows it reproduces exactly), tolerant slots normalized.

| lengths | interface | slots | n slots | own | sibling_fit | overfill | other |
|---|---|---|---|---|---|---|---|
| swap | original | lengthened | 258 | 5.8 [2.8, 9.3] | 79.1 [71.8, 85.7] | 4.7 [1.2, 8.9] | 10.5 [5.9, 15.8] |
| swap | original | shortened | 256 | 5.1 [1.9, 9.2] | 79.7 [72.6, 85.9] | 0.0 [0.0, 0.0] | 15.2 [10.1, 21.0] |
| swap | tolerant | lengthened | 258 | 5.8 [2.8, 9.3] | 79.1 [71.8, 85.7] | 4.7 [1.2, 8.9] | 10.5 [5.9, 15.8] |
| swap | tolerant | shortened | 256 | 5.1 [1.9, 9.2] | 79.7 [72.6, 85.9] | 0.0 [0.0, 0.0] | 15.2 [10.1, 21.0] |
| onesided | original | lengthened | 243 | 21.8 [16.3, 27.6] | 37.0 [30.0, 44.2] | 20.2 [14.7, 26.0] | 21.0 [15.0, 27.5] |
| onesided | tolerant | lengthened | 243 | 23.0 [17.3, 29.0] | 36.6 [29.6, 43.9] | 19.3 [13.9, 25.1] | 21.0 [15.0, 27.5] |

## 5. An alternative number normalization, re-scored offline (not in the plan; scripts/pod/exp_d/renorm.py)

The plan's normalization keeps a number's longest well-formed prefix, so ' 15_000' -> ' 15' (wrong) and ' 30. 45' ->
' 30.' (not JSON). The alternative first drops '_' between digits and a '.' not followed by a digit. "reproduced":
the offline reconstruction with the plan's normalization gives every stored diagnosis.

| decoding | surplus | lengths | n | stored set acc % | reproduced | alternative set acc % | flips (wrong->right, right->wrong) |
|---|---|---|---|---|---|---|---|
| confidence_k16_tnone_bfull_T0.0 | 0 | oracle | 400 | 84.8 | 400/400 | 84.8 | 0, 0 |
| confidence_k16_tnone_bfull_T0.0 | 1 | oracle | 400 | 9.0 | 400/400 | 9.8 | 3, 0 |
| confidence_k1_tnone_bfull_T0.0 | 0 | oracle | 400 | 90.0 | 400/400 | 90.0 | 0, 0 |
| confidence_k1_tnone_bfull_T0.0 | 1 | oracle | 400 | 28.2 | 400/400 | 29.5 | 5, 0 |
| confidence_k1_tnone_bfull_T0.0 | 2 | oracle | 400 | 20.0 | 400/400 | 20.2 | 1, 0 |
| confidence_k1_tnone_bfull_T0.0 | 8 | oracle | 400 | 67.2 | 400/400 | 67.2 | 0, 0 |
| confidence_k16_tnone_bfull_T0.0 | 0 | length_estimate | 400 | 38.2 | 400/400 | 38.2 | 1, 1 |
| confidence_k1_tnone_bfull_T0.0 | 0 | length_estimate | 400 | 47.5 | 400/400 | 47.2 | 1, 2 |

## 6. Position agents (pos-anon)

Team metrics by scripts/agents_bfcl_analysis.py (BFCL team-symmetric subset, 104 requests), one-canvas rows on the
same requests; choose-N by experiment B's metrics (ok: all cities allowed and different; invalid includes '...'; in the open
variant invalid also reflects the one-token city slot, as in results/summary/agents.md). Dream's list row equals the
one-canvas k=16 row of results/summary/agents.md (ok 0.850, duplicate 0.150, in_order 0.117).

| model | lengths | protocol | teams | set_acc | ccer | duplicate | in_order | order_correct | in_order_clean | first_mention | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | oracle | pos-anon | 104 | 0.856 | 0.038 | 0.019 | 0.817 | 0.955 | 0.874 | 0.359 | 0.010 |
| Qwen2.5-7B-Instruct | oracle | pos-anon | 104 | 0.548 | 0.269 | 0.240 | 0.538 | 0.982 | 0.589 | 0.451 | 0.010 |

One canvas, same requests (10-03 records re-diagnosed with the current taxonomy, scripts/rediagnose.py, as in
results/summary/agents_bfcl.md since eeb27cf; set accuracy is unchanged by it):

| model | lengths | protocol | teams | set_acc | ccer | duplicate | in_order | order_correct | in_order_clean | first_mention | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dream canvas k=1 | oracle | one canvas | 104 | 0.885 | 0.019 | 0.000 | 0.837 | 0.946 | 0.895 | 0.366 | 0.010 |
| Dream canvas k=16 | oracle | one canvas | 104 | 0.817 | 0.048 | 0.019 | 0.788 | 0.965 | 0.842 | 0.348 | 0.077 |
| Qwen canvas (AR) | oracle | one canvas | 104 | 0.913 | 0.010 | 0.000 | 0.875 | 0.958 | 0.916 | 0.381 | 0.010 |

| model | protocol | variant | teams | ok | duplicate | all_same | invalid | in_order |
|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | pos-anon | list | 60 | 0.850 | 0.150 | 0.050 | 0.000 | 0.117 |
| Dream-v0-Instruct-7B | pos-anon | open | 45 | 0.000 | 1.000 | 0.667 | 0.000 | - |
| Qwen2.5-7B-Instruct | pos-anon | list | 60 | 0.000 | 0.650 | 0.000 | 1.000 | 0.000 |
| Qwen2.5-7B-Instruct | pos-anon | open | 45 | 0.000 | 0.667 | 0.044 | 1.000 | - |

Agents whose own call holds the placeholder ('...' for a string, null otherwise), by agent index, and choose-N
duplicates with and without '...' counted as a city (scripts/pod/exp_d/pos_extras.py):

| model | data | teams | agents | own call holds a placeholder | ... by agent index (1, 2, 3, 4+) |
|---|---|---|---|---|---|
| qwen | agents_d | 104 | 289 | 43 (14.9%) | 0/104, 13/104, 10/47, 20/34 |
| qwen | agents_pos | 105 | 315 | 188 (59.7%) | 0/105, 83/105, 70/70, 35/35 |
| dream | agents_d | 104 | 289 | 0 (0.0%) | 0/104, 0/104, 0/47, 0/34 |
| dream | agents_pos | 105 | 315 | 0 (0.0%) | 0/105, 0/105, 0/70, 0/35 |

choose-N teams: duplicate city among the agents, counting '...' as a city (as exp_d_quick.py) and without it; teams where some agent wrote '...'.

| model | variant | teams | duplicate (with '...') | duplicate (real cities only) | some agent wrote '...' |
|---|---|---|---|---|---|
| qwen | list | 60 | 65.0% | 0.0% | 100.0% |
| qwen | open | 45 | 66.7% | 8.9% | 93.3% |
| dream | list | 60 | 15.0% | 15.0% | 0.0% |
| dream | open | 45 | 100.0% | 100.0% | 0.0% |
