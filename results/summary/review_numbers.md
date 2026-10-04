# Numbers quoted in the text (scripts/review_numbers.py)

## swap_sets (Dream, swapped lengths, k=1)
- correct requests: 111/165 (67.3%)
- of these, correct only as a reordered set: 108/111 (97.3%)

## ar_cutoff (Qwen2.5, exact vs +8)
- requests: 400; output differs: 44/400 (11.0%)
- wrong with exact lengths, correct with +8: 13; the reverse: 3

## k_failures (exact lengths; fail at k=16 but not at k=1)
| model | newly failing | does not parse | cross-call | single-call only | newly correct |
|---|---|---|---|---|---|
| Dream | 21 | 5 | 3 | 13 | 0 |
| LLaDA2.0 | 33 | 7 | 5 | 21 | 1 |

## estimate (one-forward length estimates)
- Dream: wrong estimates 1481/3063 (48.4%); too long among wrong: 1355/1481 (91.5%)
- LLaDA2.0: wrong estimates 197/723 (27.2%); too long among wrong: 91/197 (46.2%)

## uniform (choose-N list, independent uniform choices among n+3 cities)
| n | items | P(collision) |
|---|---|---|
| 2 | 20 | 20.0% |
| 3 | 20 | 44.4% |
| 4 | 20 | 65.0% |
| all | 60 | 43.2% |

## open (choose-N open requests, greedy)
| model | decoding | requests | distinct outputs per n | calls with a city not allowed |
|---|---|---|---|---|
| dream | confidence_k16_tnone_bfull_T0.0 | 45 | {2: 1, 3: 1, 4: 1} | 0/135 (0.0%) |
| dream | confidence_k1_tnone_bfull_T0.0 | 45 | {2: 2, 3: 1, 4: 1} | 46/135 (34.1%) |
| dream | confidence_k2_tnone_bfull_T0.0 | 45 | {2: 1, 3: 2, 4: 4} | 7/135 (5.2%) |
| dream | confidence_k4_tnone_bfull_T0.0 | 45 | {2: 1, 3: 1, 4: 1} | 0/135 (0.0%) |
| llada2 | confidence_k16_tnone_b32_T0.0 | 45 | {2: 7, 3: 7, 4: 10} | 94/135 (69.6%) |
| llada2 | confidence_k1_tnone_b32_T0.0 | 45 | {2: 6, 3: 6, 4: 8} | 102/135 (75.6%) |
| llada2 | confidence_k4_tnone_b32_T0.0 | 45 | {2: 7, 3: 7, 4: 10} | 94/135 (69.6%) |
| qwen | ar_greedy | 45 | {2: 2, 3: 3, 4: 4} | 100/135 (74.1%) |

## free (Dream without the skeleton, confidence_k2_tnone_bfull_T0.0)
- requests: 400; correct: 301/400 (75.3%); do not parse: 50/400 (12.5%); cross-call: 22/400 (5.5%)
- requests with each label: syntax_error 50, wrong_value 22, omitted_call 9, missing_param 6, duplicate_call 5, extra_call 5, type_error 4, cross_binding 3, wrong_function 2, inconsistent_shared_arg 1, unexpected_param 1

## teams_by (experiment C, set accuracy of teams)
| model | subset | lengths | protocol | category | n | teams | correct |
|---|---|---|---|---|---|---|---|
| dream | sym | oracle | sim-anon | all | 2 | 57 | 0/57 (0.0%) |
| dream | sym | oracle | sim-anon | all | 3 | 21 | 0/21 (0.0%) |
| dream | sym | oracle | sim-anon | all | 4 | 24 | 0/24 (0.0%) |
| dream | sym | oracle | sim-anon | all | 8 | 2 | 0/2 (0.0%) |
| dream | sym | oracle | sim-anon | parallel | all | 84 | 0/84 (0.0%) |
| dream | sym | oracle | sim-anon | parallel_multiple | all | 20 | 0/20 (0.0%) |
| dream | sym | oracle | sim-label | all | 2 | 57 | 19/57 (33.3%) |
| dream | sym | oracle | sim-label | all | 3 | 21 | 3/21 (14.3%) |
| dream | sym | oracle | sim-label | all | 4 | 24 | 8/24 (33.3%) |
| dream | sym | oracle | sim-label | all | 8 | 2 | 0/2 (0.0%) |
| dream | sym | oracle | sim-label | parallel | all | 84 | 28/84 (33.3%) |
| dream | sym | oracle | sim-label | parallel_multiple | all | 20 | 2/20 (10.0%) |
| dream | sym | oracle | sim-rule | all | 2 | 57 | 36/57 (63.2%) |
| dream | sym | oracle | sim-rule | all | 3 | 21 | 7/21 (33.3%) |
| dream | sym | oracle | sim-rule | all | 4 | 24 | 7/24 (29.2%) |
| dream | sym | oracle | sim-rule | all | 8 | 2 | 0/2 (0.0%) |
| dream | sym | oracle | sim-rule | parallel | all | 84 | 47/84 (56.0%) |
| dream | sym | oracle | sim-rule | parallel_multiple | all | 20 | 3/20 (15.0%) |
| dream | sym | oracle | turn-anon | all | 2 | 57 | 47/57 (82.5%) |
| dream | sym | oracle | turn-anon | all | 3 | 21 | 18/21 (85.7%) |
| dream | sym | oracle | turn-anon | all | 4 | 24 | 22/24 (91.7%) |
| dream | sym | oracle | turn-anon | all | 8 | 2 | 1/2 (50.0%) |
| dream | sym | oracle | turn-anon | parallel | all | 84 | 71/84 (84.5%) |
| dream | sym | oracle | turn-anon | parallel_multiple | all | 20 | 17/20 (85.0%) |
| dream | sym | oracle | turn-label | all | 2 | 57 | 48/57 (84.2%) |
| dream | sym | oracle | turn-label | all | 3 | 21 | 18/21 (85.7%) |
| dream | sym | oracle | turn-label | all | 4 | 24 | 22/24 (91.7%) |
| dream | sym | oracle | turn-label | all | 8 | 2 | 1/2 (50.0%) |
| dream | sym | oracle | turn-label | parallel | all | 84 | 72/84 (85.7%) |
| dream | sym | oracle | turn-label | parallel_multiple | all | 20 | 17/20 (85.0%) |
| dream | swap | oracle | sim-anon | all | 2 | 59 | 45/59 (76.3%) |
| dream | swap | oracle | sim-anon | all | 3 | 44 | 18/44 (40.9%) |
| dream | swap | oracle | sim-anon | all | 4 | 60 | 17/60 (28.3%) |
| dream | swap | oracle | sim-anon | all | 5 | 1 | 0/1 (0.0%) |
| dream | swap | oracle | sim-anon | all | 6 | 1 | 0/1 (0.0%) |
| dream | swap | oracle | sim-anon | parallel | all | 116 | 56/116 (48.3%) |
| dream | swap | oracle | sim-anon | parallel_multiple | all | 49 | 24/49 (49.0%) |
| dream | swap | oracle | sim-label | all | 2 | 59 | 50/59 (84.7%) |
| dream | swap | oracle | sim-label | all | 3 | 44 | 34/44 (77.3%) |
| dream | swap | oracle | sim-label | all | 4 | 60 | 33/60 (55.0%) |
| dream | swap | oracle | sim-label | all | 5 | 1 | 0/1 (0.0%) |
| dream | swap | oracle | sim-label | all | 6 | 1 | 0/1 (0.0%) |
| dream | swap | oracle | sim-label | parallel | all | 116 | 82/116 (70.7%) |
| dream | swap | oracle | sim-label | parallel_multiple | all | 49 | 35/49 (71.4%) |
| dream | swap | oracle | turn-anon | all | 2 | 59 | 56/59 (94.9%) |
| dream | swap | oracle | turn-anon | all | 3 | 44 | 43/44 (97.7%) |
| dream | swap | oracle | turn-anon | all | 4 | 60 | 53/60 (88.3%) |
| dream | swap | oracle | turn-anon | all | 5 | 1 | 1/1 (100.0%) |
| dream | swap | oracle | turn-anon | all | 6 | 1 | 1/1 (100.0%) |
| dream | swap | oracle | turn-anon | parallel | all | 116 | 110/116 (94.8%) |
| dream | swap | oracle | turn-anon | parallel_multiple | all | 49 | 44/49 (89.8%) |
| dream | swap | oracle | turn-label | all | 2 | 59 | 56/59 (94.9%) |
| dream | swap | oracle | turn-label | all | 3 | 44 | 43/44 (97.7%) |
| dream | swap | oracle | turn-label | all | 4 | 60 | 53/60 (88.3%) |
| dream | swap | oracle | turn-label | all | 5 | 1 | 1/1 (100.0%) |
| dream | swap | oracle | turn-label | all | 6 | 1 | 1/1 (100.0%) |
| dream | swap | oracle | turn-label | parallel | all | 116 | 110/116 (94.8%) |
| dream | swap | oracle | turn-label | parallel_multiple | all | 49 | 44/49 (89.8%) |
| qwen | sym | oracle | sim-anon | all | 2 | 57 | 0/57 (0.0%) |
| qwen | sym | oracle | sim-anon | all | 3 | 21 | 0/21 (0.0%) |
| qwen | sym | oracle | sim-anon | all | 4 | 24 | 0/24 (0.0%) |
| qwen | sym | oracle | sim-anon | all | 8 | 2 | 0/2 (0.0%) |
| qwen | sym | oracle | sim-anon | parallel | all | 84 | 0/84 (0.0%) |
| qwen | sym | oracle | sim-anon | parallel_multiple | all | 20 | 0/20 (0.0%) |
| qwen | sym | oracle | sim-label | all | 2 | 57 | 0/57 (0.0%) |
| qwen | sym | oracle | sim-label | all | 3 | 21 | 0/21 (0.0%) |
| qwen | sym | oracle | sim-label | all | 4 | 24 | 0/24 (0.0%) |
| qwen | sym | oracle | sim-label | all | 8 | 2 | 0/2 (0.0%) |
| qwen | sym | oracle | sim-label | parallel | all | 84 | 0/84 (0.0%) |
| qwen | sym | oracle | sim-label | parallel_multiple | all | 20 | 0/20 (0.0%) |
| qwen | sym | oracle | sim-rule | all | 2 | 57 | 27/57 (47.4%) |
| qwen | sym | oracle | sim-rule | all | 3 | 21 | 6/21 (28.6%) |
| qwen | sym | oracle | sim-rule | all | 4 | 24 | 3/24 (12.5%) |
| qwen | sym | oracle | sim-rule | all | 8 | 2 | 0/2 (0.0%) |
| qwen | sym | oracle | sim-rule | parallel | all | 84 | 33/84 (39.3%) |
| qwen | sym | oracle | sim-rule | parallel_multiple | all | 20 | 3/20 (15.0%) |
| qwen | sym | oracle | turn-anon | all | 2 | 57 | 52/57 (91.2%) |
| qwen | sym | oracle | turn-anon | all | 3 | 21 | 17/21 (81.0%) |
| qwen | sym | oracle | turn-anon | all | 4 | 24 | 21/24 (87.5%) |
| qwen | sym | oracle | turn-anon | all | 8 | 2 | 2/2 (100.0%) |
| qwen | sym | oracle | turn-anon | parallel | all | 84 | 75/84 (89.3%) |
| qwen | sym | oracle | turn-anon | parallel_multiple | all | 20 | 17/20 (85.0%) |
| qwen | sym | oracle | turn-label | all | 2 | 57 | 52/57 (91.2%) |
| qwen | sym | oracle | turn-label | all | 3 | 21 | 17/21 (81.0%) |
| qwen | sym | oracle | turn-label | all | 4 | 24 | 21/24 (87.5%) |
| qwen | sym | oracle | turn-label | all | 8 | 2 | 2/2 (100.0%) |
| qwen | sym | oracle | turn-label | parallel | all | 84 | 75/84 (89.3%) |
| qwen | sym | oracle | turn-label | parallel_multiple | all | 20 | 17/20 (85.0%) |
| qwen | swap | oracle | sim-anon | all | 2 | 59 | 0/59 (0.0%) |
| qwen | swap | oracle | sim-anon | all | 3 | 44 | 1/44 (2.3%) |
| qwen | swap | oracle | sim-anon | all | 4 | 60 | 0/60 (0.0%) |
| qwen | swap | oracle | sim-anon | all | 5 | 1 | 0/1 (0.0%) |
| qwen | swap | oracle | sim-anon | all | 6 | 1 | 0/1 (0.0%) |
| qwen | swap | oracle | sim-anon | parallel | all | 116 | 1/116 (0.9%) |
| qwen | swap | oracle | sim-anon | parallel_multiple | all | 49 | 0/49 (0.0%) |
| qwen | swap | oracle | sim-label | all | 2 | 59 | 0/59 (0.0%) |
| qwen | swap | oracle | sim-label | all | 3 | 44 | 1/44 (2.3%) |
| qwen | swap | oracle | sim-label | all | 4 | 60 | 0/60 (0.0%) |
| qwen | swap | oracle | sim-label | all | 5 | 1 | 0/1 (0.0%) |
| qwen | swap | oracle | sim-label | all | 6 | 1 | 0/1 (0.0%) |
| qwen | swap | oracle | sim-label | parallel | all | 116 | 1/116 (0.9%) |
| qwen | swap | oracle | sim-label | parallel_multiple | all | 49 | 0/49 (0.0%) |
| qwen | swap | oracle | turn-anon | all | 2 | 59 | 49/59 (83.1%) |
| qwen | swap | oracle | turn-anon | all | 3 | 44 | 40/44 (90.9%) |
| qwen | swap | oracle | turn-anon | all | 4 | 60 | 46/60 (76.7%) |
| qwen | swap | oracle | turn-anon | all | 5 | 1 | 1/1 (100.0%) |
| qwen | swap | oracle | turn-anon | all | 6 | 1 | 0/1 (0.0%) |
| qwen | swap | oracle | turn-anon | parallel | all | 116 | 93/116 (80.2%) |
| qwen | swap | oracle | turn-anon | parallel_multiple | all | 49 | 43/49 (87.8%) |
| qwen | swap | oracle | turn-label | all | 2 | 59 | 49/59 (83.1%) |
| qwen | swap | oracle | turn-label | all | 3 | 44 | 39/44 (88.6%) |
| qwen | swap | oracle | turn-label | all | 4 | 60 | 46/60 (76.7%) |
| qwen | swap | oracle | turn-label | all | 5 | 1 | 1/1 (100.0%) |
| qwen | swap | oracle | turn-label | all | 6 | 1 | 0/1 (0.0%) |
| qwen | swap | oracle | turn-label | parallel | all | 116 | 93/116 (80.2%) |
| qwen | swap | oracle | turn-label | parallel_multiple | all | 49 | 42/49 (85.7%) |
