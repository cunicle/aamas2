# Experiment D (scripts/exp_d_analysis.py)

## 1. Format-tolerant slots against the original interface (Dream, same requests)

| decoding | surplus | lengths | n | tolerant set acc % | original set acc % | wrong -> right | right -> wrong | identical canvases | tolerant parses % | original parses % |
|---|---|---|---|---|---|---|---|---|---|---|
| confidence_k16_tnone_bfull_T0.0 | 0 | length_estimate | 400 | 38.3 | 32.3 | 32 | 8 | 231/400 | 93.5 | 82.0 |
| confidence_k1_tnone_bfull_T0.0 | 0 | length_estimate | 400 | 47.5 | 42.3 | 32 | 11 | 264/400 | 96.8 | 89.5 |
| confidence_k1_tnone_bfull_T0.0 | 0 | onesided | 165 | 18.2 | 17.6 | 1 | 0 | 164/165 | 98.8 | 98.8 |
| confidence_k16_tnone_bfull_T0.0 | 0 | oracle | 400 | 84.8 | 84.5 | 1 | 0 | 396/400 | 95.0 | 94.8 |
| confidence_k1_tnone_bfull_T0.0 | 0 | oracle | 400 | 90.0 | 89.8 | 1 | 0 | 397/400 | 99.3 | 99.0 |
| confidence_k16_tnone_bfull_T0.0 | 1 | oracle | 400 | 9.0 | 5.0 | 22 | 6 | 163/400 | 81.3 | 74.8 |
| confidence_k1_tnone_bfull_T0.0 | 1 | oracle | 400 | 28.3 | 15.5 | 66 | 15 | 180/400 | 84.8 | 82.3 |
| confidence_k1_tnone_bfull_T0.0 | 2 | oracle | 400 | 20.0 | 3.3 | 69 | 2 | 112/400 | 79.8 | 75.3 |
| confidence_k1_tnone_bfull_T0.0 | 8 | oracle | 400 | 67.3 | 60.0 | 43 | 14 | 190/400 | 94.5 | 89.5 |
| confidence_k1_tnone_bfull_T0.0 | 0 | swap | 165 | 67.3 | 67.3 | 0 | 0 | 164/165 | 99.4 | 98.8 |

## 2. Slot classes with surplus masks (Dream, k=1, % of slots)

| surplus | interface | n slots | own | overfill | sibling_fit | other |
|---|---|---|---|---|---|---|
| +1 | original | 3063 | 37.5 | 44.4 | 0.2 | 17.9 |
| +1 | tolerant | 3063 | 56.7 | 24.8 | 0.2 | 18.3 |
| +2 | original | 3063 | 14.5 | 55.7 | 0.1 | 29.7 |
| +2 | tolerant | 3063 | 42.5 | 25.9 | 0.0 | 31.6 |
| +8 | original | 3063 | 78.6 | 13.0 | 0.0 | 8.4 |
| +8 | tolerant | 3063 | 84.2 | 7.5 | 0.0 | 8.3 |

What the normalization changed (tolerant string, number and boolean slots, k=1):

| surplus | slots | changed | escaped line break | decimal point | digit group | letter suffix | other |
|---|---|---|---|---|---|---|---|
| +1 | 2862 | 1330 (46.5%) | 808 | 4 | 36 | 210 | 272 |
| +2 | 2862 | 1040 (36.3%) | 21 | 883 | 6 | 2 | 128 |
| +8 | 2862 | 181 (6.3%) | 0 | 70 | 6 | 8 | 97 |

## 3. Position agents (pos-anon)

| model | data | teams | correct | agents | own call holds a placeholder | by agent index (1, 2, 3, 4+) |
|---|---|---|---|---|---|---|
| dream | agents_d | 104 | 85.6% | 289 | 0 (0.0%) | 0/104, 0/104, 0/47, 0/34 |
| dream | agents_pos | 105 | -- | 315 | 0 (0.0%) | 0/105, 0/105, 0/70, 0/35 |
| qwen | agents_d | 104 | 54.8% | 289 | 43 (14.9%) | 0/104, 13/104, 10/47, 20/34 |
| qwen | agents_pos | 105 | -- | 315 | 188 (59.7%) | 0/105, 83/105, 70/70, 35/35 |

choose-N teams of position agents: a duplicate city, counting '...' as a city or not

| model | variant | teams | duplicate | duplicate among real cities | some agent wrote '...' |
|---|---|---|---|---|---|
| dream | list | 60 | 15.0% | 15.0% | 0.0% |
| dream | open | 45 | 100.0% | 100.0% | 0.0% |
| qwen | list | 60 | 65.0% | 0.0% | 100.0% |
| qwen | open | 45 | 66.7% | 8.9% | 93.3% |
