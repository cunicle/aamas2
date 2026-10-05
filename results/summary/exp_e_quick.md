# Experiment E: quick read-out

### (a) Dream, closing token in the slot, k=1 (bfcl_closer.jsonl)

| s | decoding | requests | set accuracy % |
|---|---|---|---|
| 0 | confidence_k1_tnone_bfull_T0.0 | 400 | 70.2 |
| 1 | confidence_k1_tnone_bfull_T0.0 | 400 | 26.8 |
| 2 | confidence_k1_tnone_bfull_T0.0 | 400 | 16.8 |
| 4 | confidence_k1_tnone_bfull_T0.0 | 400 | 23.5 |
| 8 | confidence_k1_tnone_bfull_T0.0 | 400 | 39.5 |

### (a) Qwen2.5 AR (bfcl_skeleton.jsonl s=0, bfcl_surplus.jsonl)

| s | decoding | requests | set accuracy % |
|---|---|---|---|
| 0 | ar_greedy | 400 | 86.5 |
| 1 | ar_greedy | 400 | 88.0 |
| 2 | ar_greedy | 400 | 88.5 |
| 4 | ar_greedy | 400 | 88.8 |
| 8 | ar_greedy | 400 | 89.0 |

### (b) LLaDA2.0, 100 requests (bfcl_surplus.jsonl; s=0 is in bfcl_skel_k.jsonl)

| s | decoding | requests | set accuracy % |
|---|---|---|---|
| 1 | confidence_k16_tnone_b32_T0.0 | 100 | 4.0 |
| 2 | confidence_k16_tnone_b32_T0.0 | 100 | 2.0 |
| 4 | confidence_k16_tnone_b32_T0.0 | 100 | 22.0 |
| 8 | confidence_k16_tnone_b32_T0.0 | 100 | 38.0 |
| 1 | confidence_k1_tnone_b32_T0.0 | 100 | 16.0 |
| 2 | confidence_k1_tnone_b32_T0.0 | 100 | 8.0 |
| 4 | confidence_k1_tnone_b32_T0.0 | 100 | 17.0 |
| 8 | confidence_k1_tnone_b32_T0.0 | 100 | 38.0 |
| 1 | confidence_k4_tnone_b32_T0.0 | 100 | 12.0 |
| 2 | confidence_k4_tnone_b32_T0.0 | 100 | 7.0 |
| 4 | confidence_k4_tnone_b32_T0.0 | 100 | 25.0 |
| 8 | confidence_k4_tnone_b32_T0.0 | 100 | 35.0 |

### (c) probe: mean P(closing token) over slots

| model | interface | s | slots | mean P(close) % |
|---|---|---|---|---|
| Dream | closing token in slot | 0 | 3063 | 60.5 |
| Dream | closing token in slot | 1 | 3063 | 43.0 |
| Dream | closing token in slot | 2 | 3063 | 34.5 |
| Dream | closing token in slot | 4 | 3063 | 52.9 |
| Dream | closing token in slot | 8 | 3063 | 81.6 |
| Dream | original | 1 | 3063 | 1.0 |
| Dream | original | 2 | 3063 | 3.3 |
| Dream | original | 4 | 3063 | 25.4 |
| Dream | original | 8 | 3063 | 74.0 |
| LLaDA2.0 | original | 1 | 3063 | 19.3 |
| LLaDA2.0 | original | 2 | 3063 | 17.4 |
| LLaDA2.0 | original | 4 | 3063 | 53.3 |
| LLaDA2.0 | original | 8 | 3063 | 91.8 |
