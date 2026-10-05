# Experiment B: teams of n agents on choose-N

Team level. ok: all cities allowed and different; duplicate: two agents with the same city; all_same: all n agents with one city; invalid: a city outside the allowed set or an unparsed output; in_order: agent i took the i-th listed city (list only). In the open variant `invalid` reflects the one-token slot; read its duplicate rates only.

## All teams

| model | protocol | variant | temperature | teams | ok | duplicate | all_same | invalid | in_order |
|---|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | sim-anon | list | 0.0 | 60 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-anon | open | 0.0 | 45 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | list | 0.0 | 60 | 0.217 | 0.783 | 0.400 | 0.000 | 0.033 |
| Dream-v0-Instruct-7B | sim-label | open | 0.0 | 45 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | turn-anon | list | 0.0 | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.350 |
| Dream-v0-Instruct-7B | turn-anon | open | 0.0 | 45 | 0.244 | 0.000 | 0.000 | 0.756 | - |
| Dream-v0-Instruct-7B | turn-label | list | 0.0 | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.267 |
| Dream-v0-Instruct-7B | turn-label | open | 0.0 | 45 | 0.311 | 0.000 | 0.000 | 0.689 | - |
| Dream-v0-Instruct-7B | pos-anon | list | 0.0 | 60 | 0.850 | 0.150 | 0.050 | 0.000 | 0.117 |
| Dream-v0-Instruct-7B | pos-anon | open | 0.0 | 45 | 0.000 | 1.000 | 0.667 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-anon | list | 0.7 | 300 | 0.107 | 0.877 | 0.593 | 0.030 | 0.017 |
| Dream-v0-Instruct-7B | sim-anon | open | 0.7 | 225 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | list | 0.7 | 300 | 0.227 | 0.750 | 0.340 | 0.040 | 0.027 |
| Dream-v0-Instruct-7B | sim-label | open | 0.7 | 225 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.0 | 60 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.0 | 45 | 0.000 | 1.000 | 1.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | list | 0.0 | 60 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | open | 0.0 | 45 | 0.000 | 0.978 | 0.978 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-anon | list | 0.0 | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.917 |
| Qwen2.5-7B-Instruct | turn-anon | open | 0.0 | 45 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-label | list | 0.0 | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.917 |
| Qwen2.5-7B-Instruct | turn-label | open | 0.0 | 45 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | pos-anon | list | 0.0 | 60 | 0.000 | 0.650 | 0.000 | 1.000 | 0.000 |
| Qwen2.5-7B-Instruct | pos-anon | open | 0.0 | 45 | 0.000 | 0.667 | 0.044 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.7 | 300 | 0.003 | 0.997 | 0.993 | 0.000 | 0.003 |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.7 | 225 | 0.000 | 0.938 | 0.898 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | list | 0.7 | 300 | 0.000 | 1.000 | 0.973 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | open | 0.7 | 225 | 0.000 | 0.938 | 0.844 | 1.000 | - |

## By team size n

| model | protocol | variant | temperature | n | teams | ok | duplicate | all_same | invalid | in_order |
|---|---|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | sim-anon | list | 0.0 | 2 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-anon | list | 0.0 | 3 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-anon | list | 0.0 | 4 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-anon | open | 0.0 | 2 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-anon | open | 0.0 | 3 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-anon | open | 0.0 | 4 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | list | 0.0 | 2 | 20 | 0.450 | 0.550 | 0.550 | 0.000 | 0.100 |
| Dream-v0-Instruct-7B | sim-label | list | 0.0 | 3 | 20 | 0.150 | 0.850 | 0.150 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-label | list | 0.0 | 4 | 20 | 0.050 | 0.950 | 0.500 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-label | open | 0.0 | 2 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | open | 0.0 | 3 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | open | 0.0 | 4 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | turn-anon | list | 0.0 | 2 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.300 |
| Dream-v0-Instruct-7B | turn-anon | list | 0.0 | 3 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.600 |
| Dream-v0-Instruct-7B | turn-anon | list | 0.0 | 4 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.150 |
| Dream-v0-Instruct-7B | turn-anon | open | 0.0 | 2 | 15 | 0.733 | 0.000 | 0.000 | 0.267 | - |
| Dream-v0-Instruct-7B | turn-anon | open | 0.0 | 3 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Dream-v0-Instruct-7B | turn-anon | open | 0.0 | 4 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Dream-v0-Instruct-7B | turn-label | list | 0.0 | 2 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.250 |
| Dream-v0-Instruct-7B | turn-label | list | 0.0 | 3 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.350 |
| Dream-v0-Instruct-7B | turn-label | list | 0.0 | 4 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.200 |
| Dream-v0-Instruct-7B | turn-label | open | 0.0 | 2 | 15 | 0.933 | 0.000 | 0.000 | 0.067 | - |
| Dream-v0-Instruct-7B | turn-label | open | 0.0 | 3 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Dream-v0-Instruct-7B | turn-label | open | 0.0 | 4 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Dream-v0-Instruct-7B | pos-anon | list | 0.0 | 2 | 20 | 0.850 | 0.150 | 0.150 | 0.000 | 0.150 |
| Dream-v0-Instruct-7B | pos-anon | list | 0.0 | 3 | 20 | 0.900 | 0.100 | 0.000 | 0.000 | 0.150 |
| Dream-v0-Instruct-7B | pos-anon | list | 0.0 | 4 | 20 | 0.800 | 0.200 | 0.000 | 0.000 | 0.050 |
| Dream-v0-Instruct-7B | pos-anon | open | 0.0 | 2 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | pos-anon | open | 0.0 | 3 | 15 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | pos-anon | open | 0.0 | 4 | 15 | 0.000 | 1.000 | 0.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-anon | list | 0.7 | 2 | 100 | 0.250 | 0.720 | 0.720 | 0.030 | 0.050 |
| Dream-v0-Instruct-7B | sim-anon | list | 0.7 | 3 | 100 | 0.050 | 0.940 | 0.560 | 0.020 | 0.000 |
| Dream-v0-Instruct-7B | sim-anon | list | 0.7 | 4 | 100 | 0.020 | 0.970 | 0.500 | 0.040 | 0.000 |
| Dream-v0-Instruct-7B | sim-anon | open | 0.7 | 2 | 75 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-anon | open | 0.7 | 3 | 75 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-anon | open | 0.7 | 4 | 75 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | list | 0.7 | 2 | 100 | 0.450 | 0.520 | 0.520 | 0.030 | 0.070 |
| Dream-v0-Instruct-7B | sim-label | list | 0.7 | 3 | 100 | 0.210 | 0.760 | 0.200 | 0.040 | 0.010 |
| Dream-v0-Instruct-7B | sim-label | list | 0.7 | 4 | 100 | 0.020 | 0.970 | 0.300 | 0.050 | 0.000 |
| Dream-v0-Instruct-7B | sim-label | open | 0.7 | 2 | 75 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | open | 0.7 | 3 | 75 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Dream-v0-Instruct-7B | sim-label | open | 0.7 | 4 | 75 | 0.000 | 1.000 | 1.000 | 0.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.0 | 2 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.0 | 3 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.0 | 4 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.0 | 2 | 15 | 0.000 | 1.000 | 1.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.0 | 3 | 15 | 0.000 | 1.000 | 1.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.0 | 4 | 15 | 0.000 | 1.000 | 1.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | list | 0.0 | 2 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | list | 0.0 | 3 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | list | 0.0 | 4 | 20 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | open | 0.0 | 2 | 15 | 0.000 | 0.933 | 0.933 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | open | 0.0 | 3 | 15 | 0.000 | 1.000 | 1.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | open | 0.0 | 4 | 15 | 0.000 | 1.000 | 1.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-anon | list | 0.0 | 2 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.850 |
| Qwen2.5-7B-Instruct | turn-anon | list | 0.0 | 3 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| Qwen2.5-7B-Instruct | turn-anon | list | 0.0 | 4 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.900 |
| Qwen2.5-7B-Instruct | turn-anon | open | 0.0 | 2 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-anon | open | 0.0 | 3 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-anon | open | 0.0 | 4 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-label | list | 0.0 | 2 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.950 |
| Qwen2.5-7B-Instruct | turn-label | list | 0.0 | 3 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| Qwen2.5-7B-Instruct | turn-label | list | 0.0 | 4 | 20 | 1.000 | 0.000 | 0.000 | 0.000 | 0.800 |
| Qwen2.5-7B-Instruct | turn-label | open | 0.0 | 2 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-label | open | 0.0 | 3 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | turn-label | open | 0.0 | 4 | 15 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | pos-anon | list | 0.0 | 2 | 20 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| Qwen2.5-7B-Instruct | pos-anon | list | 0.0 | 3 | 20 | 0.000 | 0.950 | 0.000 | 1.000 | 0.000 |
| Qwen2.5-7B-Instruct | pos-anon | list | 0.0 | 4 | 20 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| Qwen2.5-7B-Instruct | pos-anon | open | 0.0 | 2 | 15 | 0.000 | 0.133 | 0.133 | 1.000 | - |
| Qwen2.5-7B-Instruct | pos-anon | open | 0.0 | 3 | 15 | 0.000 | 0.867 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | pos-anon | open | 0.0 | 4 | 15 | 0.000 | 1.000 | 0.000 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.7 | 2 | 100 | 0.010 | 0.990 | 0.990 | 0.000 | 0.010 |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.7 | 3 | 100 | 0.000 | 1.000 | 0.990 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-anon | list | 0.7 | 4 | 100 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.7 | 2 | 75 | 0.000 | 0.827 | 0.827 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.7 | 3 | 75 | 0.000 | 0.987 | 0.933 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-anon | open | 0.7 | 4 | 75 | 0.000 | 1.000 | 0.933 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | list | 0.7 | 2 | 100 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | list | 0.7 | 3 | 100 | 0.000 | 1.000 | 0.990 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | list | 0.7 | 4 | 100 | 0.000 | 1.000 | 0.930 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | open | 0.7 | 2 | 75 | 0.000 | 0.827 | 0.827 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | open | 0.7 | 3 | 75 | 0.000 | 0.987 | 0.880 | 1.000 | - |
| Qwen2.5-7B-Instruct | sim-label | open | 0.7 | 4 | 75 | 0.000 | 1.000 | 0.827 | 1.000 | - |

## For comparison: one canvas for all n calls (from results/summary/choose.csv; n = items)

| model | cfg | variant | n | ok | duplicate | same_step | invalid | in_order |
|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | confidence_k16_tnone_bfull_T0.0 | list | 60 | 0.850 | 0.150 | 0.150 | 0.000 | 0.117 |
| Dream-v0-Instruct-7B | confidence_k16_tnone_bfull_T0.0 | open | 45 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | confidence_k1_tnone_bfull_T0.0 | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.483 |
| Dream-v0-Instruct-7B | confidence_k1_tnone_bfull_T0.0 | open | 45 | 0.311 | 0.000 | 0.000 | 0.689 | 0.000 |
| Dream-v0-Instruct-7B | confidence_k2_tnone_bfull_T0.0 | list | 60 | 0.900 | 0.100 | 0.100 | 0.000 | 0.317 |
| Dream-v0-Instruct-7B | confidence_k2_tnone_bfull_T0.0 | open | 45 | 0.000 | 1.000 | 1.000 | 0.156 | 0.000 |
| Dream-v0-Instruct-7B | confidence_k4_tnone_bfull_T0.0 | list | 60 | 0.850 | 0.150 | 0.150 | 0.000 | 0.117 |
| Dream-v0-Instruct-7B | confidence_k4_tnone_bfull_T0.0 | open | 45 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| LLaDA2.0-mini | confidence_k16_tnone_b32_T0.0 | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.967 |
| LLaDA2.0-mini | confidence_k16_tnone_b32_T0.0 | open | 45 | 0.000 | 0.044 | 0.044 | 1.000 | 0.000 |
| LLaDA2.0-mini | confidence_k1_tnone_b32_T0.0 | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| LLaDA2.0-mini | confidence_k1_tnone_b32_T0.0 | open | 45 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| LLaDA2.0-mini | confidence_k4_tnone_b32_T0.0 | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.967 |
| LLaDA2.0-mini | confidence_k4_tnone_b32_T0.0 | open | 45 | 0.000 | 0.044 | 0.044 | 1.000 | 0.000 |
| Qwen2.5-7B-Instruct | ar_greedy | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.817 |
| Qwen2.5-7B-Instruct | ar_greedy | open | 45 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
