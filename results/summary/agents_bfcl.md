# Experiment C1 / C2: teams of agents on the BFCL parallel requests

One agent per reference call, each with its call's single-call skeleton; greedy decoding (Dream: k=1, confidence order). Team metrics: set_acc (the n calls as one array, set-level diagnosis), ccer (cross-call error label), duplicate (duplicate_call label), in_order (agent i passes reference call i, all i), order_correct (in_order among the correct teams: the paper's 95-97% measure), in_order_clean (C1: in_order on the 95 requests whose reference order is the request's mention order), first_mention (agents of a sibling group that pass the group's first reference call), unparsed (some agent did not make exactly one call). The one-canvas rows are the 10-03 runs on the same requests, by the same functions (canvas call i = agent i).

## C1: order cue, team-symmetric requests (104 items), oracle slot lengths

| model | lengths | protocol | teams | set_acc | ccer | duplicate | in_order | order_correct | in_order_clean | first_mention | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | oracle | sim-anon | 104 | 0.000 | 0.990 | 0.990 | 0.000 | nan | 0.000 | 0.842 | 0.010 |
| Dream-v0-Instruct-7B | oracle | sim-label | 104 | 0.288 | 0.654 | 0.644 | 0.288 | 1.000 | 0.316 | 0.678 | 0.010 |
| Dream-v0-Instruct-7B | oracle | turn-anon | 104 | 0.846 | 0.048 | 0.010 | 0.740 | 0.875 | 0.811 | 0.363 | 0.019 |
| Dream-v0-Instruct-7B | oracle | turn-label | 104 | 0.856 | 0.038 | 0.010 | 0.740 | 0.865 | 0.811 | 0.366 | 0.019 |
| Dream-v0-Instruct-7B | oracle | sim-rule | 104 | 0.481 | 0.452 | 0.442 | 0.471 | 0.980 | 0.516 | 0.579 | 0.010 |
| Dream-v0-Instruct-7B | oracle | pos-anon | 104 | 0.856 | 0.038 | 0.019 | 0.817 | 0.955 | 0.874 | 0.359 | 0.010 |
| Qwen2.5-7B-Instruct | oracle | sim-anon | 104 | 0.000 | 0.981 | 0.981 | 0.000 | nan | 0.000 | 0.879 | 0.019 |
| Qwen2.5-7B-Instruct | oracle | sim-label | 104 | 0.000 | 0.981 | 0.981 | 0.000 | nan | 0.000 | 0.883 | 0.019 |
| Qwen2.5-7B-Instruct | oracle | turn-anon | 104 | 0.885 | 0.019 | 0.000 | 0.837 | 0.946 | 0.884 | 0.374 | 0.019 |
| Qwen2.5-7B-Instruct | oracle | turn-label | 104 | 0.885 | 0.019 | 0.000 | 0.827 | 0.935 | 0.884 | 0.374 | 0.019 |
| Qwen2.5-7B-Instruct | oracle | sim-rule | 104 | 0.346 | 0.606 | 0.596 | 0.337 | 0.972 | 0.368 | 0.678 | 0.019 |
| Qwen2.5-7B-Instruct | oracle | pos-anon | 104 | 0.548 | 0.269 | 0.240 | 0.538 | 0.982 | 0.589 | 0.451 | 0.010 |

One canvas, same requests:

| model | lengths | protocol | teams | set_acc | ccer | duplicate | in_order | order_correct | in_order_clean | first_mention | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dream canvas k=1 | oracle | one canvas | 104 | 0.885 | 0.019 | 0.000 | 0.837 | 0.946 | 0.895 | 0.366 | 0.010 |
| Dream canvas k=16 | oracle | one canvas | 104 | 0.817 | 0.048 | 0.019 | 0.788 | 0.965 | 0.842 | 0.348 | 0.077 |
| LLaDA2.0 canvas k=1 | oracle | one canvas | 104 | 0.885 | 0.010 | 0.000 | 0.846 | 0.957 | 0.895 | 0.370 | 0.067 |
| LLaDA2.0 canvas k=16 | oracle | one canvas | 104 | 0.817 | 0.029 | 0.000 | 0.779 | 0.953 | 0.821 | 0.348 | 0.087 |
| Qwen canvas (AR) | oracle | one canvas | 104 | 0.913 | 0.010 | 0.000 | 0.875 | 0.958 | 0.916 | 0.381 | 0.010 |

### Choose-N positive control: sim-rule on the 60 list items (experiment B's metrics)

ok: all cities allowed and different; duplicate: two agents with one city; in_order: agent i took the i-th listed city. Experiment B's greedy list rows (results/summary/agents.csv) below for comparison.

| model | protocol | variant | teams | ok | duplicate | all_same | invalid | in_order |
|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | sim-rule | list | 60 | 0.550 | 0.450 | 0.133 | 0.000 | 0.100 |
| Qwen2.5-7B-Instruct | sim-rule | list | 60 | 0.300 | 0.700 | 0.333 | 0.000 | 0.083 |
| Dream-v0-Instruct-7B | sim-anon | list | 60 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Dream-v0-Instruct-7B | sim-label | list | 60 | 0.217 | 0.783 | 0.400 | 0.000 | 0.033 |
| Dream-v0-Instruct-7B | turn-anon | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.350 |
| Dream-v0-Instruct-7B | turn-label | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.267 |
| Dream-v0-Instruct-7B | pos-anon | list | 60 | 0.850 | 0.150 | 0.050 | 0.000 | 0.117 |
| Qwen2.5-7B-Instruct | sim-anon | list | 60 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | sim-label | list | 60 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Qwen2.5-7B-Instruct | turn-anon | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.917 |
| Qwen2.5-7B-Instruct | turn-label | list | 60 | 1.000 | 0.000 | 0.000 | 0.000 | 0.917 |
| Qwen2.5-7B-Instruct | pos-anon | list | 60 | 0.000 | 0.650 | 0.000 | 1.000 | 0.000 |

## C2: length cue, requests where the swap changes a slot (165 items)

| model | lengths | protocol | teams | set_acc | ccer | duplicate | in_order | order_correct | first_mention | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | oracle | sim-anon | 165 | 0.485 | 0.479 | 0.352 | 0.485 | 1.000 | 0.502 | 0.012 |
| Dream-v0-Instruct-7B | oracle | sim-label | 165 | 0.709 | 0.248 | 0.182 | 0.709 | 1.000 | 0.446 | 0.006 |
| Dream-v0-Instruct-7B | oracle | turn-anon | 165 | 0.933 | 0.006 | 0.006 | 0.921 | 0.987 | 0.393 | 0.000 |
| Dream-v0-Instruct-7B | oracle | turn-label | 165 | 0.933 | 0.006 | 0.006 | 0.915 | 0.981 | 0.393 | 0.000 |
| Dream-v0-Instruct-7B | swap | sim-anon | 165 | 0.467 | 0.485 | 0.352 | 0.006 | 0.013 | 0.494 | 0.012 |
| Dream-v0-Instruct-7B | swap | sim-label | 165 | 0.515 | 0.424 | 0.261 | 0.006 | 0.012 | 0.444 | 0.012 |
| Dream-v0-Instruct-7B | swap | turn-anon | 165 | 0.758 | 0.079 | 0.006 | 0.012 | 0.016 | 0.347 | 0.018 |
| Dream-v0-Instruct-7B | swap | turn-label | 165 | 0.770 | 0.079 | 0.000 | 0.018 | 0.024 | 0.351 | 0.018 |
| Qwen2.5-7B-Instruct | oracle | sim-anon | 165 | 0.006 | 0.873 | 0.697 | 0.006 | 1.000 | 0.669 | 0.055 |
| Qwen2.5-7B-Instruct | oracle | sim-label | 165 | 0.006 | 0.867 | 0.691 | 0.006 | 1.000 | 0.692 | 0.055 |
| Qwen2.5-7B-Instruct | oracle | turn-anon | 165 | 0.824 | 0.055 | 0.012 | 0.818 | 0.993 | 0.391 | 0.018 |
| Qwen2.5-7B-Instruct | oracle | turn-label | 165 | 0.818 | 0.055 | 0.012 | 0.812 | 0.993 | 0.389 | 0.018 |

One canvas, same requests:

| model | lengths | protocol | teams | set_acc | ccer | duplicate | in_order | order_correct | first_mention | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| Dream canvas k=1 | oracle | one canvas | 165 | 0.933 | 0.006 | 0.000 | 0.921 | 0.987 | 0.391 | 0.000 |
| Dream canvas k=1 swap | swap | one canvas | 165 | 0.673 | 0.152 | 0.000 | 0.018 | 0.027 | 0.301 | 0.012 |
| Qwen canvas (AR) | oracle | one canvas | 165 | 0.855 | 0.012 | 0.000 | 0.848 | 0.993 | 0.385 | 0.018 |

### C2 slot level: the slots the swap changes

own / sibling_fit (a sibling's value of exactly the slot's length) / sibling_other / other (incl. missing and unparsed); dir: the swap makes the slot longer or shorter than its own value.

| model | lengths | protocol | dir | n_slots | own | sibling_fit | sibling_other | other |
|---|---|---|---|---|---|---|---|---|
| Dream-v0-Instruct-7B | oracle | sim-anon | all | 514 | 0.858 | 0.049 | 0.021 | 0.072 |
| Dream-v0-Instruct-7B | oracle | sim-anon | longer | 258 | 0.857 | 0.050 | 0.027 | 0.066 |
| Dream-v0-Instruct-7B | oracle | sim-anon | shorter | 256 | 0.859 | 0.047 | 0.016 | 0.078 |
| Dream-v0-Instruct-7B | oracle | sim-label | all | 514 | 0.928 | 0.025 | 0.000 | 0.047 |
| Dream-v0-Instruct-7B | oracle | sim-label | longer | 258 | 0.919 | 0.027 | 0.000 | 0.054 |
| Dream-v0-Instruct-7B | oracle | sim-label | shorter | 256 | 0.938 | 0.023 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | oracle | turn-anon | all | 514 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | oracle | turn-anon | longer | 258 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | oracle | turn-anon | shorter | 256 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | oracle | turn-label | all | 514 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | oracle | turn-label | longer | 258 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | oracle | turn-label | shorter | 256 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream-v0-Instruct-7B | swap | sim-anon | all | 514 | 0.019 | 0.891 | 0.006 | 0.084 |
| Dream-v0-Instruct-7B | swap | sim-anon | longer | 258 | 0.019 | 0.880 | 0.004 | 0.097 |
| Dream-v0-Instruct-7B | swap | sim-anon | shorter | 256 | 0.020 | 0.902 | 0.008 | 0.070 |
| Dream-v0-Instruct-7B | swap | sim-label | all | 514 | 0.018 | 0.883 | 0.002 | 0.097 |
| Dream-v0-Instruct-7B | swap | sim-label | longer | 258 | 0.019 | 0.864 | 0.004 | 0.112 |
| Dream-v0-Instruct-7B | swap | sim-label | shorter | 256 | 0.016 | 0.902 | 0.000 | 0.082 |
| Dream-v0-Instruct-7B | swap | turn-anon | all | 514 | 0.023 | 0.842 | 0.004 | 0.130 |
| Dream-v0-Instruct-7B | swap | turn-anon | longer | 258 | 0.019 | 0.837 | 0.008 | 0.136 |
| Dream-v0-Instruct-7B | swap | turn-anon | shorter | 256 | 0.027 | 0.848 | 0.000 | 0.125 |
| Dream-v0-Instruct-7B | swap | turn-label | all | 514 | 0.033 | 0.852 | 0.002 | 0.113 |
| Dream-v0-Instruct-7B | swap | turn-label | longer | 258 | 0.031 | 0.845 | 0.004 | 0.120 |
| Dream-v0-Instruct-7B | swap | turn-label | shorter | 256 | 0.035 | 0.859 | 0.000 | 0.105 |
| Qwen2.5-7B-Instruct | oracle | sim-anon | all | 514 | 0.440 | 0.041 | 0.274 | 0.245 |
| Qwen2.5-7B-Instruct | oracle | sim-anon | longer | 258 | 0.477 | 0.039 | 0.039 | 0.446 |
| Qwen2.5-7B-Instruct | oracle | sim-anon | shorter | 256 | 0.402 | 0.043 | 0.512 | 0.043 |
| Qwen2.5-7B-Instruct | oracle | sim-label | all | 514 | 0.438 | 0.043 | 0.284 | 0.235 |
| Qwen2.5-7B-Instruct | oracle | sim-label | longer | 258 | 0.488 | 0.047 | 0.039 | 0.426 |
| Qwen2.5-7B-Instruct | oracle | sim-label | shorter | 256 | 0.387 | 0.039 | 0.531 | 0.043 |
| Qwen2.5-7B-Instruct | oracle | turn-anon | all | 514 | 0.936 | 0.004 | 0.019 | 0.041 |
| Qwen2.5-7B-Instruct | oracle | turn-anon | longer | 258 | 0.922 | 0.004 | 0.004 | 0.070 |
| Qwen2.5-7B-Instruct | oracle | turn-anon | shorter | 256 | 0.949 | 0.004 | 0.035 | 0.012 |
| Qwen2.5-7B-Instruct | oracle | turn-label | all | 514 | 0.932 | 0.004 | 0.021 | 0.043 |
| Qwen2.5-7B-Instruct | oracle | turn-label | longer | 258 | 0.919 | 0.004 | 0.004 | 0.074 |
| Qwen2.5-7B-Instruct | oracle | turn-label | shorter | 256 | 0.945 | 0.004 | 0.039 | 0.012 |

One canvas, same slots:

| model | lengths | protocol | dir | n_slots | own | sibling_fit | sibling_other | other |
|---|---|---|---|---|---|---|---|---|
| Dream canvas k=1 | oracle | one canvas | all | 514 | 0.963 | 0.000 | 0.000 | 0.037 |
| Dream canvas k=1 | oracle | one canvas | longer | 258 | 0.961 | 0.000 | 0.000 | 0.039 |
| Dream canvas k=1 | oracle | one canvas | shorter | 256 | 0.965 | 0.000 | 0.000 | 0.035 |
| Dream canvas k=1 swap | swap | one canvas | all | 514 | 0.054 | 0.794 | 0.000 | 0.152 |
| Dream canvas k=1 swap | swap | one canvas | longer | 258 | 0.058 | 0.791 | 0.000 | 0.151 |
| Dream canvas k=1 swap | swap | one canvas | shorter | 256 | 0.051 | 0.797 | 0.000 | 0.152 |
| Qwen canvas (AR) | oracle | one canvas | all | 514 | 0.938 | 0.002 | 0.010 | 0.051 |
| Qwen canvas (AR) | oracle | one canvas | longer | 258 | 0.934 | 0.000 | 0.000 | 0.066 |
| Qwen canvas (AR) | oracle | one canvas | shorter | 256 | 0.941 | 0.004 | 0.020 | 0.035 |
