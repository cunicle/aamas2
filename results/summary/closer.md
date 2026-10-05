# Experiment C3: closer in the slot vs closer in the skeleton (Dream, one canvas)

Original: the skeleton writes each value's closer after its slot, the model ends a value early with its own closer and padding. Variant: no closer in the skeleton, one more position per slot, the model writes the closer and the rest of the slot becomes spaces. exact / +s: slot = value length + s (+1 for the variant's closer); swap / onesided: on the 165 items they change. Greedy, confidence order. Every row compares the two interfaces on the items both have.

## Item level

| lengths | n_original | n_variant | set_acc_original | set_acc_variant | ccer_original | ccer_variant | syntax_original | syntax_variant | overfill_original | overfill_variant |
|---|---|---|---|---|---|---|---|---|---|---|
| exact k=1 | 400 | 400 | 0.897 | 0.703 | 0.010 | 0.005 | 0.990 | 0.767 | 0.005 | 0.000 |
| +1 k=1 | 400 | 400 | 0.155 | 0.268 | 0.028 | 0.003 | 0.823 | 0.438 | 0.547 | 0.125 |
| +2 k=1 | 400 | 400 | 0.033 | 0.168 | 0.037 | 0.007 | 0.752 | 0.330 | 0.627 | 0.125 |
| +4 k=1 | 400 | 400 | 0.170 | 0.235 | 0.037 | 0.003 | 0.807 | 0.365 | 0.510 | 0.090 |
| +8 k=1 | 400 | 400 | 0.600 | 0.395 | 0.018 | 0.003 | 0.895 | 0.430 | 0.180 | 0.007 |
| exact k=16 | 400 | 400 | 0.845 | 0.502 | 0.020 | 0.003 | 0.948 | 0.550 | 0.013 | 0.000 |
| +1 k=16 | 400 | 400 | 0.050 | 0.100 | 0.070 | 0.010 | 0.748 | 0.212 | 0.603 | 0.092 |
| swap k=1 | 165 | 165 | 0.673 | 0.358 | 0.152 | 0.055 | 0.988 | 0.467 | 0.030 | 0.030 |
| onesided k=1 | 165 | 165 | 0.176 | 0.248 | 0.424 | 0.097 | 0.988 | 0.539 | 0.224 | 0.061 |

## Slot level (all slots; scripts/slot_errors.py classes)

| lengths | interface | n_items | n_slots | own | sibling_fit | sibling | overfill | truncated | other | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| exact k=1 | original | 400 | 3063 | 0.954 | 0.008 | 0.000 | 0.002 | 0.000 | 0.027 | 0.009 |
| exact k=1 | variant | 400 | 3063 | 0.753 | 0.005 | 0.000 | 0.000 | 0.000 | 0.023 | 0.219 |
| +1 k=1 | original | 400 | 3063 | 0.340 | 0.002 | 0.003 | 0.351 | 0.004 | 0.117 | 0.184 |
| +1 k=1 | variant | 400 | 3063 | 0.378 | 0.000 | 0.002 | 0.062 | 0.001 | 0.020 | 0.537 |
| +2 k=1 | original | 400 | 3063 | 0.120 | 0.001 | 0.000 | 0.440 | 0.001 | 0.182 | 0.255 |
| +2 k=1 | variant | 400 | 3063 | 0.241 | 0.000 | 0.004 | 0.071 | 0.000 | 0.027 | 0.657 |
| +4 k=1 | original | 400 | 3063 | 0.296 | 0.000 | 0.001 | 0.342 | 0.001 | 0.154 | 0.205 |
| +4 k=1 | variant | 400 | 3063 | 0.283 | 0.000 | 0.003 | 0.050 | 0.000 | 0.027 | 0.637 |
| +8 k=1 | original | 400 | 3063 | 0.736 | 0.000 | 0.006 | 0.104 | 0.000 | 0.054 | 0.099 |
| +8 k=1 | variant | 400 | 3063 | 0.443 | 0.000 | 0.008 | 0.003 | 0.000 | 0.008 | 0.537 |
| exact k=16 | original | 400 | 3063 | 0.920 | 0.007 | 0.000 | 0.002 | 0.000 | 0.021 | 0.050 |
| exact k=16 | variant | 400 | 3063 | 0.545 | 0.002 | 0.000 | 0.000 | 0.001 | 0.016 | 0.436 |
| +1 k=16 | original | 400 | 3063 | 0.211 | 0.002 | 0.001 | 0.382 | 0.004 | 0.129 | 0.272 |
| +1 k=16 | variant | 400 | 3063 | 0.165 | 0.000 | 0.000 | 0.052 | 0.000 | 0.018 | 0.765 |
| swap k=1 | original | 165 | 1380 | 0.486 | 0.440 | 0.000 | 0.010 | 0.009 | 0.045 | 0.011 |
| swap k=1 | variant | 165 | 1380 | 0.229 | 0.228 | 0.000 | 0.004 | 0.002 | 0.009 | 0.527 |
| onesided k=1 | original | 165 | 1380 | 0.812 | 0.084 | 0.001 | 0.036 | 0.002 | 0.054 | 0.011 |
| onesided k=1 | variant | 165 | 1380 | 0.477 | 0.028 | 0.002 | 0.009 | 0.001 | 0.040 | 0.443 |

## Swapped slots (the slots the swap changes, 165 items; the paper's Table 3 for the original)

| lengths | interface | slots | n_slots | own | sibling_fit | sibling | overfill | truncated | other | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| exact k=1 | original | all | 514 | 0.963 | 0.000 | 0.000 | 0.002 | 0.000 | 0.035 | 0.000 |
| exact k=1 | original | longer | 258 | 0.961 | 0.000 | 0.000 | 0.000 | 0.000 | 0.039 | 0.000 |
| exact k=1 | original | shorter | 256 | 0.965 | 0.000 | 0.000 | 0.004 | 0.000 | 0.031 | 0.000 |
| exact k=1 | variant | all | 514 | 0.732 | 0.000 | 0.000 | 0.000 | 0.000 | 0.033 | 0.235 |
| exact k=1 | variant | longer | 258 | 0.733 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | 0.236 |
| exact k=1 | variant | shorter | 256 | 0.730 | 0.000 | 0.000 | 0.000 | 0.000 | 0.035 | 0.234 |
| swap k=1 | original | all | 514 | 0.054 | 0.794 | 0.000 | 0.023 | 0.023 | 0.097 | 0.008 |
| swap k=1 | original | longer | 258 | 0.058 | 0.791 | 0.000 | 0.047 | 0.000 | 0.097 | 0.008 |
| swap k=1 | original | shorter | 256 | 0.051 | 0.797 | 0.000 | 0.000 | 0.047 | 0.098 | 0.008 |
| swap k=1 | variant | all | 514 | 0.016 | 0.411 | 0.000 | 0.012 | 0.006 | 0.021 | 0.535 |
| swap k=1 | variant | longer | 258 | 0.016 | 0.411 | 0.000 | 0.023 | 0.000 | 0.019 | 0.531 |
| swap k=1 | variant | shorter | 256 | 0.016 | 0.410 | 0.000 | 0.000 | 0.012 | 0.023 | 0.539 |

## One-sided lengthening (165 items, k=1): the lengthened slot (i*, p)

own: its own value, ended early; overfill: its own value and more; sibling_fit: the value of j* (the longest sibling, whose slot stays exact), i.e. two calls with the same value; other = other_sibling + truncated + unparsed + wrong. jstar_own: j*'s slot holds j*'s own value.

| interface | n_items | n_slots | own | overfill | sibling_fit | other | other_sibling | truncated | unparsed | wrong | jstar_own |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 165 | 243 | 0.218 | 0.206 | 0.354 | 0.222 | 0.016 | 0.008 | 0.012 | 0.185 | 0.909 |
| variant | 165 | 243 | 0.267 | 0.049 | 0.103 | 0.580 | 0.004 | 0.000 | 0.465 | 0.111 | 0.473 |

## Teacher-forced probe: the first position after the gold value

Original surplus s: s masks, then the skeleton's closer. Variant surplus s: 1 + s positions, then the next fixed text (s = 0: the one position must be the closer). P(closer): tokens that start with the slot's closer; pick: what the skeleton constraint would commit there.

| surplus | interface | class | n | mean_p_close | median_p_pad | mean_p_content | pick_closer | pick_pad | pick_content |
|---|---|---|---|---|---|---|---|---|---|
| 0 | variant | all | 3063 | 0.605 | 0.000 | 0.395 | 0.900 | 0.000 | 0.100 |
| 1 | original | all | 3063 | 0.010 | 0.000 | 0.990 | 0.024 | 0.000 | 0.976 |
| 1 | variant | all | 3063 | 0.430 | 0.000 | 0.570 | 0.628 | 0.000 | 0.372 |
| 2 | original | all | 3063 | 0.033 | 0.000 | 0.967 | 0.174 | 0.000 | 0.826 |
| 2 | variant | all | 3063 | 0.345 | 0.000 | 0.655 | 0.612 | 0.000 | 0.388 |
| 4 | original | all | 3063 | 0.254 | 0.000 | 0.746 | 0.514 | 0.000 | 0.486 |
| 4 | variant | all | 3063 | 0.529 | 0.000 | 0.471 | 0.781 | 0.000 | 0.219 |
| 8 | original | all | 3063 | 0.740 | 0.000 | 0.260 | 0.895 | 0.000 | 0.105 |
| 8 | variant | all | 3063 | 0.816 | 0.000 | 0.184 | 0.935 | 0.000 | 0.065 |

By slot class:

| surplus | interface | class | n | mean_p_close | median_p_pad | mean_p_content | pick_closer | pick_pad | pick_content |
|---|---|---|---|---|---|---|---|---|---|
| 0 | variant | boolean | 103 | 0.654 | 0.000 | 0.346 | 1.000 | 0.000 | 0.000 |
| 1 | original | boolean | 103 | 0.053 | 0.000 | 0.947 | 0.039 | 0.000 | 0.961 |
| 1 | variant | boolean | 103 | 0.428 | 0.000 | 0.572 | 0.981 | 0.000 | 0.019 |
| 2 | original | boolean | 103 | 0.159 | 0.000 | 0.841 | 1.000 | 0.000 | 0.000 |
| 2 | variant | boolean | 103 | 0.195 | 0.000 | 0.805 | 0.951 | 0.000 | 0.049 |
| 4 | original | boolean | 103 | 0.292 | 0.000 | 0.708 | 0.932 | 0.000 | 0.068 |
| 4 | variant | boolean | 103 | 0.483 | 0.000 | 0.517 | 0.981 | 0.000 | 0.019 |
| 8 | original | boolean | 103 | 0.873 | 0.000 | 0.127 | 1.000 | 0.000 | 0.000 |
| 8 | variant | boolean | 103 | 0.868 | 0.000 | 0.132 | 0.990 | 0.000 | 0.010 |
| 0 | variant | float | 281 | 0.784 | 0.000 | 0.216 | 0.904 | 0.000 | 0.096 |
| 1 | original | float | 281 | 0.002 | 0.000 | 0.998 | 0.000 | 0.000 | 1.000 |
| 1 | variant | float | 281 | 0.145 | 0.000 | 0.855 | 0.310 | 0.000 | 0.690 |
| 2 | original | float | 281 | 0.025 | 0.000 | 0.975 | 0.046 | 0.000 | 0.954 |
| 2 | variant | float | 281 | 0.071 | 0.000 | 0.929 | 0.181 | 0.000 | 0.819 |
| 4 | original | float | 281 | 0.098 | 0.000 | 0.902 | 0.114 | 0.000 | 0.886 |
| 4 | variant | float | 281 | 0.311 | 0.000 | 0.689 | 0.480 | 0.000 | 0.520 |
| 8 | original | float | 281 | 0.548 | 0.000 | 0.452 | 0.808 | 0.000 | 0.192 |
| 8 | variant | float | 281 | 0.722 | 0.000 | 0.277 | 0.925 | 0.000 | 0.075 |
| 0 | variant | generic | 201 | 0.416 | 0.000 | 0.584 | 0.458 | 0.000 | 0.542 |
| 1 | original | generic | 201 | 0.035 | 0.000 | 0.965 | 0.045 | 0.000 | 0.955 |
| 1 | variant | generic | 201 | 0.057 | 0.000 | 0.942 | 0.025 | 0.000 | 0.975 |
| 2 | original | generic | 201 | 0.013 | 0.000 | 0.986 | 0.000 | 0.000 | 1.000 |
| 2 | variant | generic | 201 | 0.027 | 0.000 | 0.973 | 0.000 | 0.000 | 1.000 |
| 4 | original | generic | 201 | 0.045 | 0.000 | 0.955 | 0.010 | 0.000 | 0.990 |
| 4 | variant | generic | 201 | 0.100 | 0.000 | 0.900 | 0.119 | 0.000 | 0.881 |
| 8 | original | generic | 201 | 0.433 | 0.000 | 0.567 | 0.577 | 0.000 | 0.423 |
| 8 | variant | generic | 201 | 0.455 | 0.000 | 0.545 | 0.642 | 0.000 | 0.358 |
| 0 | variant | integer | 1014 | 0.808 | 0.000 | 0.192 | 0.921 | 0.000 | 0.079 |
| 1 | original | integer | 1014 | 0.009 | 0.000 | 0.991 | 0.006 | 0.000 | 0.994 |
| 1 | variant | integer | 1014 | 0.193 | 0.000 | 0.807 | 0.424 | 0.000 | 0.576 |
| 2 | original | integer | 1014 | 0.016 | 0.000 | 0.984 | 0.179 | 0.000 | 0.821 |
| 2 | variant | integer | 1014 | 0.079 | 0.000 | 0.921 | 0.390 | 0.000 | 0.610 |
| 4 | original | integer | 1014 | 0.201 | 0.000 | 0.799 | 0.510 | 0.000 | 0.490 |
| 4 | variant | integer | 1014 | 0.420 | 0.000 | 0.580 | 0.750 | 0.000 | 0.250 |
| 8 | original | integer | 1014 | 0.752 | 0.000 | 0.248 | 0.934 | 0.000 | 0.066 |
| 8 | variant | integer | 1014 | 0.823 | 0.000 | 0.177 | 0.967 | 0.000 | 0.033 |
| 0 | variant | string | 1464 | 0.452 | 0.000 | 0.548 | 0.939 | 0.000 | 0.061 |
| 1 | original | string | 1464 | 0.005 | 0.000 | 0.995 | 0.037 | 0.000 | 0.963 |
| 1 | variant | string | 1464 | 0.699 | 0.000 | 0.301 | 0.889 | 0.000 | 0.111 |
| 2 | original | string | 1464 | 0.040 | 0.000 | 0.960 | 0.160 | 0.000 | 0.840 |
| 2 | variant | string | 1464 | 0.636 | 0.000 | 0.364 | 0.910 | 0.000 | 0.090 |
| 4 | original | string | 1464 | 0.347 | 0.000 | 0.653 | 0.634 | 0.000 | 0.366 |
| 4 | variant | string | 1464 | 0.709 | 0.000 | 0.291 | 0.936 | 0.000 | 0.064 |
| 8 | original | string | 1464 | 0.800 | 0.000 | 0.200 | 0.920 | 0.000 | 0.080 |
| 8 | variant | string | 1464 | 0.875 | 0.000 | 0.125 | 0.952 | 0.000 | 0.048 |

## Variant closers

ended: the slot's value has a closer; bare_closer: of those, the closing token is just the closer ('"', ',' or '}'; for arrays / dicts the closer ends its token) rather than e.g. '",' or '}}' (the extra text stays on the canvas, decoding keeps the closer only); no_closer: the slot was filled to the end without one.

| lengths | n_slots | ended | bare_closer | no_closer |
|---|---|---|---|---|
| exact k=1 | 3063 | 0.924 | 0.444 | 0.076 |
| +1 k=1 | 3063 | 0.832 | 0.743 | 0.168 |
| +2 k=1 | 3063 | 0.785 | 0.738 | 0.215 |
| +4 k=1 | 3063 | 0.876 | 0.532 | 0.124 |
| +8 k=1 | 3063 | 0.930 | 0.530 | 0.070 |
| exact k=16 | 3063 | 0.870 | 0.472 | 0.130 |
| +1 k=16 | 3063 | 0.736 | 0.790 | 0.264 |
| swap k=1 | 1380 | 0.804 | 0.497 | 0.196 |
| onesided k=1 | 1380 | 0.838 | 0.480 | 0.162 |

What a bare-closer-only constraint would have done where the variant closed a number / boolean / string slot with a longer token: the committing step's top-5 candidates rank a bare closer above every content token (bare), a content token above every bare closer (content: the restriction would likely have made it write more content instead of closing), or only longer closers (top5_closers).

| lengths | longer_closers | bare | content | top5_closers |
|---|---|---|---|---|
| exact k=1 | 1506 | 0.208 | 0.398 | 0.394 |
| +1 k=1 | 645 | 0.549 | 0.254 | 0.197 |
| +2 k=1 | 620 | 0.594 | 0.311 | 0.095 |
| +4 k=1 | 1240 | 0.549 | 0.435 | 0.015 |
| +8 k=1 | 1327 | 0.659 | 0.336 | 0.005 |
| exact k=16 | 1376 | 0.105 | 0.451 | 0.443 |
| +1 k=16 | 469 | 0.486 | 0.448 | 0.066 |
| swap k=1 | 540 | 0.248 | 0.452 | 0.300 |
| onesided k=1 | 581 | 0.267 | 0.382 | 0.351 |

## Supplementary: slots read one by one

The tables above use the parsed output, as the paper does: in the variant one slot filled without a closer makes the whole output unparseable, and slot_errors.py then counts every slot of that item as unparsed. Here each slot's value is read from its own tokens (the text before its closer, or the whole slot if it has none) in both interfaces and classified the same way. unclosed: variant slots without a closer (they make the output fail to parse); unclosed_overfill: of all slots, the unclosed ones whose content is the own value followed by more.

| lengths | interface | n_slots | own | sibling_fit | sibling | overfill | truncated | other | unclosed | unclosed_overfill |
|---|---|---|---|---|---|---|---|---|---|---|
| exact k=1 | original | 3063 | 0.960 | 0.009 | 0.000 | 0.002 | 0.000 | 0.030 |  |  |
| exact k=1 | variant | 3063 | 0.899 | 0.008 | 0.001 | 0.048 | 0.000 | 0.044 | 0.076 | 0.047 |
| +1 k=1 | original | 3063 | 0.375 | 0.002 | 0.003 | 0.444 | 0.004 | 0.172 |  |  |
| +1 k=1 | variant | 3063 | 0.625 | 0.001 | 0.004 | 0.262 | 0.002 | 0.106 | 0.168 | 0.102 |
| +2 k=1 | original | 3063 | 0.145 | 0.001 | 0.000 | 0.557 | 0.001 | 0.296 |  |  |
| +2 k=1 | variant | 3063 | 0.491 | 0.000 | 0.006 | 0.387 | 0.001 | 0.115 | 0.215 | 0.151 |
| +4 k=1 | original | 3063 | 0.333 | 0.000 | 0.002 | 0.427 | 0.005 | 0.234 |  |  |
| +4 k=1 | variant | 3063 | 0.670 | 0.000 | 0.005 | 0.241 | 0.001 | 0.082 | 0.124 | 0.084 |
| +8 k=1 | original | 3063 | 0.786 | 0.000 | 0.007 | 0.130 | 0.000 | 0.076 |  |  |
| +8 k=1 | variant | 3063 | 0.811 | 0.000 | 0.011 | 0.112 | 0.000 | 0.066 | 0.070 | 0.043 |
| exact k=16 | original | 3063 | 0.951 | 0.008 | 0.000 | 0.002 | 0.001 | 0.038 |  |  |
| exact k=16 | variant | 3063 | 0.849 | 0.006 | 0.001 | 0.082 | 0.001 | 0.060 | 0.130 | 0.082 |
| +1 k=16 | original | 3063 | 0.254 | 0.002 | 0.001 | 0.537 | 0.004 | 0.202 |  |  |
| +1 k=16 | variant | 3063 | 0.472 | 0.001 | 0.001 | 0.392 | 0.006 | 0.128 | 0.264 | 0.172 |
| swap k=1 | original | 1380 | 0.492 | 0.440 | 0.000 | 0.010 | 0.009 | 0.049 |  |  |
| swap k=1 | variant | 1380 | 0.566 | 0.270 | 0.002 | 0.071 | 0.012 | 0.079 | 0.196 | 0.063 |
| onesided k=1 | original | 1380 | 0.815 | 0.084 | 0.001 | 0.036 | 0.002 | 0.061 |  |  |
| onesided k=1 | variant | 1380 | 0.755 | 0.040 | 0.004 | 0.122 | 0.001 | 0.078 | 0.162 | 0.101 |

Swapped slots, read one by one:

| lengths | interface | n_slots | own | sibling_fit | sibling | overfill | truncated | other |
|---|---|---|---|---|---|---|---|---|
| exact k=1 | original | 514 | 0.963 | 0.000 | 0.000 | 0.002 | 0.000 | 0.035 |
| exact k=1 | variant | 514 | 0.881 | 0.000 | 0.004 | 0.066 | 0.000 | 0.049 |
| swap k=1 | original | 514 | 0.054 | 0.794 | 0.000 | 0.023 | 0.025 | 0.103 |
| swap k=1 | variant | 514 | 0.247 | 0.492 | 0.000 | 0.093 | 0.021 | 0.146 |

One-sided lengthening, the lengthened slot read on its own (own / overfill / sibling_fit = j*'s value / other), and j*'s slot holding j*'s value:

| interface | n_slots | own | overfill | sibling_fit | other | jstar_own |
|---|---|---|---|---|---|---|
| original | 243 | 0.218 | 0.202 | 0.354 | 0.226 | 0.909 |
| variant | 243 | 0.403 | 0.239 | 0.140 | 0.218 | 0.741 |

## Canvas samples (variant, k=1)

⟦ ⟧ mark the slots, ␣ a space token inside a slot, ∅ padding. The same item in the original interface below each.

### parallel_1, exact k=1: variant correct, original correct

```text
variant:  [{"name": "calculate_em_force", "arguments": {"b_field":⟦␣5,⟧ "area":⟦␣2,⟧ "d_time":⟦␣4}}⟧}, {"name": "calculate_em_force", "arguments": {"b_field":⟦␣5,⟧ "area":⟦␣2,⟧ "d_time":⟦␣10}}⟧}]
decoded:  [{"name": "calculate_em_force", "arguments": {"b_field": 5, "area": 2, "d_time": 4}}, {"name": "calculate_em_force", "arguments": {"b_field": 5, "area": 2, "d_time": 10}}]
original: [{"name": "calculate_em_force", "arguments": {"b_field":⟦␣5⟧, "area":⟦␣2⟧, "d_time":⟦␣4⟧}}, {"name": "calculate_em_force", "arguments": {"b_field":⟦␣5⟧, "area":⟦␣2⟧, "d_time":⟦␣10⟧}}]
decoded:  [{"name": "calculate_em_force", "arguments": {"b_field": 5, "area": 2, "d_time": 4}}, {"name": "calculate_em_force", "arguments": {"b_field": 5, "area": 2, "d_time": 10}}]
```

### parallel_7, +1 k=1: variant correct, original ['wrong_value']

```text
variant:  [{"name": "math.factorial", "arguments": {"number":⟦␣5}0⟧}, {"name": "math.factorial", "arguments": {"number":⟦␣10}0⟧}, {"name": "math.factorial", "arguments": {"number":⟦␣15}0⟧}]
decoded:  [{"name": "math.factorial", "arguments": {"number": 5}}, {"name": "math.factorial", "arguments": {"number": 10}}, {"name": "math.factorial", "arguments": {"number": 15}}]
original: [{"name": "math.factorial", "arguments": {"number":⟦␣50⟧}}, {"name": "math.factorial", "arguments": {"number":⟦␣100⟧}}, {"name": "math.factorial", "arguments": {"number":⟦␣150⟧}}]
decoded:  [{"name": "math.factorial", "arguments": {"number": 50}}, {"name": "math.factorial", "arguments": {"number": 100}}, {"name": "math.factorial", "arguments": {"number": 150}}]
```

