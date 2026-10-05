# Uncertainty for the main comparisons (scripts/stats.py)

| section | model | metric | A | B | n | A % | B % | A - B | 95% CI | A only / B only | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | Dream | acc | exact k=1 | exact k=16 | 400 | 89.8 | 84.5 | 5.3 | [3.3, 7.5] | 21 / 0 | <1e-4 | <1e-4 |
| 4 | Dream | ccer | exact k=1 | exact k=16 | 400 | 1.0 | 2.0 | -1.0 | [-2.0, -0.3] | 0 / 4 | 0.1250 | 1.0000 |
| 4 | Dream | acc | exact k=1 | exact tau=0.9 | 400 | 89.8 | 89.0 | 0.8 | [-0.5, 2.0] | 5 / 2 | 0.4531 | 1.0000 |
| 6 | Dream | acc | exact k=1 | +1 k=1 | 400 | 89.8 | 15.5 | 74.3 | [69.8, 78.5] | 298 / 1 | <1e-4 | <1e-4 |
| 6 | Dream | acc | +1 k=1 | +1 k=16 | 400 | 15.5 | 5.0 | 10.5 | [7.5, 13.8] | 45 / 3 | <1e-4 | <1e-4 |
| 7 | Dream | ccer | estimate k=1 | exact k=16 | 400 | 6.5 | 2.0 | 4.5 | [2.0, 7.2] | 23 / 5 | 0.0009 | 0.0182 |
| 7 | Dream | ccer | +1 k=1 | exact k=16 | 400 | 2.8 | 2.0 | 0.8 | [-1.0, 2.5] | 8 / 5 | 0.5811 | 1.0000 |
| 7 | Dream | ccer | +8 k=1 | exact k=16 | 400 | 1.8 | 2.0 | -0.3 | [-1.8, 1.3] | 4 / 5 | 1.0000 | 1.0000 |
| 7 | Dream | ccer | swap k=1 | exact k=16 | 165 | 15.2 | 1.2 | 13.9 | [8.5, 20.0] | 24 / 1 | <1e-4 | <1e-4 |
| 8 | Dream | acc | exact k=1 | estimate k=1 | 400 | 89.8 | 42.3 | 47.5 | [42.3, 52.8] | 198 / 8 | <1e-4 | <1e-4 |
| 4 | LLaDA2.0 | acc | exact k=1 | exact k=16 | 400 | 87.8 | 79.8 | 8.0 | [5.3, 10.8] | 33 / 1 | <1e-4 | <1e-4 |
| 4 | LLaDA2.0 | ccer | exact k=1 | exact k=16 | 400 | 2.5 | 3.8 | -1.3 | [-2.8, 0.3] | 2 / 7 | 0.1797 | 1.0000 |
| 6 | LLaDA2.0 | acc | exact k=1 | +1 k=1 | 100 | 92.0 | 16.0 | 76.0 | [67.0, 84.0] | 77 / 1 | <1e-4 | <1e-4 |
| 6 | LLaDA2.0 | acc | +1 k=1 | +1 k=16 | 100 | 16.0 | 4.0 | 12.0 | [5.0, 19.0] | 13 / 1 | 0.0018 | 0.0330 |
| 7 | LLaDA2.0 | ccer | estimate k=1 | exact k=16 | 100 | 12.0 | 3.0 | 9.0 | [2.0, 16.0] | 12 / 3 | 0.0352 | 0.4922 |
| 7 | LLaDA2.0 | ccer | +1 k=1 | exact k=16 | 100 | 7.0 | 3.0 | 4.0 | [-1.0, 10.0] | 6 / 2 | 0.2891 | 1.0000 |
| 7 | LLaDA2.0 | ccer | +8 k=1 | exact k=16 | 100 | 35.0 | 3.0 | 32.0 | [23.0, 41.0] | 32 / 0 | <1e-4 | <1e-4 |
| 7 | LLaDA2.0 | ccer | swap k=1 | exact k=16 | 34 | 29.4 | 8.8 | 20.6 | [0.0, 38.2] | 10 / 3 | 0.0923 | 1.0000 |
| 8 | LLaDA2.0 | acc | exact k=1 | estimate k=1 | 100 | 92.0 | 39.0 | 53.0 | [43.0, 63.0] | 53 / 0 | <1e-4 | <1e-4 |
| 8 | Dream | acc | +8 k=4 beta=2 | +8 k=4 beta=0 | 400 | 72.3 | 57.0 | 15.3 | [10.3, 20.3] | 87 / 26 | <1e-4 | <1e-4 |
| 8 | Dream | acc | +2 k=4 beta=8 | +2 k=4 beta=0 | 400 | 13.8 | 2.8 | 11.0 | [7.8, 14.5] | 48 / 4 | <1e-4 | <1e-4 |
| 5 | Dream | dup (list) | canvas k=16 | canvas k=1 | 60 | 15.0 | 0.0 | 15.0 | [6.7, 25.0] | 9 / 0 | 0.0039 | 0.0586 |
| 5 | Dream | dup (open) | canvas k=16 | canvas k=1 | 45 | 100.0 | 0.0 | 100.0 | [100.0, 100.0] | 45 / 0 | <1e-4 | <1e-4 |
| 5 | Dream | dup (list) | team same time, numbered | canvas k=16 | 60 | 78.3 | 15.0 | 63.3 | [51.7, 75.0] | 38 / 0 | <1e-4 | <1e-4 |
| 5 | Dream | dup (list) | team same time, anonymous | team same time, numbered | 60 | 100.0 | 78.3 | 21.7 | [11.7, 31.7] | 13 / 0 | 0.0002 | 0.0051 |
| 5 | Qwen | dup (list) | team same time, numbered | team turns, numbered | 60 | 100.0 | 0.0 | 100.0 | [100.0, 100.0] | 60 / 0 | <1e-4 | <1e-4 |
| 5 | Dream | dup (list) | team same time, numbered | team turns, numbered | 60 | 78.3 | 0.0 | 78.3 | [68.3, 88.3] | 47 / 0 | <1e-4 | <1e-4 |
| 4C | Dream | acc | canvas k=16 | team same time, numbered | 104 | 81.7 | 28.8 | 52.9 | [42.3, 62.5] | 56 / 1 | <1e-4 | <1e-4 |
| 4C | Dream | acc | team same time, rule | team same time, numbered | 104 | 48.1 | 28.8 | 19.2 | [10.6, 27.9] | 22 / 2 | <1e-4 | 0.0008 |
| 4C | Dream | acc | canvas k=16 | team same time, rule | 104 | 81.7 | 48.1 | 33.7 | [24.0, 43.3] | 36 / 1 | <1e-4 | <1e-4 |
| 4C | Dream | acc | canvas k=1 | team turns, numbered | 104 | 88.5 | 85.6 | 2.9 | [-1.0, 7.7] | 4 / 1 | 0.3750 | 1.0000 |
| 4C | Dream | acc | canvas k=1 | team turns, anonymous | 104 | 88.5 | 84.6 | 3.8 | [1.0, 7.7] | 4 / 0 | 0.1250 | 1.0000 |
| 4C | Dream | acc | canvas k=16, one tool | team same time, numbered, one tool | 86 | 81.4 | 32.6 | 48.8 | [38.4, 59.3] | 43 / 1 | <1e-4 | <1e-4 |
| 4C | Dream | acc | canvas k=16, one tool | team same time, rule, one tool | 86 | 81.4 | 55.8 | 25.6 | [16.3, 34.9] | 23 / 1 | <1e-4 | <1e-4 |
| 4C | Qwen | acc | canvas (AR) | team turns, anonymous | 104 | 91.3 | 88.5 | 2.9 | [0.0, 6.7] | 3 / 0 | 0.2500 | 1.0000 |
| 4C | Qwen | acc | canvas (AR) | team turns, numbered | 104 | 91.3 | 88.5 | 2.9 | [0.0, 6.7] | 3 / 0 | 0.2500 | 1.0000 |
| 4C | Qwen | acc | team same time, rule | team same time, numbered | 104 | 34.6 | 0.0 | 34.6 | [26.0, 44.2] | 36 / 0 | <1e-4 | <1e-4 |
| 6C | Dream | acc | team same time, anonymous, exact (C2) | Qwen team same time, anonymous, exact (C2) | 165 | 48.5 | 0.6 | 47.9 | [40.0, 55.8] | 79 / 0 | <1e-4 | <1e-4 |
| 4D | Dream | acc | team position | team same time, numbered | 104 | 85.6 | 28.8 | 56.7 | [47.1, 66.3] | 59 / 0 | <1e-4 | <1e-4 |
| 4D | Dream | acc | team position | canvas k=16 | 104 | 85.6 | 81.7 | 3.8 | [1.0, 7.7] | 4 / 0 | 0.1250 | 1.0000 |
| 4D | Dream | acc | canvas k=1 | team position | 104 | 88.5 | 85.6 | 2.9 | [0.0, 6.7] | 3 / 0 | 0.2500 | 1.0000 |
| 4D | Qwen | acc | team position | team same time, rule | 104 | 54.8 | 34.6 | 20.2 | [8.7, 31.7] | 31 / 10 | 0.0015 | 0.0276 |
| 4D | Qwen | acc | canvas (AR) | team position | 104 | 91.3 | 54.8 | 36.5 | [27.9, 46.2] | 38 / 0 | <1e-4 | <1e-4 |
| 6D | Dream | acc | tolerant +1 k=1 | original +1 k=1 | 400 | 28.3 | 15.5 | 12.8 | [8.5, 17.0] | 66 / 15 | <1e-4 | <1e-4 |
| 6D | Dream | acc | tolerant +2 k=1 | original +2 k=1 | 400 | 20.0 | 3.3 | 16.8 | [13.0, 20.5] | 69 / 2 | <1e-4 | <1e-4 |
| 6D | Dream | acc | tolerant estimate k=1 | original estimate k=1 | 400 | 47.5 | 42.3 | 5.3 | [2.0, 8.5] | 32 / 11 | 0.0019 | 0.0330 |
| 7 | Dream | ccer | estimate k=16 | estimate k=1 | 400 | 10.3 | 6.5 | 3.8 | [1.5, 6.0] | 19 / 4 | 0.0026 | 0.0416 |
| 7 | Dream | acc | estimate k=1 | estimate k=16 | 400 | 42.3 | 32.3 | 10.0 | [6.3, 13.8] | 52 / 12 | <1e-4 | <1e-4 |

Robustness (not in the Holm family):

| model | metric | A | B | n | A % | B % | A - B | 95% CI | A only / B only | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| Dream | ccer without Shared | estimate k=1 | exact k=16 | 400 | 4.5 | 1.3 | 3.3 | [1.0, 5.5] | 17 / 4 | 0.0072 |
| Dream | ccer without Shared | estimate k=16 | estimate k=1 | 400 | 6.5 | 4.5 | 2.0 | [0.5, 3.8] | 10 / 2 | 0.0386 |

| model | metric | interaction | n | points | 95% CI | positive / negative requests | sign test p |
|---|---|---|---|---|---|---|---|
| Dream | ccer | (estimate k=16 - estimate k=1) - (exact k=16 - exact k=1) | 400 | 2.8 | [0.3, 5.3] | 19 / 8 | 0.0522 |
| Dream | ccer without Shared | (estimate k=16 - estimate k=1) - (exact k=16 - exact k=1) | 400 | 1.3 | [-0.5, 3.3] | 10 / 5 | 0.3018 |

| model | condition | metric | n | rate % | 95% Wilson interval |
|---|---|---|---|---|---|
| LLaDA2.0 | exact k=1 | acc | 100 | 92.0 | [85.0, 95.9] |
| LLaDA2.0 | +1 k=1 | acc | 100 | 16.0 | [10.1, 24.4] |
| LLaDA2.0 | estimate k=1 | acc | 100 | 39.0 | [30.0, 48.8] |
| LLaDA2.0 | +8 k=1 | acc | 100 | 38.0 | [29.1, 47.8] |
| LLaDA2.0 | swap k=1 | ccer | 34 | 29.4 | [16.8, 46.2] |
| LLaDA2.0 | estimate k=1 | ccer | 100 | 12.0 | [7.0, 19.8] |
