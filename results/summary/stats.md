# Uncertainty for the main comparisons (scripts/stats.py)

| section | model | metric | A | B | n | A % | B % | A - B | 95% CI | A only / B only | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | Dream | acc | exact k=1 | exact k=16 | 400 | 89.8 | 84.5 | 5.3 | [3.3, 7.5] | 21 / 0 | <1e-4 |
| 4 | Dream | ccer | exact k=1 | exact k=16 | 400 | 1.3 | 2.0 | -0.8 | [-2.0, 0.3] | 1 / 4 | 0.3750 |
| 4 | Dream | acc | exact k=1 | exact tau=0.9 | 400 | 89.8 | 89.0 | 0.8 | [-0.5, 2.0] | 5 / 2 | 0.4531 |
| 6 | Dream | acc | exact k=1 | +1 k=1 | 400 | 89.8 | 15.5 | 74.3 | [69.8, 78.5] | 298 / 1 | <1e-4 |
| 6 | Dream | acc | +1 k=1 | +1 k=16 | 400 | 15.5 | 5.0 | 10.5 | [7.5, 13.8] | 45 / 3 | <1e-4 |
| 7 | Dream | ccer | estimate k=1 | exact k=16 | 400 | 7.5 | 2.0 | 5.5 | [3.0, 8.5] | 27 / 5 | 0.0001 |
| 7 | Dream | ccer | +1 k=1 | exact k=16 | 400 | 5.8 | 2.0 | 3.8 | [1.5, 6.3] | 19 / 4 | 0.0026 |
| 7 | Dream | ccer | +8 k=1 | exact k=16 | 400 | 1.8 | 2.0 | -0.3 | [-1.8, 1.3] | 4 / 5 | 1.0000 |
| 7 | Dream | ccer | swap k=1 | exact k=16 | 165 | 16.4 | 1.2 | 15.2 | [9.7, 21.2] | 26 / 1 | <1e-4 |
| 8 | Dream | acc | exact k=1 | estimate k=1 | 400 | 89.8 | 42.3 | 47.5 | [42.3, 52.8] | 198 / 8 | <1e-4 |
| 4 | LLaDA2.0 | acc | exact k=1 | exact k=16 | 400 | 87.8 | 79.8 | 8.0 | [5.3, 10.8] | 33 / 1 | <1e-4 |
| 4 | LLaDA2.0 | ccer | exact k=1 | exact k=16 | 400 | 2.5 | 4.5 | -2.0 | [-3.5, -0.5] | 1 / 9 | 0.0215 |
| 6 | LLaDA2.0 | acc | exact k=1 | +1 k=1 | 100 | 92.0 | 16.0 | 76.0 | [67.0, 84.0] | 77 / 1 | <1e-4 |
| 6 | LLaDA2.0 | acc | +1 k=1 | +1 k=16 | 100 | 16.0 | 4.0 | 12.0 | [5.0, 19.0] | 13 / 1 | 0.0018 |
| 7 | LLaDA2.0 | ccer | estimate k=1 | exact k=16 | 100 | 14.0 | 3.0 | 11.0 | [4.0, 19.0] | 14 / 3 | 0.0127 |
| 7 | LLaDA2.0 | ccer | +1 k=1 | exact k=16 | 100 | 12.0 | 3.0 | 9.0 | [2.0, 16.0] | 11 / 2 | 0.0225 |
| 7 | LLaDA2.0 | ccer | +8 k=1 | exact k=16 | 100 | 36.0 | 3.0 | 33.0 | [24.0, 42.0] | 33 / 0 | <1e-4 |
| 7 | LLaDA2.0 | ccer | swap k=1 | exact k=16 | 34 | 44.1 | 8.8 | 35.3 | [14.7, 55.9] | 15 / 3 | 0.0075 |
| 8 | LLaDA2.0 | acc | exact k=1 | estimate k=1 | 100 | 92.0 | 39.0 | 53.0 | [43.0, 63.0] | 53 / 0 | <1e-4 |
| 8 | Dream | acc | +8 k=4 beta=2 | +8 k=4 beta=0 | 400 | 72.3 | 57.0 | 15.3 | [10.3, 20.3] | 87 / 26 | <1e-4 |
| 8 | Dream | acc | +2 k=4 beta=8 | +2 k=4 beta=0 | 400 | 13.8 | 2.8 | 11.0 | [7.8, 14.5] | 48 / 4 | <1e-4 |
| 5 | Dream | dup (list) | canvas k=16 | canvas k=1 | 60 | 15.0 | 0.0 | 15.0 | [6.7, 25.0] | 9 / 0 | 0.0039 |
| 5 | Dream | dup (open) | canvas k=16 | canvas k=1 | 45 | 100.0 | 0.0 | 100.0 | [100.0, 100.0] | 45 / 0 | <1e-4 |
| 5 | Dream | dup (list) | team same time, numbered | canvas k=16 | 60 | 78.3 | 15.0 | 63.3 | [51.7, 75.0] | 38 / 0 | <1e-4 |
| 5 | Dream | dup (list) | team same time, anonymous | team same time, numbered | 60 | 100.0 | 78.3 | 21.7 | [11.7, 31.7] | 13 / 0 | 0.0002 |
| 5 | Qwen | dup (list) | team same time, numbered | team turns, numbered | 60 | 100.0 | 0.0 | 100.0 | [100.0, 100.0] | 60 / 0 | <1e-4 |
| 5 | Dream | dup (list) | team same time, numbered | team turns, numbered | 60 | 78.3 | 0.0 | 78.3 | [68.3, 88.3] | 47 / 0 | <1e-4 |
| 4C | Dream | acc | canvas k=16 | team same time, numbered | 104 | 81.7 | 28.8 | 52.9 | [42.3, 62.5] | 56 / 1 | <1e-4 |
| 4C | Dream | acc | team same time, rule | team same time, numbered | 104 | 48.1 | 28.8 | 19.2 | [10.6, 27.9] | 22 / 2 | <1e-4 |
| 4C | Dream | acc | canvas k=16 | team same time, rule | 104 | 81.7 | 48.1 | 33.7 | [24.0, 43.3] | 36 / 1 | <1e-4 |
| 4C | Dream | acc | canvas k=1 | team turns, numbered | 104 | 88.5 | 85.6 | 2.9 | [-1.0, 7.7] | 4 / 1 | 0.3750 |
| 4C | Qwen | acc | team same time, rule | team same time, numbered | 104 | 34.6 | 0.0 | 34.6 | [26.0, 44.2] | 36 / 0 | <1e-4 |
| 6C | Dream | acc | team same time, anonymous, exact (C2) | Qwen team same time, anonymous, exact (C2) | 165 | 48.5 | 0.6 | 47.9 | [40.0, 55.8] | 79 / 0 | <1e-4 |

| model | condition | metric | n | rate % | 95% Wilson interval |
|---|---|---|---|---|---|
| LLaDA2.0 | exact k=1 | acc | 100 | 92.0 | [85.0, 95.9] |
| LLaDA2.0 | +1 k=1 | acc | 100 | 16.0 | [10.1, 24.4] |
| LLaDA2.0 | estimate k=1 | acc | 100 | 39.0 | [30.0, 48.8] |
| LLaDA2.0 | +8 k=1 | acc | 100 | 38.0 | [29.1, 47.8] |
| LLaDA2.0 | swap k=1 | ccer | 34 | 44.1 | [28.9, 60.5] |
| LLaDA2.0 | estimate k=1 | ccer | 100 | 14.0 | [8.5, 22.1] |
