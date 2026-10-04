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
| LLaDA2.0 | 33 | 7 | 7 | 19 | 1 |

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
