# Experiment F: quick read-out

Simultaneous agents that get their role in the prompt, as a number (sim-label) or as a number and the stated rule (sim-rule), each in its original wording (experiments B and C) and in three others (ptcdiag/data/agents.py, WORDINGS). Reference rows: position agents (experiment D) and numbered agents that take turns (experiments B and C).

## BFCL: the team-symmetric requests (oracle slot lengths, greedy)

set acc: the team's n calls are correct as a set, with a 95% Wilson interval. dup: a duplicated call. in order: agent i makes reference call i, every i. first: agents of a sibling group that make the group's first reference call. Differences are in points of set accuracy, with a 95% paired-bootstrap interval and the exact McNemar p.

### Dream

| role cue | wording | teams | set acc % | dup % | in order % | first % | unparsed % | minus original | minus position agents |
|---|---|---|---|---|---|---|---|---|---|
| number | original | 104 | 28.8 [21.0, 38.2] | 64.4 | 28.8 | 67.8 | 1.0 | - | -56.7 [-66.3, -47.1], p=<1e-4 |
| number | para | 104 | 45.2 [36.0, 54.8] | 47.1 | 44.2 | 59.0 | 1.0 | 16.3 [9.6, 24.0], p=<1e-4 | -40.4 [-50.0, -30.8], p=<1e-4, Holm p=<1e-4 |
| number | ordinal | 104 | 39.4 [30.6, 49.0] | 53.8 | 39.4 | 62.3 | 1.0 | 10.6 [4.8, 17.3], p=0.0034 | -46.2 [-55.8, -36.5], p=<1e-4, Holm p=<1e-4 |
| number | first | 104 | 5.8 [2.7, 12.0] | 91.3 | 5.8 | 78.0 | 1.0 | -23.1 [-31.7, -15.4], p=<1e-4 | -79.8 [-87.5, -72.1], p=<1e-4, Holm p=<1e-4 |
| rule | original | 104 | 48.1 [38.7, 57.6] | 44.2 | 47.1 | 57.9 | 1.0 | - | -37.5 [-47.1, -28.8], p=<1e-4 |
| rule | para | 104 | 55.8 [46.2, 64.9] | 37.5 | 54.8 | 53.5 | 1.0 | 7.7 [1.9, 14.4], p=0.0386 | -29.8 [-39.4, -20.2], p=<1e-4, Holm p=<1e-4 |
| rule | direct | 104 | 67.3 [57.8, 75.6] | 23.1 | 65.4 | 46.9 | 1.0 | 19.2 [11.5, 26.9], p=<1e-4 | -18.3 [-26.0, -10.6], p=<1e-4, Holm p=<1e-4 |
| rule | first | 104 | 33.7 [25.3, 43.2] | 60.6 | 33.7 | 63.4 | 1.0 | -14.4 [-22.1, -6.7], p=0.0015 | -51.9 [-61.5, -42.3], p=<1e-4, Holm p=<1e-4 |
| position | (reference) | 104 | 85.6 [77.6, 91.1] | 1.9 | 81.7 | 35.9 | 1.0 | - | - |
| turns, numbered | (reference) | 104 | 85.6 [77.6, 91.1] | 1.0 | 74.0 | 36.6 | 1.9 | - | - |

Dream, number: set accuracy over the four wordings from 5.8% (first) to 45.2% (para).
Dream, rule: set accuracy over the four wordings from 33.7% (first) to 67.3% (direct).

### Qwen2.5

| role cue | wording | teams | set acc % | dup % | in order % | first % | unparsed % | minus original | minus position agents |
|---|---|---|---|---|---|---|---|---|---|
| number | original | 104 | 0.0 [0.0, 3.6] | 98.1 | 0.0 | 88.3 | 1.9 | - | -54.8 [-64.4, -45.2], p=<1e-4 |
| number | para | 104 | 45.2 [36.0, 54.8] | 48.1 | 44.2 | 57.5 | 1.0 | 45.2 [35.6, 54.8], p=<1e-4 | -9.6 [-21.2, 1.9], p=0.1539, Holm p=0.3643 |
| number | ordinal | 104 | 4.8 [2.1, 10.8] | 93.3 | 4.8 | 81.7 | 1.9 | 4.8 [1.0, 9.6], p=0.0625 | -50.0 [-59.6, -40.4], p=<1e-4, Holm p=<1e-4 |
| number | first | 104 | 0.0 [0.0, 3.6] | 99.0 | 0.0 | 90.8 | 1.0 | 0.0 [0.0, 0.0], p=1.0000 | -54.8 [-64.4, -45.2], p=<1e-4, Holm p=<1e-4 |
| rule | original | 104 | 34.6 [26.2, 44.2] | 59.6 | 33.7 | 67.8 | 1.9 | - | -20.2 [-31.7, -8.7], p=0.0015 |
| rule | para | 104 | 56.7 [47.1, 65.8] | 33.7 | 55.8 | 51.6 | 1.9 | 22.1 [13.5, 31.7], p=<1e-4 | 1.9 [-9.6, 13.5], p=0.8679, Holm p=0.8679 |
| rule | direct | 104 | 64.4 [54.9, 73.0] | 26.0 | 62.5 | 48.0 | 1.0 | 29.8 [20.2, 39.4], p=<1e-4 | 9.6 [-1.0, 20.2], p=0.1214, Holm p=0.3643 |
| rule | first | 104 | 9.6 [5.3, 16.8] | 85.6 | 8.7 | 81.0 | 1.9 | -25.0 [-33.7, -16.3], p=<1e-4 | -45.2 [-54.8, -35.6], p=<1e-4, Holm p=<1e-4 |
| position | (reference) | 104 | 54.8 [45.2, 64.0] | 24.0 | 53.8 | 45.1 | 1.0 | - | - |
| turns, numbered | (reference) | 104 | 88.5 [80.9, 93.3] | 0.0 | 82.7 | 37.4 | 1.9 | - | - |

Qwen2.5, number: set accuracy over the four wordings from 0.0% (first) to 45.2% (para).
Qwen2.5, rule: set accuracy over the four wordings from 9.6% (first) to 64.4% (direct).

Holm's correction is over the 12 tests of a new wording against the same model's position agents.

## Choose-N: the 60 list requests (greedy)

ok: all cities allowed and different. dup: two agents take one city, with a 95% Wilson interval. all same: every agent takes one city. in order: agent i takes the i-th listed city. first listed: share of agents that take the city listed first. The difference is in points of dup.

### Dream

| role cue | wording | teams | ok % | dup % | all same % | in order % | first listed % | dup minus original |
|---|---|---|---|---|---|---|---|---|
| number | original | 60 | 21.7 | 78.3 [66.4, 86.9] | 40.0 | 3.3 | 35.0 | - |
| number | para | 60 | 30.0 | 70.0 [57.5, 80.1] | 31.7 | 3.3 | 37.8 | -8.3 [-16.7, -1.7], p=0.0625 |
| number | ordinal | 60 | 21.7 | 78.3 [66.4, 86.9] | 41.7 | 3.3 | 38.9 | 0.0 [-6.7, 6.7], p=1.0000 |
| number | first | 60 | 8.3 | 91.7 [81.9, 96.4] | 75.0 | 1.7 | 52.8 | 13.3 [5.0, 21.7], p=0.0078 |
| rule | original | 60 | 55.0 | 45.0 [33.1, 57.5] | 13.3 | 10.0 | 38.9 | - |
| rule | para | 60 | 50.0 | 50.0 [37.7, 62.3] | 20.0 | 5.0 | 31.7 | 5.0 [-6.7, 16.7], p=0.5811 |
| rule | direct | 60 | 61.7 | 38.3 [27.1, 51.0] | 11.7 | 11.7 | 36.1 | -6.7 [-15.0, 0.0], p=0.2188 |
| rule | first | 60 | 30.0 | 70.0 [57.5, 80.1] | 41.7 | 5.0 | 46.1 | 25.0 [13.3, 36.7], p=0.0003 |
| position | (reference) | 60 | 85.0 | 15.0 [8.1, 26.1] | 5.0 | 11.7 | 30.0 | - |
| turns, numbered | (reference) | 60 | 100.0 | 0.0 [0.0, 6.0] | 0.0 | 26.7 | 23.9 | - |

### Qwen2.5

| role cue | wording | teams | ok % | dup % | all same % | in order % | first listed % | dup minus original |
|---|---|---|---|---|---|---|---|---|
| number | original | 60 | 0.0 | 100.0 [94.0, 100.0] | 100.0 | 0.0 | 100.0 | - |
| number | para | 60 | 88.3 | 11.7 [5.8, 22.2] | 0.0 | 23.3 | 34.4 | -88.3 [-95.0, -80.0], p=<1e-4 |
| number | ordinal | 60 | 0.0 | 100.0 [94.0, 100.0] | 93.3 | 0.0 | 96.7 | 0.0 [0.0, 0.0], p=1.0000 |
| number | first | 60 | 0.0 | 100.0 [94.0, 100.0] | 100.0 | 0.0 | 100.0 | 0.0 [0.0, 0.0], p=1.0000 |
| rule | original | 60 | 30.0 | 70.0 [57.5, 80.1] | 33.3 | 8.3 | 55.0 | - |
| rule | para | 60 | 93.3 | 6.7 [2.6, 15.9] | 0.0 | 40.0 | 33.3 | -63.3 [-76.7, -50.0], p=<1e-4 |
| rule | direct | 60 | 86.7 | 13.3 [6.9, 24.2] | 0.0 | 28.3 | 37.2 | -56.7 [-70.0, -43.3], p=<1e-4 |
| rule | first | 60 | 1.7 | 98.3 [91.1, 99.7] | 68.3 | 0.0 | 87.8 | 28.3 [16.7, 40.0], p=<1e-4 |
| position | (reference) | 60 | 0.0 | 65.0 [52.4, 75.8] | 0.0 | 0.0 | 33.3 | - |
| turns, numbered | (reference) | 60 | 100.0 | 0.0 [0.0, 6.0] | 0.0 | 91.7 | 33.3 | - |

