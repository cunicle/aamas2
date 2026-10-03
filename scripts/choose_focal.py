"""Experiment B, list variant: which listed city do the agents take? Per model x protocol x
temperature, the share of agents (over all agents of all teams) that take the city listed
first, and the share of teams in which every agent takes the same city. Agents that act at
the same time and share a focal point (the first listed city) collide on it.

  tar xzf release/agents_2026-10-03.tar.gz; tar xzf release/agents_b2_2026-10-03.tar.gz
  python scripts/choose_focal.py results/qwen/agents.jsonl results/qwen/agents_sample.jsonl \
      results/dream/agents.jsonl results/dream/agents_sample.jsonl
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    args = ap.parse_args()
    agg = defaultdict(lambda: [0, 0, 0, 0])  # agents, first listed; teams, all same
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                if r["variant"] != "list":
                    continue
                key = (r["model_id"].split("/")[-1], r["protocol"], r["temperature"])
                a = agg[key]
                a[0] += len(r["cities"])
                a[1] += sum(c == r["listed"][0] for c in r["cities"])
                a[2] += 1
                a[3] += len(set(r["cities"])) == 1 and None not in r["cities"]
    print("| model | protocol | temperature | agents | first listed city | teams | all agents same city |")
    print("|---|---|---|---|---|---|---|")
    for (m, p, t), (n, first, teams, same) in sorted(agg.items()):
        print(f"| {m} | {p} | {t} | {n} | {first} ({first / n:.3f}) | {teams} | {same} ({same / teams:.3f}) |")


if __name__ == "__main__":
    main()
