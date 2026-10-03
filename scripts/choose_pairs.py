"""Choose-N probes, pair by pair: do two calls' city slots collide when they are committed in the same step?

Each call of a choose-N skeleton has one masked position, its one-token city slot, so the
masked positions of a record are the calls' slots in call order, and their commit steps
say which slots moved together. Over every pair of calls of every syntax-ok record, per
model x decoding config x variant: pairs committed in the same step and how many of them
hold the same city, and the same for pairs committed in different steps (a later slot that
copies an earlier one).

  python scripts/choose_pairs.py results/dream/choose.jsonl results/llada2/choose.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_jsonl  # noqa: E402
from ptcdiag.eval.bfcl_checker import standardize_string  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="data/choose.jsonl")
    args = ap.parse_args()

    exs = {e.id: e for e in load_jsonl(args.data)}
    agg = defaultdict(lambda: [0, 0, 0, 0])  # same-step pairs, collisions; other pairs, collisions
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                if r["id"] not in exs or "trace" not in r or not r.get("syntax_ok"):
                    continue
                steps = [s for s in r["trace"]["commit_step"] if s != -1]
                calls = r.get("calls") or []
                if len(calls) != len(steps):
                    continue
                vals = [standardize_string(str(c.get("arguments", {}).get("city", ""))) for c in calls]
                key = (r.get("model_id", r["model"]).split("/")[-1], r["cfg_tag"], exs[r["id"]].meta["variant"])
                for a, b in combinations(range(len(vals)), 2):
                    i = 0 if steps[a] == steps[b] else 2
                    agg[key][i] += 1
                    agg[key][i + 1] += vals[a] == vals[b]
    print("| model | cfg | variant | same-step pairs | same city | rate | other pairs | same city | rate |")
    print("|---|---|---|---|---|---|---|---|---|")

    def rate(hit, n):
        return f"{hit / n:.3f}" if n else "--"

    for (m, tag, v), (s, sc, o, oc) in sorted(agg.items()):
        print(f"| {m} | {tag} | {v} | {s} | {sc} | {rate(sc, s)} | {o} | {oc} | {rate(oc, o)} |")


if __name__ == "__main__":
    main()
