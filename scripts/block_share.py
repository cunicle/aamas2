"""How often sibling slots of different calls share a decoding block (LLaDA2.0, 32-token blocks).

A block-causal model decodes one block at a time, with causal attention across blocks: two
slots in different blocks are never committed in the same step, and the later one sees the
earlier one. Only slots that share a block can be committed together. This counts, over the
exact-length skeleton runs, the pairs of sibling slots (the same parameter in two calls of
the same function) that have at least one position in a common block, and the same for every
pair of calls' slots in the choose-N probes (one one-token city slot per call).

The slot positions come from the run records themselves: the masked positions of the
skeleton are the generation positions with a commit step (fixed skeleton tokens have -1),
and the skeleton lays the slots out in gold call order, then gold parameter order, each
slot a contiguous run (build_skeleton). Blocks are aligned to absolute canvas positions
(LLaDA2.0's absolute_blocks), i.e. block = (gen_start + i) // block_length.

  python scripts/block_share.py results/llada2/bfcl_skel_k.jsonl results/llada2/choose.jsonl \
      --choose-data data/choose.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import sibling_groups  # noqa: E402


def slot_runs(rec):
    """Contiguous runs of masked generation positions, as lists of absolute positions."""
    tr = rec["trace"]
    runs, cur = [], []
    for i, s in enumerate(tr["commit_step"]):
        if s != -1:
            cur.append(tr["gen_start"] + i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def slot_keys(ex):
    return [(ci, p) for ci, (f, params) in enumerate(ex.gold_calls)
            for p, acc in params.items() if any(a != "" for a in acc)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--choose-data", default="data/choose.jsonl")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--cfg", default="confidence_k16_tnone_b32_T0.0")
    args = ap.parse_args()

    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    if os.path.exists(args.choose_data):
        exs.update({e.id: e for e in load_examples("probe:" + args.choose_data)})
    pairs = defaultdict(lambda: [0, 0])  # source -> [same block, all]
    skipped = 0
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                if r.get("cfg_tag") != args.cfg or r["id"] not in exs or r.get("surplus") \
                        or (r.get("length_mode") or "oracle") != "oracle" or "trace" not in r:
                    continue
                ex = exs[r["id"]]
                runs, keys = slot_runs(r), slot_keys(ex)
                if len(runs) != len(keys):
                    skipped += 1
                    continue
                where = {k: {p // args.block for p in run} for k, run in zip(keys, runs)}
                src = "choose" if r["id"].startswith("choose_") else ex.category
                for (fname, p), calls in sibling_groups(ex).items():
                    for a, b in combinations(calls, 2):
                        pairs[src][0] += bool(where[a, p] & where[b, p])
                        pairs[src][1] += 1
    print(f"| source | sibling slot pairs | sharing a block | share |")
    print("|---|---|---|---|")
    for src, (same, n) in sorted(pairs.items()):
        print(f"| {src} | {n} | {same} | {same / n:.3f} |")
    if skipped:
        print(f"\n({skipped} records skipped: slot runs did not match the skeleton's slots)")


if __name__ == "__main__":
    main()
