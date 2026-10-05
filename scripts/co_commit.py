"""How often sibling slots decide together: the share of sibling slot pairs whose first
content tokens are committed in the same denoising step, on the exact-length BFCL skeleton runs.

The first content token of a value (its first token that is not whitespace; a number slot
starts with the space after '":') usually decides which entity it names ("HSBC" vs "Wells"),
so two sibling slots whose first content tokens move in the same step choose their entities
without seeing each other. Slot positions come from the run records as in
scripts/block_share.py.

  python scripts/co_commit.py results/dream/bfcl_skel_k.jsonl results/llada2/bfcl_skel_k.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from block_share import slot_keys, slot_runs  # noqa: E402

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import sibling_groups  # noqa: E402

HUB = {"dream": "Dream-org/Dream-v0-Instruct-7B", "llada2": "inclusionAI/LLaDA2.0-mini"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    toks = {}
    # (model, cfg) -> [pairs with first tokens in the same step, pairs with any shared step,
    #                  all pairs, requests with at least one same-step first-token pair, requests]
    acc = defaultdict(lambda: [0, 0, 0, 0, 0])
    skipped = 0
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                if r["id"] not in exs or r.get("surplus") or (r.get("length_mode") or "oracle") != "oracle" \
                        or "trace" not in r:
                    continue
                ex = exs[r["id"]]
                runs, keys = slot_runs(r), slot_keys(ex)
                if len(runs) != len(keys):
                    skipped += 1
                    continue
                cs, g0 = r["trace"]["commit_step"], r["trace"]["gen_start"]
                if r["model"] not in toks:
                    toks[r["model"]] = AutoTokenizer.from_pretrained(HUB.get(r["model"], r["model"]), trust_remote_code=True)
                tok, gen = toks[r["model"]], r["gen_ids"]
                # each slot's steps, its first content token's step first
                steps = {}
                for k, run in zip(keys, runs):
                    content = [p for p in run if tok.decode([gen[p - g0]]).strip()] or run
                    steps[k] = [cs[content[0] - g0]] + [cs[p - g0] for p in run]
                a = acc[r["model"].split("/")[-1], r["cfg_tag"]]
                any_req = False
                for (fname, p), calls in sibling_groups(ex).items():
                    for x, y in combinations(calls, 2):
                        sx, sy = steps[x, p], steps[y, p]
                        same_first = sx[0] == sy[0]
                        a[0] += same_first
                        a[1] += bool(set(sx) & set(sy))
                        a[2] += 1
                        any_req |= same_first
                a[3] += any_req
                a[4] += 1
    print("| model | decoding | sibling slot pairs | first content tokens in the same step "
          "| any token in the same step | requests with a same-step first-content-token pair |")
    print("|---|---|---|---|---|---|")
    for (model, cfg), (same, anys, n, req, nreq) in sorted(acc.items()):
        if n:
            print(f"| {model} | {cfg} | {n} | {same} ({100 * same / n:.1f}%) | {anys} ({100 * anys / n:.1f}%) "
                  f"| {req} of {nreq} ({100 * req / nreq:.1f}%) |")
    if skipped:
        print(f"\n({skipped} records skipped: slot runs did not match the skeleton's slots)")


if __name__ == "__main__":
    main()
