"""Numbers the text quotes that no other script reports (CPU, raw records).

  swap_sets     Dream, swapped lengths, k=1: correct requests whose calls match the reference
                only as a reordered set (the optimal matching is not the identity)
  ar_cutoff     Qwen2.5 in the skeleton with exact lengths vs eight surplus masks: requests whose
                output differs, and set-accuracy transitions (exact-length slots can cut off a
                value that the AR model tokenizes differently in context)
  k_failures    requests that fail at k=16 but not at k=1 (exact lengths), by kind: output
                that does not parse, a cross-call error, single-call errors only
  estimate      wrong one-forward length estimates that are too long
  uniform       choose-N list: probability that n agents choosing uniformly and independently
                among the n+3 listed cities collide, per n and averaged over the 60 items

  python scripts/review_numbers.py --results results --choose data/choose.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter
from fractions import Fraction
from math import perm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from paper_tables import DREAM_K, LLADA_K, estimate_slots, fmt, runs_by  # noqa: E402
from ptcdiag.data import load_jsonl  # noqa: E402
from ptcdiag.eval.taxonomy import is_cross_call  # noqa: E402


def pct(a, b):
    return f"{a}/{b} ({fmt(Fraction(100 * a, b))}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--choose", default="data/choose.jsonl")
    args = ap.parse_args()
    res = args.results
    out = ["# Numbers quoted in the text (scripts/review_numbers.py)", ""]

    d = runs_by([f"{res}/dream/{f}.jsonl" for f in ("bfcl_skel_k", "bfcl_swap")])
    sw = d["swap", 0, 0.0, DREAM_K[1]]
    ok = [r for r in sw if r["diagnosis"]["correct"]]
    reordered = [r for r in ok if any(p != g for p, g in r["diagnosis"]["matching"])]
    out += ["## swap_sets (Dream, swapped lengths, k=1)",
            f"- correct requests: {pct(len(ok), len(sw))}",
            f"- of these, correct only as a reordered set: {pct(len(reordered), len(ok))}", ""]

    q = runs_by([f"{res}/qwen/{f}.jsonl" for f in ("bfcl_skeleton", "bfcl_surplus")])
    q0 = {r["id"]: r for r in q["oracle", 0, 0.0, "ar_greedy"]}
    q8 = {r["id"]: r for r in q["oracle", 8, 0.0, "ar_greedy"]}
    ids = sorted(set(q0) & set(q8))
    differ = sum(q0[i]["text"] != q8[i]["text"] for i in ids)
    fixed = sum(not q0[i]["diagnosis"]["correct"] and q8[i]["diagnosis"]["correct"] for i in ids)
    broken = sum(q0[i]["diagnosis"]["correct"] and not q8[i]["diagnosis"]["correct"] for i in ids)
    out += ["## ar_cutoff (Qwen2.5, exact vs +8)",
            f"- requests: {len(ids)}; output differs: {pct(differ, len(ids))}",
            f"- wrong with exact lengths, correct with +8: {fixed}; the reverse: {broken}", ""]

    out += ["## k_failures (exact lengths; fail at k=16 but not at k=1)",
            "| model | newly failing | does not parse | cross-call | single-call only | newly correct |",
            "|---|---|---|---|---|---|"]
    for name, model, tags in [("Dream", "dream", DREAM_K), ("LLaDA2.0", "llada2", LLADA_K)]:
        runs = runs_by([f"{res}/{model}/bfcl_skel_k.jsonl"])
        a = {r["id"]: r for r in runs["oracle", 0, 0.0, tags[1]]}
        b = {r["id"]: r for r in runs["oracle", 0, 0.0, tags[16]]}
        new = [b[i] for i in a if a[i]["diagnosis"]["correct"] and not b[i]["diagnosis"]["correct"]]
        back = sum(not a[i]["diagnosis"]["correct"] and b[i]["diagnosis"]["correct"] for i in a)
        syn = sum(not r["syntax_ok"] for r in new)
        cc = sum(r["syntax_ok"] and is_cross_call(r["diagnosis"]["labels"]) for r in new)
        out.append(f"| {name} | {len(new)} | {syn} | {cc} | {len(new) - syn - cc} | {back} |")
    out.append("")

    out += ["## estimate (one-forward length estimates)"]
    for name, model in [("Dream", "dream"), ("LLaDA2.0", "llada2")]:
        diffs, _, _ = estimate_slots(f"{res}/{model}/length_estimate.jsonl")
        wrong = [x for x in diffs if x]
        out.append(f"- {name}: wrong estimates {pct(len(wrong), len(diffs))}; too long among wrong: "
                   f"{pct(sum(x > 0 for x in wrong), len(wrong))}")
    out.append("")

    exs = [e for e in load_jsonl(args.choose) if e.meta["variant"] == "list"]
    ns = Counter(e.meta["n"] for e in exs)
    out += ["## uniform (choose-N list, independent uniform choices among n+3 cities)",
            "| n | items | P(collision) |", "|---|---|---|"]
    total = Fraction(0)
    for n in sorted(ns):
        m = n + 3
        p = 1 - Fraction(perm(m, n), m ** n)
        total += p * ns[n]
        out.append(f"| {n} | {ns[n]} | {fmt(100 * p)}% |")
    out.append(f"| all | {sum(ns.values())} | {fmt(100 * total / sum(ns.values()))}% |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
