"""What Dream would write into surplus masks if the type mask let it (Section 6), from the
teacher-forced probe records (results/dream/length_prior_s{1,2}.jsonl; the same probe as
Figure 3c). Each record holds one slot with its value written and s masks left after it, and
`top_unconstrained`, the most probable token at the first surplus position over the whole
vocabulary.

A token is outside the slot's type mask if the slot's class does not accept it
(`ptcdiag.decoding.constraints._classify`) and it is not a closing token (a quote for a
string; a comma or brace for any other value), since closing is what the probe measures
separately.

  python scripts/typemask_probe.py --results results > results/summary/typemask_probe.md
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.decoding.constraints import _classify  # noqa: E402


def closing(rec):
    t = rec["top_unconstrained"]
    return t.startswith('"') if rec["class"] == "string" else t.strip()[:1] in (",", "}")


def outside(rec):
    return rec["class"] not in _classify(rec["top_unconstrained"]) and not closing(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    load = lambda s: [json.loads(line) for line in open(f"{args.results}/dream/length_prior_s{s}.jsonl")]

    s1 = load(1)
    out = [r for r in s1 if outside(r)]
    print("Dream, teacher-forced probe; the most probable token at the first surplus position.\n")
    print(f"One surplus mask: outside the type mask (not closing) in {len(out)} of {len(s1)} slots "
          f"({100 * len(out) / len(s1):.1f}%); closing in {sum(map(closing, s1))}.\n")
    print("| class | slots | outside the type mask | most frequent such tokens |")
    print("|---|---|---|---|")
    for c, n in sorted(Counter(r["class"] for r in s1).items()):
        o = [r["top_unconstrained"] for r in out if r["class"] == c]
        top = ", ".join(f"{t!r} {k}" for t, k in Counter(o).most_common(4))
        print(f"| {c} | {n} | {len(o)} | {top} |")

    s2 = load(2)
    ints = [r for r in s2 if r["class"] == "integer"]
    point = sum(r["top_unconstrained"] == "." for r in ints)
    print(f"\nTwo surplus masks: {point} of {len(ints)} integer slots would write a decimal point.")


if __name__ == "__main__":
    main()
