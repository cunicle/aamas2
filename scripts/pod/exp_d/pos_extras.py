"""Experiment D report helper (not part of the paper's analysis): position agents whose own call
holds the placeholder ("..." for a string, null otherwise), per model, data and agent index, and the
choose-N duplicate rate with and without the placeholder counted as a city.

  python pos_extras.py <results dir>
"""
import json
import os
import sys
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(rdir):
    print("| model | data | teams | agents | own call holds a placeholder | ... by agent index (1, 2, 3, 4+) |")
    print("|---|---|---|---|---|---|")
    for m in ("qwen", "dream"):
        for name in ("agents_d", "agents_pos"):
            p = os.path.join(rdir, m, name + ".jsonl")
            if not os.path.exists(p):
                print(f"| {m} | {name} | TBD |"); continue
            rs = [r for r in read(p) if "error" not in r]
            n_ag = ph = 0
            by_i = defaultdict(lambda: [0, 0])
            for r in rs:
                for a in r["agents"]:
                    n_ag += 1
                    if name == "agents_pos":
                        hit = a["city"] == "..."
                    else:
                        c = a["call"]
                        hit = c is not None and any(v is None or v == "..." for v in c["arguments"].values())
                    ph += hit
                    k = min(a["i"], 4)
                    by_i[k][0] += hit
                    by_i[k][1] += 1
            cells = ", ".join(f"{by_i[k][0]}/{by_i[k][1]}" for k in sorted(by_i))
            print(f"| {m} | {name} | {len(rs)} | {n_ag} | {ph} ({100 * ph / max(n_ag, 1):.1f}%) | {cells} |")
    print("\nchoose-N teams: duplicate city among the agents, counting '...' as a city (as exp_d_quick.py) "
          "and without it; teams where some agent wrote '...'.\n")
    print("| model | variant | teams | duplicate (with '...') | duplicate (real cities only) | some agent wrote '...' |")
    print("|---|---|---|---|---|---|")
    for m in ("qwen", "dream"):
        p = os.path.join(rdir, m, "agents_pos.jsonl")
        if not os.path.exists(p):
            continue
        by = defaultdict(list)
        for r in read(p):
            if "error" not in r:
                by[r["variant"]].append(r)
        for var, rs in sorted(by.items()):
            d1 = d2 = ph = 0
            for r in rs:
                c = [x for x in r["cities"] if x is not None]
                real = [x for x in c if x != "..."]
                d1 += len(c) != len(set(c))
                d2 += len(real) != len(set(real))
                ph += len(real) < len(c)
            n = len(rs)
            print(f"| {m} | {var} | {n} | {100 * d1 / n:.1f}% | {100 * d2 / n:.1f}% | {100 * ph / n:.1f}% |")


if __name__ == "__main__":
    main(*sys.argv[1:])
