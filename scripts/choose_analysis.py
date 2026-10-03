"""Choose-N probes: did the N calls pick N different allowed cities?

Per model x decoding config x variant (list / open), over the items of data/choose.jsonl:
  ok          every call has an allowed city and all N differ
  duplicate   two calls with the same city
  same_step   a duplicate whose two cities were committed in the same decoding step
              (dLLM records: the city slots are the only masked positions, one token each)
  invalid     a city outside the allowed set (or no parse)
  in_order    list variant: the calls take the first N listed cities, in listed order

In the open variant the one-token slots cut multi-token cities ("New", "Los") or make the
models abbreviate ("LA", "NY"), so `invalid` there measures the slot, not the model; only
the duplicate rates of that variant are informative.

  python scripts/choose_analysis.py results/dream/choose.jsonl results/llada2/choose.jsonl \
      results/qwen/choose.jsonl --csv results/summary/choose.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_jsonl  # noqa: E402
from ptcdiag.eval.bfcl_checker import standardize_string  # noqa: E402


def item_stats(ex, rec):
    n = ex.meta["n"]
    allowed = {standardize_string(c) for c in ex.ground_truth[0]["get_weather"]["city"]}
    calls = rec.get("calls") or []
    vals = [c.get("arguments", {}).get("city") for c in calls] if rec.get("syntax_ok") else []
    vals = [standardize_string(v) if isinstance(v, str) else None for v in vals]
    valid = len(vals) == n and all(v in allowed for v in vals)
    dup = len(vals) == n and len(set(vals)) < n
    same_step = False
    for st in (rec.get("trace") or {}).get("steps", []):
        toks = st["tokens"]
        same_step |= len(toks) != len(set(toks))
    listed = [standardize_string(c) for c in ex.meta.get("listed", [])]
    return {"ok": valid and not dup, "duplicate": dup, "same_step": dup and same_step,
            "invalid": not valid, "in_order": bool(listed) and vals == listed[:n]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="data/choose.jsonl")
    ap.add_argument("--csv")
    args = ap.parse_args()

    exs = {e.id: e for e in load_jsonl(args.data)}
    groups = defaultdict(list)
    for path in args.results:
        if not os.path.exists(path):
            print(f"(missing {path})")
            continue
        with open(path) as f:
            for r in map(json.loads, f):
                if r["id"] in exs and "diagnosis" in r:
                    m = r.get("model_id", r.get("model")).split("/")[-1]
                    groups[(m, r["cfg_tag"], exs[r["id"]].meta["variant"])].append(r)

    keys = ["ok", "duplicate", "same_step", "invalid", "in_order"]
    rows = []
    for (m, tag, variant), rs in sorted(groups.items()):
        st = [item_stats(exs[r["id"]], r) for r in rs]
        row = {"model": m, "cfg": tag, "variant": variant, "n": len(rs)}
        row.update({k: sum(s[k] for s in st) / len(st) for k in keys})
        rows.append(row)
    cols = ["model", "cfg", "variant", "n"] + keys
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
