"""What the wrong actions would do: the share of turns whose calls, if executed, would

  wrong_entity   act on another call's entity: a slot holding a sibling's reference value
                 (the same argument of another call to the same tool)
  magnitude      pass a number off by a factor of 10 or more (500000 -> 5000000)
  duplicate      issue the same call twice where the reference does not
  unparsed       not be executable at all (the output does not parse)
  any_wrong      be wrong in any way (the BFCL verdict)

per model x slot-length condition x decoding config. Each predicted call is compared with
the reference call that the diagnosis's optimal matching gives it, so two calls that trade
all their values (a reordering that the set comparison accepts and that executes the same
actions) do not count as acting on the wrong entity. Wrong-entity and magnitude are
counted over parsed turns but reported as shares of all turns. No tokenizer needed.

  python scripts/consequences.py results/dream/bfcl_skel_k.jsonl results/dream/bfcl_surplus.jsonl \
      results/dream/bfcl_swap.jsonl results/dream/bfcl_estimate.jsonl
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import sibling_groups  # noqa: E402
from ptcdiag.eval.taxonomy import value_ok  # noqa: E402

KEYS = ["wrong_entity", "magnitude", "duplicate", "unparsed", "any_wrong"]


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def turn(ex, rec):
    out = {k: False for k in KEYS}
    out["any_wrong"] = not rec["diagnosis"]["correct"]
    out["duplicate"] = "duplicate_call" in rec["diagnosis"]["labels"]
    if not rec.get("syntax_ok"):
        out["unparsed"] = True
        return out
    calls = rec.get("calls") or []
    sibs = sibling_groups(ex)
    for pi, gi in rec["diagnosis"].get("matching", []):
        fname, params = ex.gold_calls[gi]
        if pi >= len(calls) or calls[pi].get("name") != fname:
            continue
        args = calls[pi].get("arguments", {})
        f = ex.function(fname)
        for p, acc in params.items():
            if p not in args or value_ok(f, p, args[p], acc):
                continue
            v = args[p]
            if any(value_ok(f, p, v, ex.gold_calls[gj][1][p]) for gj in sibs.get((fname, p), []) if gj != gi):
                out["wrong_entity"] = True
            g = [_num(a) for a in acc if _num(a) not in (None, 0)]
            if _num(v) is not None and g:
                ratio = abs(v) / abs(g[0]) if v else 0.0
                if ratio >= 10 or (ratio and ratio <= 0.1):
                    out["magnitude"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--csv")
    args = ap.parse_args()

    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    groups = defaultdict(list)
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                if r.get("mode") != "skeleton" or "diagnosis" not in r or r["id"] not in exs \
                        or float(r.get("end_bias") or 0):
                    continue
                cond = (r.get("model_id") or r["model"]).split("/")[-1], r.get("length_mode") or "oracle", \
                    r.get("surplus") or 0, r["cfg_tag"]
                groups[cond].append(turn(exs[r["id"]], r))
    cols = ["model", "length_mode", "surplus", "cfg", "n"] + KEYS
    rows = []
    for (m, mode, s, tag), ts in sorted(groups.items()):
        row = {"model": m, "length_mode": mode, "surplus": s, "cfg": tag, "n": len(ts)}
        row.update({k: sum(t[k] for t in ts) / len(ts) for k in KEYS})
        rows.append(row)
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
