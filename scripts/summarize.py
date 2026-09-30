"""Aggregate result JSONL files into the tables behind Figures 2-4.

  python scripts/summarize.py results/*.jsonl --by category
  python scripts/summarize.py results/probe_*.jsonl --by meta.n --by meta.entities --csv out.csv

Columns: n, BFCL accuracy, set-level accuracy, syntax rate, cross-call error rate
(CCER), single-call error rate (SCER), mean NFE, and the rate of every label.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.eval.taxonomy import ALL_LABELS, is_cross_call, is_single_call  # noqa: E402


def get(rec, dotted):
    cur = rec
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def summarize(records, by):
    groups = defaultdict(list)
    for r in records:
        if "diagnosis" not in r:
            continue
        key = (r.get("model_id", r.get("model")), r["mode"], r["cfg_tag"]) + tuple(get(r, b) for b in by)
        groups[key].append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(k) for k in kv[0])):
        n = len(rs)
        labels = [r["diagnosis"]["labels"] for r in rs]
        row = {"model": key[0], "mode": key[1], "cfg": key[2]}
        row.update({b: v for b, v in zip(by, key[3:])})
        row.update({
            "n": n,
            "bfcl_acc": sum(r["diagnosis"]["bfcl_valid"] for r in rs) / n,
            "set_acc": sum(r["diagnosis"]["correct"] for r in rs) / n,
            "syntax": sum(r["syntax_ok"] for r in rs) / n,
            "ccer": sum(is_cross_call(lab) for lab in labels) / n,
            "scer": sum(is_single_call(lab) for lab in labels) / n,
            "nfe": sum(r.get("nfe", 0) for r in rs) / n,
        })
        for lab in ALL_LABELS:
            row[lab] = sum(lab in ls for ls in labels) / n
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", action="append", default=[], help="record field, e.g. category or meta.n")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    records = []
    for path in args.files:
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        crashed = [r for r in recs if "error" in r]
        if crashed:
            print(f"WARNING {path}: {len(crashed)}/{len(recs)} runs raised an exception, "
                  f"e.g. {crashed[0]['error'][:200]}")
        records += recs
    rows = summarize(records, args.by)
    if not rows:
        print("no evaluated records")
        return

    main_cols = ["model", "mode", "cfg"] + args.by + ["n", "bfcl_acc", "set_acc", "syntax", "ccer", "scer", "nfe"]
    print("| " + " | ".join(main_cols) + " |")
    print("|" + "---|" * len(main_cols))
    for r in rows:
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in main_cols]
        print("| " + " | ".join(cells) + " |")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
