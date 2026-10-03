"""Cross-call error labels: one-token-at-a-time decoding with wrong slot lengths vs the most parallel decoding with exact ones.

The worry about parallel decoding is that same-step commits produce cross-call
coordination errors (duplicated calls, cross-bound arguments, chimera values, shared
arguments that disagree). This table asks how often the same labels appear when the
decoding is sequential (confidence order, k=1) but the slot lengths are wrong (surplus,
length swap, one-forward estimate), next to the most parallel run with exact slot
lengths (k=16, surplus 0). Each pair is computed on the items the mismatch run has
(LLaDA2.0 ran the length conditions on a 100-item subset, the swap on 34 items), so the
two rows of a pair share their denominator. Labels come from each record's diagnosis
(ptcdiag/eval/taxonomy.py); ccer is the share of items with any cross-call label, as
in the other tables. No tokenizer needed.

  python scripts/masquerade.py results/dream/bfcl_skel_k.jsonl results/dream/bfcl_surplus.jsonl \
      results/dream/bfcl_swap.jsonl results/dream/bfcl_estimate.jsonl
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.eval.taxonomy import is_cross_call  # noqa: E402

LABELS = ["duplicate_call", "cross_binding", "chimera_value", "inconsistent_shared_arg"]
PARALLEL = ("oracle", 0, 0.0, "k16")


def key(r):
    k = r["cfg_tag"].split("_")[1]
    return (r.get("length_mode") or "oracle", r.get("surplus") or 0, float(r.get("end_bias") or 0), k)


def describe(c):
    mode, s, _, k = c
    length = {"oracle": f"surplus +{s}" if s else "exact", "swap": "swap",
              "length_estimate": "estimate"}[mode]
    return f"{length}, k={k[1:]}"


def row(model, cond, recs, ids):
    n = len(ids)
    out = {"model": model, "condition": describe(cond), "n": n}
    for lab in LABELS:
        out[lab] = sum(lab in recs[i]["diagnosis"]["labels"] for i in ids) / n
    out["ccer"] = sum(is_cross_call(recs[i]["diagnosis"]["labels"]) for i in ids) / n
    out["set_acc"] = sum(recs[i]["diagnosis"]["correct"] for i in ids) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()

    runs = defaultdict(lambda: defaultdict(dict))  # model -> condition -> id -> record
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                # confidence-ordered, no threshold: the k sweep's own schedule
                if r.get("mode") != "skeleton" or "diagnosis" not in r \
                        or not r["cfg_tag"].startswith("confidence_") or "_tnone_" not in r["cfg_tag"]:
                    continue
                runs[r.get("model_id") or r["model"]][key(r)][r["id"]] = r

    rows = []
    for model, conds in sorted(runs.items()):
        if PARALLEL not in conds:
            print(f"(no exact-length k=16 run for {model})", file=sys.stderr)
            continue
        mismatched = sorted(c for c in conds if c[3] == "k1" and c[2] == 0.0
                            and (c[0] != "oracle" or c[1] > 0))
        last = None
        for c in mismatched:
            ids = sorted(i for i in conds[c] if i in conds[PARALLEL])
            if ids != last:  # the reference row, once per item set
                rows.append(row(model, PARALLEL, conds[PARALLEL], ids))
                last = ids
            rows.append(row(model, c, conds[c], ids))

    cols = ["model", "condition", "n"] + LABELS + ["ccer", "set_acc"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
