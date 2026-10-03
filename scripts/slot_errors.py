"""What each skeleton slot was filled with, slot by slot (skeleton calls are in gold order).

Every value slot of a skeleton record is classified against its own call's gold value
(BFCL value semantics) and its siblings' (the same parameter of other calls to the same
function):
  own         the call's own gold value
  sibling     a sibling's gold value; `fit` = the sibling's value has exactly the slot's
              length (what a model binding values to slots by length would write)
  overfill    the own gold value followed by more ("Taylor Swift. Swift", 500000 -> 5000000000)
  truncated   a proper prefix of the own gold value (the slot was too short for it)
  other       anything else
  unparsed    no value (the whole output failed to parse, or the call / argument is missing)

Groups by model, length mode, surplus, end bias and decoding config. With --swap-slots
only the slots the length swap changes count (`swapped_lengths`), on the items that have
a swap record for that model, so the oracle-length rows of the same slots are the
baseline; rows are then also split by whether the swapped slot is longer or shorter than
its own value. Item-level set accuracy and cross-call error rate are over the same items.

  python scripts/slot_errors.py results/dream/bfcl_skel_k.jsonl results/dream/bfcl_swap.jsonl \
      --swap-slots --csv results/summary/swap_dream.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer  # noqa: E402

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import (lengths_from_list, oracle_lengths, sibling_groups,  # noqa: E402
                                          swapped_lengths)
from ptcdiag.eval.taxonomy import _canon, is_cross_call, value_ok  # noqa: E402

KINDS = ["own", "sibling_fit", "sibling", "overfill", "truncated", "other", "unparsed"]


def _flat(v):
    c = _canon(v)
    return c if isinstance(c, str) else json.dumps(c, separators=(",", ":"))


def classify(example, rec, ci, p, slot_len):
    fname, params = example.gold_calls[ci]
    calls = rec.get("calls") or []
    if not rec.get("syntax_ok") or ci >= len(calls) or calls[ci].get("name") != fname \
            or p not in calls[ci].get("arguments", {}):
        return "unparsed"
    v = calls[ci]["arguments"][p]
    f = example.function(fname)
    if value_ok(f, p, v, params[p]):
        return "own"
    sibs = [cj for cj in sibling_groups(example).get((fname, p), []) if cj != ci]
    for cj in sibs:
        if value_ok(f, p, v, example.gold_calls[cj][1][p]):
            return "sibling_fit" if rec["_oracle"].get((cj, p)) == slot_len else "sibling"
    fv = _flat(v)
    for a in params[p]:
        if a == "":
            continue
        fa = _flat(a)
        if fv != fa and fv.startswith(fa):
            return "overfill"
        if fv and fv != fa and fa.startswith(fv):
            return "truncated"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--swap-slots", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()

    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    recs = []
    for path in args.results:
        if not os.path.exists(path):
            print(f"(missing {path})")
            continue
        with open(path) as f:
            recs += [r for r in map(json.loads, f)
                     if r.get("mode") == "skeleton" and "diagnosis" in r and r["id"] in exs]

    def model(r):
        return r.get("model_id", r.get("model"))

    toks, oracle, swapped = {}, {}, {}
    for r in recs:
        m = model(r)
        if m not in toks:
            toks[m] = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        if (m, r["id"]) not in oracle:
            oracle[m, r["id"]] = oracle_lengths(toks[m], exs[r["id"]])
            swapped[m, r["id"]] = swapped_lengths(toks[m], exs[r["id"]])
        r["_oracle"] = oracle[m, r["id"]]
    swap_items = defaultdict(set)
    for r in recs:
        if r.get("length_mode") == "swap":
            swap_items[model(r)].add(r["id"])

    groups = defaultdict(list)
    for r in recs:
        m = model(r)
        if args.swap_slots and r["id"] not in swap_items[m]:
            continue
        groups[(m, r.get("length_mode", "oracle"), r.get("surplus", 0), r.get("end_bias", 0.0) or 0.0,
                r["cfg_tag"])].append(r)

    rows = []
    for (m, lm, s, b, tag), rs in sorted(groups.items()):
        kinds = defaultdict(Counter)  # direction -> kind counts
        for r in rs:
            ex = exs[r["id"]]
            lengths = dict(r["_oracle"])
            lengths.update(lengths_from_list(r["lengths"]) if r.get("lengths") else {})
            for (ci, p), n in r["_oracle"].items():
                sw = swapped[m, r["id"]][ci, p]
                if args.swap_slots and sw == n:
                    continue
                k = classify(ex, r, ci, p, lengths[ci, p] + s)
                kinds["all"][k] += 1
                if args.swap_slots:
                    kinds["longer" if sw > n else "shorter"][k] += 1
        n_items = len(rs)
        for d, cnt in kinds.items():
            tot = sum(cnt.values())
            row = {"model": m.split("/")[-1], "length_mode": lm, "surplus": s, "end_bias": b, "cfg": tag,
                   "slots": d, "n_slots": tot}
            row.update({k: cnt[k] / tot for k in KINDS})
            if d == "all":
                row["n_items"] = n_items
                row["set_acc"] = sum(r["diagnosis"]["correct"] for r in rs) / n_items
                row["ccer"] = sum(is_cross_call(r["diagnosis"]["labels"]) for r in rs) / n_items
            rows.append(row)

    cols = ["model", "length_mode", "surplus", "end_bias", "cfg", "slots", "n_slots"] + KINDS + \
        ["n_items", "set_acc", "ccer"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, ""))
                                for c in cols) + " |")
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
