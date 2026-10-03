"""Tables for the length-prior study: errors vs slot surplus (and k, end bias).

Groups skeleton records by model, length mode (oracle lengths, or lengths estimated by
the model: scripts/length_estimate.py; length-swap runs are left out, see
scripts/slot_errors.py --swap-slots), surplus (extra masks per slot; records without the
field are surplus 0), end bias and decoding config, on the items every length mode /
surplus level of that model has (LLaDA2.0 ran those on a 100-item subset). Columns: set accuracy,
syntax rate, cross-call error rate (CCER), single-call error rate (SCER), and the share
of items with an overfilled value (a wrong value that starts with a gold value and goes
on: "Taylor Swift. Swift", 5000000000 for 500000).

  python scripts/length_analysis.py results/dream/bfcl_skel_k.jsonl results/dream/bfcl_surplus.jsonl \
      results/dream/bfcl_endbias.jsonl results/llada2/bfcl_skel_k.jsonl results/llada2/bfcl_surplus.jsonl \
      results/qwen/bfcl_skeleton.jsonl results/qwen/bfcl_surplus.jsonl --csv results/summary/length.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.eval.taxonomy import is_cross_call, is_single_call  # noqa: E402


def _norm(v):
    return json.dumps(v, ensure_ascii=False).strip('"') if not isinstance(v, str) else v


def overfilled(example, detail):
    """A wrong value that is a gold value followed by more text."""
    if detail["label"] != "wrong_value" or "gold" not in detail:
        return False
    v = _norm(detail.get("value"))
    acc = example.gold_calls[detail["gold"]][1].get(detail["param"], [])
    return any(a != "" and v != _norm(a) and v.startswith(_norm(a)) for a in acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
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
                     if r.get("mode") == "skeleton" and "diagnosis" in r and r["id"] in exs
                     and r.get("length_mode") != "swap"]

    def model(r):
        return r.get("model_id", r.get("model"))

    surpluses = defaultdict(lambda: defaultdict(set))  # model -> (length mode, surplus) -> item ids
    for r in recs:
        surpluses[model(r)][r.get("length_mode", "oracle"), r.get("surplus", 0)].add(r["id"])
    common = {m: set.intersection(*by_s.values()) for m, by_s in surpluses.items()}

    groups = defaultdict(list)
    for r in recs:
        m = model(r)
        if r["id"] in common[m]:
            groups[(m, r.get("length_mode", "oracle"), r.get("surplus", 0), r.get("end_bias", 0.0) or 0.0,
                    r["cfg_tag"])].append(r)

    rows = []
    for (m, lm, s, b, tag), rs in sorted(groups.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][4], kv[0][2])):
        n = len(rs)
        labels = [r["diagnosis"]["labels"] for r in rs]
        rows.append({
            "model": m.split("/")[-1], "length_mode": lm, "surplus": s, "end_bias": b, "cfg": tag, "n": n,
            "set_acc": sum(r["diagnosis"]["correct"] for r in rs) / n,
            "syntax": sum(r["syntax_ok"] for r in rs) / n,
            "ccer": sum(is_cross_call(lab) for lab in labels) / n,
            "scer": sum(is_single_call(lab) for lab in labels) / n,
            "overfill": sum(any(overfilled(exs[r["id"]], d) for d in r["diagnosis"]["details"]) for r in rs) / n,
            "nfe": sum(r.get("nfe", 0) for r in rs) / n,
        })
    cols = ["model", "length_mode", "surplus", "end_bias", "cfg", "n", "set_acc", "syntax", "ccer", "scer", "overfill", "nfe"]
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
