"""Counterfactual attribution (Table 1): sequentialise the steps behind each error,
plus the placebo on correct outputs.

  python scripts/attribute.py --model Dream-org/Dream-v0-Instruct-7B \
      --results results/dream_skeleton_k.jsonl --data bfcl:parallel,parallel_multiple \
      --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/dream_attr.jsonl

Only deterministic configs (temperature 0, order != random) are valid here.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.analysis.counterfactual import attribute, placebo, reproduces  # noqa: E402
from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.adapters import load_adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--cfg-tag", required=True)
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument("--max-placebo", type=int, default=200)
    ap.add_argument("--check-repro", type=int, default=5, help="records to check for determinism")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    exs = {ex.id: ex for spec in args.data for ex in load_examples(spec, args.bfcl_dir)}
    with open(args.results) as f:
        recs = [r for r in map(json.loads, f)
                if r.get("cfg_tag") == args.cfg_tag and "trace" in r and r["id"] in exs]
    adapter = load_adapter(args.model, device=args.device)

    repro = [reproduces(adapter, exs[r["id"]], r, step=0) for r in recs[: args.check_repro]]
    print(f"determinism check: {sum(repro)}/{len(repro)} records reproduce exactly", flush=True)

    rng = random.Random(0)
    wrong = [r for r in recs if not r["diagnosis"]["correct"] and r["syntax_ok"]][: args.max_errors]
    right = [r for r in recs if r["diagnosis"]["correct"]]
    rng.shuffle(right)
    right = right[: args.max_placebo]

    fixed, tried, no_step = Counter(), Counter(), Counter()
    broken = n_placebo = 0
    with open(args.out, "w") as f:
        for r in wrong:
            for res in attribute(adapter, exs[r["id"]], r):
                tried[res["label"]] += 1
                no_step[res["label"]] += not res["steps_tried"]
                fixed[res["label"]] += res["fixed_by"] is not None
                f.write(json.dumps({"id": r["id"], "kind": "error", **res}) + "\n")
        for r in right:
            res = placebo(adapter, exs[r["id"]], r, rng)
            if res is None:
                continue
            n_placebo += 1
            broken += res["broken"]
            f.write(json.dumps({"id": r["id"], "kind": "placebo", **res}) + "\n")

    base = broken / n_placebo if n_placebo else float("nan")
    print(f"placebo: {broken}/{n_placebo} correct outputs broken (floor {base:.3f})")
    print("errors committed only in single-token steps cannot be caused by simultaneity "
          "and count as not attributable")
    for lab in sorted(tried):
        p = fixed[lab] / tried[lab]
        print(f"{lab:26s} n={tried[lab]:4d}  no-parallel-step={no_step[lab]:4d}  "
              f"fixed={fixed[lab]:4d} ({p:.3f})  attributable ~ {p - base:.3f}")


if __name__ == "__main__":
    main()
