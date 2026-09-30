"""Run an autoregressive baseline with the same prompt and modes as the dLLMs.

  python scripts/run_ar.py --model Qwen/Qwen2.5-7B-Instruct \
      --data bfcl:parallel,parallel_multiple --mode skeleton --out results/qwen_skeleton.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.ar import ARModel, run_example_ar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--mode", default="free", choices=["free", "skeleton"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    examples = [ex for spec in args.data for ex in load_examples(spec, args.bfcl_dir)]
    if args.limit:
        examples = examples[: args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["id"], r["mode"]) for r in map(json.loads, f)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ar = ARModel(args.model, device=args.device)
    with open(args.out, "a") as f:
        for i, ex in enumerate(examples):
            if (ex.id, args.mode) in done:
                continue
            try:
                rec = run_example_ar(ar, ex, args.mode, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                rec = {"id": ex.id, "category": ex.category, "mode": args.mode, "error": repr(e)}
            rec["model_id"] = args.model
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(examples)}", flush=True)


if __name__ == "__main__":
    main()
