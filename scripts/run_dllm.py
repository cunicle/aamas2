"""Run a dLLM over datasets x decoding configurations; one JSONL record per run.

Example (dose-response on BFCL parallel categories, oracle skeleton):
  python scripts/run_dllm.py --model Dream-org/Dream-v0-Instruct-7B \
      --data bfcl:parallel,parallel_multiple --mode skeleton \
      --k 1,2,4,8 --block-length none --order confidence \
      --out results/dream_skeleton_k.jsonl

Re-running with the same --out skips runs that are already there.
"""

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.adapters import load_adapter  # noqa: E402
from ptcdiag.decoding.sampler import DecodeConfig  # noqa: E402
from ptcdiag.pipeline import run_example  # noqa: E402


def as_list(s, typ):
    return [None if v.strip().lower() == "none" else typ(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True,
                    help="bfcl:<cat>[,<cat>] or probe:<path.jsonl>; repeatable")
    ap.add_argument("--mode", default="free", choices=["free", "skeleton"])
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", default="none", help="comma list; 'none' = no blocks")
    ap.add_argument("--k", default="1", help="comma list of tokens per step")
    ap.add_argument("--threshold", default="none", help="comma list; 'none' = fixed k")
    ap.add_argument("--order", default="confidence", help="comma list of orders")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    examples = [ex for spec in args.data for ex in load_examples(spec, args.bfcl_dir)]
    if args.limit:
        examples = examples[: args.limit]
    configs = [
        DecodeConfig(gen_length=args.gen_length, block_length=b, k=k, threshold=t, order=o,
                     temperature=args.temperature, seed=s)
        for b, k, t, o, s in itertools.product(
            as_list(args.block_length, int), as_list(args.k, int), as_list(args.threshold, float),
            args.order.split(","), as_list(args.seeds, int))
    ]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["mode"], r["cfg_tag"], r["cfg"]["seed"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    adapter = load_adapter(args.model, device=args.device)
    total = len(examples) * len(configs)
    print(f"{len(examples)} examples x {len(configs)} configs = {total} runs "
          f"({len(done)} already done)", flush=True)

    t0, n = time.time(), 0
    with open(args.out, "a") as f:
        for cfg in configs:
            for ex in examples:
                key = (ex.id, args.mode, cfg.tag(), cfg.seed)
                if key in done:
                    continue
                try:
                    rec = run_example(adapter, ex, cfg, args.mode, keep_trace=not args.no_trace)
                except Exception as e:  # keep the sweep going; failures are recorded
                    rec = {"id": ex.id, "category": ex.category, "mode": args.mode,
                           "cfg": cfg.to_dict(), "cfg_tag": cfg.tag(), "error": repr(e)}
                rec["model_id"] = args.model
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0:
                    rate = (time.time() - t0) / n
                    print(f"{n}/{total - len(done)}  {rate:.2f}s/run  cfg={cfg.tag()}", flush=True)


if __name__ == "__main__":
    main()
