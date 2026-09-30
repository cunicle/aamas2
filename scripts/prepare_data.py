"""Download BFCL v4 (Python single-turn categories) and generate ParaProbe.

  python scripts/prepare_data.py --bfcl-dir data/bfcl --probe-out data/paraprobe.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import bfcl, save_jsonl  # noqa: E402
from ptcdiag.data.paraprobe import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--probe-out", default="data/paraprobe.jsonl")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bfcl.download(args.bfcl_dir)
    for c in bfcl.CATEGORIES:
        print(f"bfcl {c}: {len(bfcl.load(c, args.bfcl_dir))}")
    probes = generate(per_cell=args.per_cell, seed=args.seed)
    os.makedirs(os.path.dirname(args.probe_out) or ".", exist_ok=True)
    save_jsonl(probes, args.probe_out)
    print(f"paraprobe: {len(probes)} examples -> {args.probe_out}")


if __name__ == "__main__":
    main()
