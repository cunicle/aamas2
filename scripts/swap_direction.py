"""Swapped and lengthened slots split by direction, read from their own tokens (CPU, raw records).

The swap gives each changed slot a sibling's length: a slot made shorter than its own value cannot
hold it, so taking the sibling's value that fits is forced there; only a lengthened slot has a free
choice between its own value (closing early) and the sibling's. This splits the swapped slots of
Dream at k=1 into lengthened and shortened ones, in both interfaces (closer after the slot, the
paper's; closer in the slot, experiment C3), and adds the one-sided lengthening. Shares come with a
95% bootstrap interval that resamples requests (slots of one request are not independent).

  python scripts/swap_direction.py --results results
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from closer_analysis import classify_value, read_jsonl, slot_contents  # noqa: E402
from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import lengths_from_list, onesided_pairs, oracle_lengths  # noqa: E402

DREAM = "Dream-org/Dream-v0-Instruct-7B"
K1 = "confidence_k1_tnone_bfull_T0.0"
KINDS = ("own", "sibling_fit", "overfill", "other")


def clustered(per_req, kind, resamples=10000, seed=0):
    """Share of slots of `kind` over requests {id: Counter}, with a request-resampling 95% interval."""
    ids = sorted(per_req)
    k = np.array([per_req[i][kind] for i in ids], float)
    n = np.array([sum(per_req[i].values()) for i in ids], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(resamples, len(ids)))
    boot = k[idx].sum(1) / n[idx].sum(1)
    return 100 * k.sum() / n.sum(), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DREAM, trust_remote_code=True)
    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple", args.bfcl_dir)}
    d = f"{args.results}/dream"
    files = {("swap", "original"): f"{d}/bfcl_swap.jsonl", ("swap", "variant"): f"{d}/bfcl_closer_swap.jsonl",
             ("onesided", "original"): f"{d}/bfcl_onesided.jsonl",
             ("onesided", "variant"): f"{d}/bfcl_closer_onesided.jsonl"}
    out = ["| lengths | interface | slots | n slots | " + " | ".join(KINDS) + " |", "|---|---|---|---|" + "---|" * len(KINDS)]
    for (mode, iface), path in files.items():
        if not os.path.exists(path):
            continue
        per = defaultdict(lambda: defaultdict(Counter))  # direction -> id -> kinds
        for r in read_jsonl(path):
            if r.get("cfg_tag") != K1 or r.get("length_mode") != mode or "error" in r:
                continue
            ex = exs[r["id"]]
            oracle = oracle_lengths(tok, ex)
            lens = lengths_from_list(r["lengths"])
            if mode == "onesided":
                targets = {(i, p): "lengthened" for (_, p), (i, _) in onesided_pairs(tok, ex).items()}
            else:
                targets = {cp: ("lengthened" if lens[cp] > oracle[cp] else "shortened")
                           for cp in oracle if lens[cp] != oracle[cp]}
            vals = slot_contents(tok, ex, r, iface == "variant")
            for (ci, p), direction in targets.items():
                v, _ = vals[ci, p]
                k = classify_value(ex, ci, p, v, lens[ci, p], oracle)
                per[direction][r["id"]][k if k in KINDS else "other"] += 1
        for direction, reqs in sorted(per.items()):
            n = sum(sum(c.values()) for c in reqs.values())
            cells = []
            for kind in KINDS:
                v, lo, hi = clustered(reqs, kind)
                cells.append(f"{v:.1f} [{lo:.1f}, {hi:.1f}]")
            out.append(f"| {mode} | {iface} | {direction} | {n} | " + " | ".join(cells) + " |")
    print("Dream, k=1, slots read from their own tokens; % of slots with a 95% request-resampling interval.\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
