"""Quick read-out of experiment E (the Figure 3 points not run before) next to the points that
were: set accuracy per model, slot surplus s and decoding, and the probe's mean probability of a
closing token. Reads the run records as they are (the BFCL verdict stored at run time).

  python scripts/exp_e_quick.py --results results > results/summary/exp_e_quick.md
"""

import argparse
import json
import os
from collections import defaultdict


def records(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in map(json.loads, f) if "error" not in r]


def acc_table(title, rows, key):
    out = [f"### {title}", "", "| s | decoding | requests | set accuracy % |", "|---|---|---|---|"]
    agg = defaultdict(list)
    for r in rows:
        agg[key(r)].append(bool(r["diagnosis"]["correct"]))
    for (s, tag), v in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out.append(f"| {s} | {tag} | {len(v)} | {100 * sum(v) / len(v):.1f} |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    D, L, Q = (f"{a.results}/{m}" for m in ("dream", "llada2", "qwen"))
    surplus = lambda r: (r.get("surplus") or 0, r["cfg_tag"])
    out = ["# Experiment E: quick read-out", ""]
    out += acc_table("(a) Dream, closing token in the slot, k=1 (bfcl_closer.jsonl)",
                     [r for r in records(f"{D}/bfcl_closer.jsonl") if "_k1_" in r["cfg_tag"]], surplus)
    out += acc_table("(a) Qwen2.5 AR (bfcl_skeleton.jsonl s=0, bfcl_surplus.jsonl)",
                     records(f"{Q}/bfcl_skeleton.jsonl") + records(f"{Q}/bfcl_surplus.jsonl"), surplus)
    out += acc_table("(b) LLaDA2.0, 100 requests (bfcl_surplus.jsonl; s=0 is in bfcl_skel_k.jsonl)",
                     records(f"{L}/bfcl_surplus.jsonl"), surplus)
    out += ["### (c) probe: mean P(closing token) over slots", "", "| model | interface | s | slots | mean P(close) % |",
            "|---|---|---|---|---|"]
    probes = [("Dream", "original", s, f"{D}/length_prior_s{s}.jsonl") for s in (1, 2, 8)]
    probes += [("Dream", "original", 4, f"{D}/length_prior.jsonl")]
    probes += [("Dream", "closing token in slot", s, f"{D}/length_prior_closer_s{s}.jsonl") for s in (0, 1, 2, 4, 8)]
    probes += [("LLaDA2.0", "original", s, f"{L}/length_prior_s{s}.jsonl") for s in (1, 2, 8)]
    probes += [("LLaDA2.0", "original", 4, f"{L}/length_prior.jsonl")]
    for model, iface, s, path in sorted(probes, key=lambda p: (p[0], p[1], p[2])):
        rs = records(path)
        if rs:
            out.append(f"| {model} | {iface} | {s} | {len(rs)} | {100 * sum(r['p_close'] for r in rs) / len(rs):.1f} |")
        else:
            out.append(f"| {model} | {iface} | {s} | -- | not run |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
