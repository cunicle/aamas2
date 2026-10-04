"""Cross-call errors where the canvas cannot tell the calls apart, and the order the models fill them in.

With per-value slot lengths, sibling slots of different lengths can be told apart by
length alone, so a low cross-call error rate (CCER) could come from the canvas rather
than from the model. Two checks on the oracle-length skeleton runs (surplus 0):

  symmetric   set accuracy and CCER on the items whose every sibling group is
              length-symmetric (`length_symmetric`), next to all items
  order       on BFCL `parallel` items (one function; gold calls listed in mention
              order) that are length-symmetric: among syntax-ok records with every
              call matched, the share whose optimal matching is the identity, i.e. the
              model wrote the i-th mentioned entity into the i-th call's slots; and the
              same share over all such items (records that do not parse or leave a call
              unmatched count as not in order)

  python scripts/symmetry.py Dream-org/Dream-v0-Instruct-7B results/dream/bfcl_skel_*.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer  # noqa: E402

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import length_symmetric  # noqa: E402
from ptcdiag.eval.taxonomy import is_cross_call  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    sym = {i: all(length_symmetric(tok, e).values()) for i, e in exs.items()}
    groups = defaultdict(list)
    for path in args.results:
        with open(path) as f:
            for r in map(json.loads, f):
                if r.get("mode") == "skeleton" and "diagnosis" in r and r["id"] in exs \
                        and not r.get("surplus", 0) and r.get("length_mode", "oracle") == "oracle":
                    groups[r["cfg_tag"]].append(r)

    def acc(rs):
        return sum(r["diagnosis"]["correct"] for r in rs) / max(len(rs), 1)

    def ccer(rs):
        return sum(is_cross_call(r["diagnosis"]["labels"]) for r in rs) / max(len(rs), 1)

    print(f"## {args.model}: all items vs length-symmetric items")
    print("| cfg | n | set_acc | ccer | n sym | set_acc sym | ccer sym |")
    print("|---|---|---|---|---|---|---|")
    for tag, rs in sorted(groups.items()):
        sy = [r for r in rs if sym[r["id"]]]
        print(f"| {tag} | {len(rs)} | {acc(rs):.3f} | {ccer(rs):.3f} | {len(sy)} | {acc(sy):.3f} | {ccer(sy):.3f} |")

    print(f"\n## {args.model}: symmetric BFCL parallel items, slots filled in mention order")
    print("| cfg | records (syntax ok, all calls matched) | identity order | correct "
          "| all symmetric parallel items | identity order over all |")
    print("|---|---|---|---|---|---|")
    for tag, rs in sorted(groups.items()):
        n = ident = ok = n_all = 0
        for r in rs:
            ex = exs[r["id"]]
            if ex.category != "parallel" or not sym[r["id"]]:
                continue
            n_all += 1  # fixed denominator: unparsed or unmatched records count as not in order
            m = r["diagnosis"].get("matching", [])
            if not r.get("syntax_ok") or len(m) != len(ex.gold_calls):
                continue
            n += 1
            ident += all(p == q for p, q in m)
            ok += r["diagnosis"]["correct"]
        if n:
            print(f"| {tag} | {n} | {ident / n:.3f} | {ok / n:.3f} | {n_all} | {ident / n_all:.3f} |")


if __name__ == "__main__":
    main()
