"""Experiment D report helper (not part of the paper's analysis): swap / one-sided slot classes of
the format-tolerant runs next to the original interface, computed as scripts/swap_direction.py does
(slots read from their own tokens), with tolerant slots read as the tolerant decoder reads them
(normalize_value on the slot text up to its closer).

  python swap_tolerant.py <new results dir> <10-03 results dir> <exp C results dir>
"""
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from closer_analysis import classify_value, read_jsonl, slot_contents  # noqa: E402
from swap_direction import KINDS, clustered  # noqa: E402
from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import (STRING_TYPES, build_skeleton, lengths_from_list, normalize_value,  # noqa: E402
                                          onesided_pairs, oracle_lengths, slot_class, slot_closers, value_end)

K1 = "confidence_k1_tnone_bfull_T0.0"


def tolerant_contents(tok, ex, r):
    """{(call, param): value} of a tolerant record, each slot read on its own as `cuts` reads it."""
    special = set(tok.all_special_ids)
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens)
    gen = r["gen_ids"]
    from ptcdiag.decoding.constraints import token_texts
    tt = token_texts(tok)
    out = {}
    for s in slots:
        st = ["" if gen[g] in special or gen[g] >= len(tt) else tt[gen[g]] for g in s.positions]
        end = value_end(st, slot_closers(s.type), slot_class(s.type) == "generic")
        if slot_class(s.type) != "generic":
            upto = s.positions[:end[0]] if end is not None else s.positions
            text = normalize_value(tok.decode([gen[g] for g in upto if gen[g] not in special]), s.type)
        else:
            text = "".join(st) if end is None else "".join(st[:end[0]]) + st[end[0]][:end[1]]
        if s.type in STRING_TYPES:
            v = text
        else:
            try:
                v = json.loads(text.strip())
            except ValueError:
                v = text.strip()
        out[s.call, s.param] = v
    return out


def main(new, rel, relc):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}
    files = [("swap", "original", f"{rel}/dream/bfcl_swap.jsonl"),
             ("swap", "tolerant", f"{new}/dream/bfcl_tolerant_swap.jsonl"),
             ("onesided", "original", f"{relc}/dream/bfcl_onesided.jsonl"),
             ("onesided", "tolerant", f"{new}/dream/bfcl_tolerant_onesided.jsonl")]
    out = ["| lengths | interface | slots | n slots | " + " | ".join(KINDS) + " |",
           "|---|---|---|---|" + "---|" * len(KINDS)]
    for mode, iface, path in files:
        if not os.path.exists(path):
            out.append(f"| {mode} | {iface} | TBD (no file) |")
            continue
        per = defaultdict(lambda: defaultdict(Counter))
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
            if iface == "tolerant":
                vals = tolerant_contents(tok, ex, r)
            else:
                vals = {k: v for k, (v, _) in slot_contents(tok, ex, r, False).items()}
            for (ci, p), direction in targets.items():
                k = classify_value(ex, ci, p, vals[ci, p], lens[ci, p], oracle)
                per[direction][r["id"]][k if k in KINDS else "other"] += 1
        for direction, reqs in sorted(per.items()):
            n = sum(sum(c.values()) for c in reqs.values())
            cells = []
            for kind in KINDS:
                v, lo, hi = clustered(reqs, kind)
                cells.append(f"{v:.1f} [{lo:.1f}, {hi:.1f}]")
            out.append(f"| {mode} | {iface} | {direction} | {n} | " + " | ".join(cells) + " |")
    print("Dream, k=1, slots read from their own tokens (tolerant slots normalized as decoded); "
          "% of slots with a 95% request-resampling interval.\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
