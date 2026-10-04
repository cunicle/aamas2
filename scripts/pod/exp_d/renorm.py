"""Experiment D report helper, NOT pre-registered: re-score format-tolerant records offline with an
alternative normalization of number slots, from the stored final canvases (no GPU). The plan's
normalization (normalize_value) keeps a number's longest well-formed prefix, so ' 15_000' -> ' 15'
and ' 30. 45' -> ' 30.'. The alternative first drops '_' between digits and a '.' not followed by a
digit. First checks that the reconstruction with the plan's normalization reproduces every stored
diagnosis.

  python renorm.py <bfcl_tolerant.jsonl>
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import (STRING_TYPES, build_skeleton, lengths_from_list, normalize_value,  # noqa: E402
                                          slot_class, slot_closers, token_texts, value_end)
from ptcdiag.eval.taxonomy import diagnose  # noqa: E402
from ptcdiag.prompting import parse_tool_calls  # noqa: E402


def alt_normalize(raw, t):
    if t in ("integer", "float"):
        raw = re.sub(r"(?<=\d)_(?=\d)", "", raw)
        raw = re.sub(r"(?<=\d)\.(?!\d)", "", raw)
    return normalize_value(raw, t)


def rescore(tok, tt, ex, r, norm):
    special = set(tok.all_special_ids)
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens)
    gen = r["gen_ids"]
    texts = {}
    for s in slots:
        st = ["" if gen[g] in special or gen[g] >= len(tt) else tt[gen[g]] for g in s.positions]
        end = value_end(st, slot_closers(s.type), slot_class(s.type) == "generic")
        if slot_class(s.type) != "generic":
            upto = s.positions[:end[0]] if end is not None else s.positions
            texts[s.call, s.param] = norm(tok.decode([gen[g] for g in upto if gen[g] not in special]), s.type), s.type
        else:
            texts[s.call, s.param] = ("".join(st) if end is None else "".join(st[:end[0]]) + st[end[0]][:end[1]]), s.type
    # the decoded text: the skeleton with each slot's text, as the decoder writes it
    out = []
    for ci, (fname, params) in enumerate(ex.gold_calls):
        args = []
        for p in params:
            if (ci, p) not in texts:
                continue
            v, t = texts[ci, p]
            args.append(f'"{p}": "{v}"' if t in STRING_TYPES else f'"{p}":{v}')
        out.append('{"name": "%s", "arguments": {%s}}' % (fname, ", ".join(args)))
    text = "[" + ", ".join(out) + "]"
    return diagnose(ex, parse_tool_calls(text)).correct, text


def main(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    tt = token_texts(tok)
    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}
    by = defaultdict(list)
    for r in map(json.loads, open(path, encoding="utf-8")):
        if "error" not in r:
            by[(r["cfg_tag"], r.get("surplus", 0), r.get("length_mode", "oracle"))].append(r)
    print("| decoding | surplus | lengths | n | stored set acc % | reproduced | alternative set acc % | flips (wrong->right, right->wrong) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, rs in sorted(by.items()):
        rep = alt = up = down = 0
        for r in rs:
            ex = exs[r["id"]]
            c0, _ = rescore(tok, tt, ex, r, normalize_value)
            c1, _ = rescore(tok, tt, ex, r, alt_normalize)
            rep += c0 == r["diagnosis"]["correct"]
            alt += c1
            up += c1 and not r["diagnosis"]["correct"]
            down += r["diagnosis"]["correct"] and not c1
        st = sum(r["diagnosis"]["correct"] for r in rs)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {len(rs)} | {100 * st / len(rs):.1f} | {rep}/{len(rs)} | "
              f"{100 * alt / len(rs):.1f} | {up}, {down} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
