"""Experiment D for the paper (CPU, raw records): format-tolerant slots and position agents.

  python scripts/exp_d_analysis.py --results results > results/summary/exp_d.md

1. Format-tolerant slots against the original interface on the same requests (Dream): set accuracy,
   requests that turn right or wrong, and canvases identical token for token.
2. Slot classes with surplus masks at k=1: tolerant slots as decoded (normalized), original slots from
   their own tokens (as scripts/closer_analysis.py reads them), and what the normalization removed.
3. Position agents (pos-anon): set accuracy on the 104 team-symmetric BFCL requests; agents whose own
   call holds a placeholder ('...' or null) instead of a value, by agent index; choose-N duplicates
   with and without the placeholder counted as a city.
Rates are exact fractions rounded half up once, as in the paper.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from closer_analysis import classify_value, slot_contents  # noqa: E402
from paper_tables import fmt  # noqa: E402
from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import (STRING_TYPES, build_skeleton, lengths_from_list, normalize_value,  # noqa: E402
                                          oracle_lengths, slot_class, slot_closers, token_texts, value_end)

DREAM = "Dream-org/Dream-v0-Instruct-7B"
KINDS = ("own", "overfill", "sibling_fit", "other")


def pct(a, b):
    return fmt(Fraction(100 * a, b)) if b else "--"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if "error" not in r]


def key(r):
    return r["cfg_tag"], r.get("surplus", 0), r.get("length_mode", "oracle")


def tolerant_contents(tok, ex, r):
    """{(call, param): (raw text, value)} of a tolerant record: each slot's text before its closer,
    normalized as the decoder normalizes it (normalize_value), as a string or parsed JSON."""
    texts, special = token_texts(tok), set(tok.all_special_ids)
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens)
    gen = r["gen_ids"]
    out = {}
    for s in slots:
        st = ["" if gen[g] in special or gen[g] >= len(texts) else texts[gen[g]] for g in s.positions]
        end = value_end(st, slot_closers(s.type), slot_class(s.type) == "generic")
        raw = "".join(st) if end is None else "".join(st[:end[0]]) + st[end[0]][:end[1]]
        if slot_class(s.type) == "generic":
            norm = raw
        else:
            norm = normalize_value(tok.decode([gen[g] for g in s.positions[:end[0] if end else None]
                                               if gen[g] not in special and gen[g] != tok.mask_token_id]), s.type)
        if s.type in STRING_TYPES:
            v = json.loads('"' + norm + '"') if norm is not None else norm
        else:
            try:
                v = json.loads(norm.strip())
            except ValueError:
                v = norm.strip()
        out[s.call, s.param] = (raw, v)
    return out


def filler_kind(raw):
    t = raw.rstrip()
    if t.endswith("\\n") or "\\n" in raw:
        return "escaped line break"
    if re.search(r"\d\.0*$", t) or re.search(r"\d\.\s*$", t):
        return "decimal point"
    if re.search(r"\d_\d", t):
        return "digit group"
    if re.search(r"[0-9.][A-Za-z]+$", t):
        return "letter suffix"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DREAM, trust_remote_code=True)
    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple", args.bfcl_dir)}
    d = f"{args.results}/dream"
    out = ["# Experiment D (scripts/exp_d_analysis.py)", ""]

    # 1. tolerant against the original interface, same requests
    orig = {}
    for f in ("bfcl_skel_k", "bfcl_surplus", "bfcl_swap", "bfcl_estimate", "bfcl_onesided"):
        for r in read(f"{d}/{f}.jsonl"):
            if not r.get("end_bias") and not r.get("closer_in_slot"):
                orig[key(r) + (r["id"],)] = r
    tol = defaultdict(dict)
    for f in ("bfcl_tolerant", "bfcl_tolerant_swap", "bfcl_tolerant_onesided", "bfcl_tolerant_estimate"):
        for r in read(f"{d}/{f}.jsonl"):
            tol[key(r)][r["id"]] = r
    out += ["## 1. Format-tolerant slots against the original interface (Dream, same requests)", "",
            "| decoding | surplus | lengths | n | tolerant set acc % | original set acc % | wrong -> right | right -> wrong "
            "| identical canvases | tolerant parses % | original parses % |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(tol, key=lambda k: (k[2], k[1], k[0])):
        rs = tol[k]
        pairs = [(rs[i], orig[k + (i,)]) for i in rs if k + (i,) in orig]
        n = len(pairs)
        ta = sum(a["diagnosis"]["correct"] for a, _ in pairs)
        oa = sum(b["diagnosis"]["correct"] for _, b in pairs)
        up = sum(a["diagnosis"]["correct"] and not b["diagnosis"]["correct"] for a, b in pairs)
        down = sum(b["diagnosis"]["correct"] and not a["diagnosis"]["correct"] for a, b in pairs)
        same = sum(a["gen_ids"] == b["gen_ids"] for a, b in pairs)
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {n} | {pct(ta, n)} | {pct(oa, n)} | {up} | {down} | {same}/{n} "
                   f"| {pct(sum(a['syntax_ok'] for a, _ in pairs), n)} | {pct(sum(b['syntax_ok'] for _, b in pairs), n)} |")
    out.append("")

    # 2. slot classes with surplus masks, k=1
    k1 = "confidence_k1_tnone_bfull_T0.0"
    out += ["## 2. Slot classes with surplus masks (Dream, k=1, % of slots)", "",
            "| surplus | interface | n slots | " + " | ".join(KINDS) + " |", "|---|---|---|" + "---|" * len(KINDS)]
    removed = {}
    for s in (1, 2, 8):
        for iface in ("original", "tolerant"):
            kinds, n = Counter(), 0
            rem = Counter()
            for i, rt in sorted(tol[k1, s, "oracle"].items()):
                ex = exs[i]
                oracle = oracle_lengths(tok, ex)
                if iface == "original":
                    vals = {cp: v for cp, (v, _) in slot_contents(tok, ex, orig[(k1, s, "oracle", i)], False).items()}
                else:
                    tc = tolerant_contents(tok, ex, rt)
                    vals = {cp: v for cp, (_, v) in tc.items()}
                    for (ci, p), (raw, v) in tc.items():
                        t = ex.function(ex.gold_calls[ci][0])["parameters"]["properties"].get(p, {}).get("type", "string")
                        if slot_class(t) == "generic":
                            continue
                        rem["slots"] += 1
                        if normalize_value(raw, t) != raw:
                            rem["changed"] += 1
                            rem[filler_kind(raw)] += 1
                for (ci, p), v in vals.items():
                    kk = classify_value(ex, ci, p, v, oracle[ci, p] + s, oracle)
                    kinds[kk if kk in KINDS else "other"] += 1
                    n += 1
            out.append(f"| +{s} | {iface} | {n} | " + " | ".join(pct(kinds[k], n) for k in KINDS) + " |")
            if iface == "tolerant":
                removed[s] = rem
    out += ["", "What the normalization changed (tolerant string, number and boolean slots, k=1):", "",
            "| surplus | slots | changed | escaped line break | decimal point | digit group | letter suffix | other |",
            "|---|---|---|---|---|---|---|---|"]
    for s, rem in removed.items():
        out.append(f"| +{s} | {rem['slots']} | {rem['changed']} ({pct(rem['changed'], rem['slots'])}%) | "
                   + " | ".join(str(rem[c]) for c in ("escaped line break", "decimal point", "digit group",
                                                      "letter suffix", "other")) + " |")
    out.append("")

    # 3. position agents
    out += ["## 3. Position agents (pos-anon)", "",
            "| model | data | teams | correct | agents | own call holds a placeholder | by agent index (1, 2, 3, 4+) |",
            "|---|---|---|---|---|---|---|"]
    for model in ("dream", "qwen"):
        for f in ("agents_d", "agents_pos"):
            path = f"{args.results}/{model}/{f}.jsonl"
            if not os.path.exists(path):
                continue
            rs = read(path)
            ph, by = 0, defaultdict(lambda: [0, 0])
            for r in rs:
                for a in r["agents"]:
                    if f == "agents_d":
                        call = a.get("call")
                        held = call is not None and any(v in ("...", None) for v in call["arguments"].values())
                    else:
                        held = a.get("city") == "..."
                    idx = min(a["i"], 4)
                    by[idx][0] += held
                    by[idx][1] += 1
                    ph += held
            agents = sum(v[1] for v in by.values())
            correct = (f"{pct(sum(r['team']['diagnosis']['correct'] for r in rs), len(rs))}%" if f == "agents_d" else "--")
            out.append(f"| {model} | {f} | {len(rs)} | {correct} | {agents} | {ph} ({pct(ph, agents)}%) | "
                       + ", ".join(f"{by[i][0]}/{by[i][1]}" for i in (1, 2, 3, 4)) + " |")
    out += ["", "choose-N teams of position agents: a duplicate city, counting '...' as a city or not",
            "", "| model | variant | teams | duplicate | duplicate among real cities | some agent wrote '...' |",
            "|---|---|---|---|---|---|"]
    for model in ("dream", "qwen"):
        path = f"{args.results}/{model}/agents_pos.jsonl"
        if not os.path.exists(path):
            continue
        groups = defaultdict(list)
        for r in read(path):
            groups[r["variant"]].append(r)
        for var, rs in sorted(groups.items()):
            dup = sum(len([c for c in r["cities"] if c]) != len({c for c in r["cities"] if c}) for r in rs)
            real = [[c for c in r["cities"] if c and c != "..."] for r in rs]
            dup_real = sum(len(c) != len(set(c)) for c in real)
            dots = sum("..." in r["cities"] for r in rs)
            out.append(f"| {model} | {var} | {len(rs)} | {pct(dup, len(rs))}% | {pct(dup_real, len(rs))}% | "
                       f"{pct(dots, len(rs))}% |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
