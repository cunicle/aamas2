"""Experiment C3: the closer-in-slot interface variant next to the original skeleton (Dream, one canvas).

  python scripts/closer_analysis.py --results results --original <release 2026-10-03>/results \
      --csv results/summary/closer.csv > results/summary/closer.md

Original interface (the skeleton writes each closer after its slot; values end with closer + padding):
the 10-03 runs (dream/bfcl_skel_k, bfcl_surplus, bfcl_swap, length_prior*) and the new one-sided run
(--results dream/bfcl_onesided.jsonl). Variant (the model writes the closer inside the slot, one more
position per slot, spaces after it): --results dream/bfcl_closer*.jsonl, length_prior_closer_s*.jsonl.
Conditions: slot lengths exact (s=0) / +s, k = 1 or 16, on the 400 BFCL items; swap and one-sided
lengthening at k=1 on the 165 items where they change a slot. Each table compares the two interfaces
on the items both have for that condition.

  items      set accuracy, syntax, cross-call error rate, items with an overfilled value (as in
             scripts/length_analysis.py)
  slots      every slot's value against its own and its siblings' gold values (scripts/slot_errors.py):
             own / sibling_fit / sibling / overfill / truncated / other / unparsed; the slot length
             compared with a sibling's is the content length (oracle or swapped + s; the variant's
             closer position is not counted)
  swap       the same classes on the slots the swap changes (the paper's Table 3 slots), exact vs swap
  onesided   the lengthened slot (i*, p) of every unequal sibling group (`onesided_pairs`): own (its
             value, ended early) / overfill (its value and more) / sibling_fit (j*'s value: two calls
             with the same value) / other (of which: another sibling's value, truncated, unparsed,
             anything else); and whether j*'s exact slot still holds j*'s own value
  probe      teacher-forced, the first position after the gold value: mean P(closer) (tokens starting
             with the slot's closer), median P(pad), mean P(content), and the constrained pick. Original
             surplus s: s masks before the skeleton's closer; variant s: 1 + s positions before the
             next fixed text (s = 0: one position, which can only be the closer)
  closers    variant only: share of slots whose value ended with a closer, share of those whose
             closing token is the bare closer ('"', ',', '}') rather than e.g. '",' or '}}', and slots
             filled to the end without one (the output then does not parse)
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.constraints import (STRING_TYPES, build_skeleton, lengths_from_list,  # noqa: E402
                                          onesided_pairs, oracle_lengths, sibling_groups, slot_class,
                                          slot_closers, swapped_lengths, token_texts, value_end)
from ptcdiag.eval.taxonomy import is_cross_call, value_ok  # noqa: E402

DREAM = "Dream-org/Dream-v0-Instruct-7B"
TAGS = {1: "confidence_k1_tnone_bfull_T0.0", 16: "confidence_k16_tnone_bfull_T0.0"}
CONDS = [("oracle", 0, 1), ("oracle", 1, 1), ("oracle", 2, 1), ("oracle", 4, 1), ("oracle", 8, 1), ("oracle", 0, 16),
         ("oracle", 1, 16), ("swap", 0, 1), ("onesided", 0, 1)]
IFACES = ["original", "variant"]
ONESIDED = ["own", "overfill", "sibling_fit", "other"]
OTHER_SUB = ["other_sibling", "truncated", "unparsed", "wrong"]


def read_jsonl(path):
    if not os.path.exists(path):
        print(f"(missing {path})", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cond_of(r):
    """(length mode, surplus, k) of a skeleton record in one of CONDS, else None."""
    if r.get("mode") != "skeleton" or "diagnosis" not in r or r.get("end_bias", 0.0):
        return None
    k = next((k for k, tag in TAGS.items() if r.get("cfg_tag") == tag), None)
    c = (r.get("length_mode", "oracle"), r.get("surplus", 0), k)
    return c if c in CONDS else None


def cond_label(c):
    lm, s, k = c
    return f"{'exact' if lm == 'oracle' and not s else f'+{s}' if lm == 'oracle' else lm} k={k}"


def fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return "" if v is None else str(v)


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def item_metrics(exs, rs, la):
    n = len(rs)
    if not n:
        return {"n": 0}
    return {"n": n, "set_acc": sum(r["diagnosis"]["correct"] for r in rs) / n,
            "syntax": sum(bool(r["syntax_ok"]) for r in rs) / n,
            "ccer": sum(is_cross_call(r["diagnosis"]["labels"]) for r in rs) / n,
            "overfill": sum(any(la.overfilled(exs[r["id"]], d) for d in r["diagnosis"]["details"]) for r in rs) / n}


def slot_lengths_of(r, oracle):
    lens = dict(oracle)
    lens.update(lengths_from_list(r["lengths"]) if r.get("lengths") else {})
    return lens


def onesided_kind(ex, r, i, j, p):
    """Class of the value in the lengthened slot (i, p); j = j*."""
    fname, params = ex.gold_calls[i]
    calls = r.get("calls") or []
    if not r.get("syntax_ok") or i >= len(calls) or calls[i].get("name") != fname \
            or p not in calls[i].get("arguments", {}):
        return "unparsed"
    v, desc = calls[i]["arguments"][p], ex.function(fname)
    if value_ok(desc, p, v, params[p]):
        return "own"
    if value_ok(desc, p, v, ex.gold_calls[j][1][p]):
        return "sibling_fit"
    se = load_slot_errors()
    fv = se._flat(v)
    for a in params[p]:
        if a != "" and fv != se._flat(a) and fv.startswith(se._flat(a)):
            return "overfill"
    if any(value_ok(desc, p, v, ex.gold_calls[cj][1][p]) for cj in sibling_groups(ex)[fname, p] if cj not in (i, j)):
        return "other_sibling"
    for a in params[p]:
        if a != "" and fv and fv != se._flat(a) and se._flat(a).startswith(fv):
            return "truncated"
    return "wrong"


_SE = []


def load_slot_errors():
    if not _SE:
        import slot_errors

        _SE.append(slot_errors)
    return _SE[0]


def closer_stats(tok, ex, r):
    """Variant record: per slot, (ended, bare closer, overflow) from its canvas."""
    texts = token_texts(tok)
    special = set(tok.all_special_ids)
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens, closer_in_slot=True)
    gen = r["gen_ids"]
    if len(gen) != len(ids):
        return []
    out = []
    for s in slots:
        st = ["" if gen[g] in special or gen[g] >= len(texts) else texts[gen[g]] for g in s.positions]
        end = value_end(st, slot_closers(s.type), slot_class(s.type) == "generic")
        if end is None:
            out.append((False, False, True))
        else:
            i, off = end
            out.append((True, len(st[i]) == off + 1 and (off == 0 or slot_class(s.type) == "generic"), False))
    return out


def slot_contents(tok, ex, r, closer_in_slot):
    """{(call, param): (value, closed)} read from each slot's own tokens, whether or not the whole
    output parses: the text before the slot's closer (the whole slot when it has none; closed =
    False), as a string for string slots, else as JSON when it parses and as text otherwise. In
    the variant a single unclosed slot makes the output unparseable, so the item-level parse that
    scripts/slot_errors.py reads marks every slot of that item unparsed; this reads them one by one."""
    texts = token_texts(tok)
    special = set(tok.all_special_ids)
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens, closer_in_slot)
    gen = r["gen_ids"]
    if len(gen) != len(ids):
        return {}
    out = {}
    for s in slots:
        st = ["" if gen[g] in special or gen[g] >= len(texts) else texts[gen[g]] for g in s.positions]
        end = value_end(st, slot_closers(s.type), slot_class(s.type) == "generic")
        text = "".join(st) if end is None else "".join(st[:end[0]]) + st[end[0]][:end[1]]
        if s.type in STRING_TYPES:
            v = text
        else:
            try:
                v = json.loads(text.strip())
            except ValueError:
                v = text.strip()
        out[s.call, s.param] = (v, end is not None)
    return out


def classify_value(ex, ci, p, v, slot_len, oracle):
    """scripts/slot_errors.py's classes for a value read from its slot (no 'unparsed')."""
    se = load_slot_errors()
    fname, params = ex.gold_calls[ci]
    desc = ex.function(fname)
    if value_ok(desc, p, v, params[p]):
        return "own"
    for cj in [cj for cj in sibling_groups(ex).get((fname, p), []) if cj != ci]:
        if value_ok(desc, p, v, ex.gold_calls[cj][1][p]):
            return "sibling_fit" if oracle.get((cj, p)) == slot_len else "sibling"
    fv = se._flat(v)
    for a in params[p]:
        if a == "":
            continue
        fa = se._flat(a)
        if fv != fa and fv.startswith(fa):
            return "overfill"
        if fv and fv != fa and fa.startswith(fv):
            return "truncated"
    return "other"


def bare_vs_content(tok, ex, r):
    """Variant record with a trace: for every scalar slot closed with a longer token ('",', '}}'),
    whether the step's top-5 candidates rank a bare closer above every content token ('bare'),
    a content token above every bare closer ('content'), or show neither ('top5_closers')."""
    texts = token_texts(tok)
    special = set(tok.all_special_ids)
    bare = {t: tok(t, add_special_tokens=False)["input_ids"][0] for t in ('"', ",", "}")}
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens, closer_in_slot=True)
    gen, P = r["gen_ids"], r["trace"]["gen_start"]
    at = {p: (st, j) for st in r["trace"]["steps"] for j, p in enumerate(st["positions"])}
    out = []
    for s in slots:
        if slot_class(s.type) == "generic":
            continue
        st_ = ["" if gen[g] in special or gen[g] >= len(texts) else texts[gen[g]] for g in s.positions]
        end = value_end(st_, slot_closers(s.type), False)
        if end is None or len(st_[end[0]]) == 1 or P + s.positions[end[0]] not in at:
            continue
        step, j = at[P + s.positions[end[0]]]
        top = step["topk_ids"][j]
        ok = {bare['"']} if s.type in STRING_TYPES else {bare[","], bare["}"]}
        rb = next((k for k, t in enumerate(top) if t in ok), None)
        rc = next((k for k, t in enumerate(top) if not texts[t].startswith(slot_closers(s.type))), None)
        out.append("bare" if rb is not None and (rc is None or rb < rc)
                   else "content" if rc is not None else "top5_closers")
    return out


def render(tok, ex, r, closer_in_slot):
    """The final canvas: slots in brackets, spaces inside slots as '␣', padding as '∅'."""
    texts = token_texts(tok)
    special = set(tok.all_special_ids)
    lens = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lens, closer_in_slot)
    first = {s.positions[0] for s in slots}
    last = {s.positions[-1] for s in slots}
    inside = {g for s in slots for g in s.positions}
    out = []
    for g, t in enumerate(r["gen_ids"]):
        if g in first:
            out.append("⟦")
        txt = "∅" if t in special else texts[t] if t < len(texts) else "?"
        out.append(txt.replace(" ", "␣") if g in inside and not txt.strip() else txt)
        if g in last:
            out.append("⟧")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="results/ with experiment C's new files")
    ap.add_argument("--original", required=True, help="results/ of the 10-03 runs (release archive)")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--csv")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from transformers import AutoTokenizer

    import length_analysis as la

    se = load_slot_errors()
    tok = AutoTokenizer.from_pretrained(DREAM, trust_remote_code=True)
    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    D, O = os.path.join(args.results, "dream"), os.path.join(args.original, "dream")
    sources = {"original": [os.path.join(O, f) for f in ("bfcl_skel_k.jsonl", "bfcl_surplus.jsonl", "bfcl_swap.jsonl")]
               + [os.path.join(D, "bfcl_onesided.jsonl")],
               "variant": [os.path.join(D, f) for f in ("bfcl_closer.jsonl", "bfcl_closer_swap.jsonl",
                                                        "bfcl_closer_onesided.jsonl")]}
    recs = defaultdict(dict)  # (interface, cond) -> {id: record}
    for iface, paths in sources.items():
        for path in paths:
            for r in read_jsonl(path):
                c = cond_of(r)
                if c is None or r["id"] not in exs or bool(r.get("closer_in_slot")) != (iface == "variant"):
                    continue
                if r.get("model_id", DREAM) != DREAM:
                    continue
                recs[iface, c][r["id"]] = r
    oracle = {i: oracle_lengths(tok, e) for i, e in exs.items()}

    def common(c):
        have = [set(recs[f, c]) for f in IFACES if recs[f, c]]
        return set.intersection(*have) if have else set()

    rows_items, rows_slots, rows_swap, rows_one, rows_probe, rows_closers = [], [], [], [], [], []
    for c in CONDS:
        ids = common(c)
        row = {"lengths": cond_label(c)}
        for f in IFACES:
            m = item_metrics(exs, [recs[f, c][i] for i in sorted(ids) if i in recs[f, c]], la)
            row[f"n_{f}"] = m["n"] or None
            for k in ("set_acc", "syntax", "ccer", "overfill"):
                row[f"{k}_{f}"] = m.get(k)
        rows_items.append(row)
        for f in IFACES:
            kinds = Counter()
            for i in sorted(ids):
                r = recs[f, c].get(i)
                if r is None:
                    continue
                r["_oracle"] = oracle[i]
                lens = slot_lengths_of(r, oracle[i])
                for (ci, p) in oracle[i]:
                    kinds[se.classify(exs[i], r, ci, p, lens[ci, p] + c[1])] += 1
            tot = sum(kinds.values())
            if tot:
                rows_slots.append({"lengths": cond_label(c), "interface": f, "n_items": len(ids), "n_slots": tot,
                                   **{k: kinds[k] / tot for k in se.KINDS}})

    # swap slots: exact vs swap at k=1 on the swap items
    swap_ids = common(("swap", 0, 1))
    for c in [("oracle", 0, 1), ("swap", 0, 1)]:
        for f in IFACES:
            kinds = defaultdict(Counter)
            for i in sorted(swap_ids):
                r = recs[f, c].get(i)
                if r is None:
                    continue
                r["_oracle"] = oracle[i]
                lens, sw = slot_lengths_of(r, oracle[i]), swapped_lengths(tok, exs[i])
                for (ci, p), n in oracle[i].items():
                    if sw[ci, p] == n:
                        continue
                    k = se.classify(exs[i], r, ci, p, lens[ci, p] + c[1])
                    kinds["all"][k] += 1
                    kinds["longer" if sw[ci, p] > n else "shorter"][k] += 1
            for d in ("all", "longer", "shorter"):
                tot = sum(kinds[d].values())
                if tot:
                    rows_swap.append({"lengths": cond_label(c), "interface": f, "slots": d, "n_slots": tot,
                                      **{k: kinds[d][k] / tot for k in se.KINDS}})

    # one-sided lengthening: the lengthened slots
    one_ids = common(("onesided", 0, 1))
    for f in IFACES:
        kinds, jown, n_j = Counter(), 0, 0
        for i in sorted(one_ids):
            r = recs[f, ("onesided", 0, 1)].get(i)
            if r is None:
                continue
            ex = exs[i]
            for (fname, p), (ii, jj) in onesided_pairs(tok, ex).items():
                kinds[onesided_kind(ex, r, ii, jj, p)] += 1
                n_j += 1
                calls = r.get("calls") or []
                jown += bool(r.get("syntax_ok") and jj < len(calls) and p in calls[jj].get("arguments", {})
                             and value_ok(ex.function(fname), p, calls[jj]["arguments"][p], ex.gold_calls[jj][1][p]))
        tot = sum(kinds.values())
        if tot:
            row = {"interface": f, "n_items": len(one_ids), "n_slots": tot}
            row.update({k: kinds[k] / tot for k in ("own", "overfill", "sibling_fit")})
            row["other"] = sum(kinds[k] for k in OTHER_SUB) / tot
            row.update({k: kinds[k] / tot for k in OTHER_SUB})
            row["jstar_own"] = jown / n_j
            rows_one.append(row)

    # probe
    probe = {("original", 1): "length_prior_s1.jsonl", ("original", 2): "length_prior_s2.jsonl",
             ("original", 4): "length_prior.jsonl", ("original", 8): "length_prior_s8.jsonl"}
    probe.update({("variant", s): f"length_prior_closer_s{s}.jsonl" for s in (0, 1, 2, 4, 8)})
    by_class = defaultdict(list)
    for (f, s), name in sorted(probe.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows = read_jsonl(os.path.join(O if f == "original" else D, name))
        rows = [r for r in rows if r.get("model_id", DREAM) == DREAM and r["id"] in exs]
        for r in rows:
            by_class[f, s, r["class"]].append(r)
            by_class[f, s, "all"].append(r)
    for (f, s, cls), rs in sorted(by_class.items(), key=lambda kv: (kv[0][2] != "all", kv[0][2], kv[0][1], kv[0][0])):
        n = len(rs)
        rows_probe.append({"surplus": s, "interface": f, "class": cls, "n": n,
                           "mean_p_close": statistics.mean(r["p_close"] for r in rs),
                           "median_p_pad": statistics.median(r["p_pad"] for r in rs),
                           "mean_p_content": statistics.mean(r["p_content"] for r in rs),
                           **{f"pick_{k}": sum(r["constrained_kind"] == k for r in rs) / n
                              for k in ("closer", "pad", "content")}})

    # variant closers
    for c in CONDS:
        st = []
        for i, r in sorted(recs["variant", c].items()):
            st += closer_stats(tok, exs[i], r)
        if st:
            ended = [s for s in st if s[0]]
            rows_closers.append({"lengths": cond_label(c), "n_slots": len(st), "ended": len(ended) / len(st),
                                 "bare_closer": sum(s[1] for s in ended) / max(len(ended), 1),
                                 "no_closer": sum(s[2] for s in st) / len(st)})

    # supplementary: every slot read from its own tokens (both interfaces), so that an unclosed slot
    # does not turn the other slots of its item into "unparsed"
    rows_content, rows_content_swap, rows_content_one, rows_bare = [], [], [], []
    content_kinds = [k for k in se.KINDS if k != "unparsed"]
    for c in CONDS:
        ids = common(c)
        for f in IFACES:
            kinds, unclosed = Counter(), Counter()
            for i in sorted(ids):
                r = recs[f, c].get(i)
                if r is None:
                    continue
                lens = slot_lengths_of(r, oracle[i])
                for (ci, p), (v, closed) in slot_contents(tok, exs[i], r, f == "variant").items():
                    k = classify_value(exs[i], ci, p, v, lens[ci, p] + c[1], oracle[i])
                    kinds[k] += 1
                    unclosed[k] += not closed
            tot = sum(kinds.values())
            if tot:
                row = {"lengths": cond_label(c), "interface": f, "n_slots": tot,
                       **{k: kinds[k] / tot for k in content_kinds}}
                if f == "variant":
                    row["unclosed"] = sum(unclosed.values()) / tot
                    row["unclosed_overfill"] = unclosed["overfill"] / tot
                rows_content.append(row)
    for c in [("oracle", 0, 1), ("swap", 0, 1)]:
        for f in IFACES:
            kinds = Counter()
            for i in sorted(swap_ids):
                r = recs[f, c].get(i)
                if r is None:
                    continue
                lens, sw = slot_lengths_of(r, oracle[i]), swapped_lengths(tok, exs[i])
                for (ci, p), (v, closed) in slot_contents(tok, exs[i], r, f == "variant").items():
                    if sw[ci, p] != oracle[i][ci, p]:
                        kinds[classify_value(exs[i], ci, p, v, lens[ci, p] + c[1], oracle[i])] += 1
            tot = sum(kinds.values())
            if tot:
                rows_content_swap.append({"lengths": cond_label(c), "interface": f, "n_slots": tot,
                                          **{k: kinds[k] / tot for k in content_kinds}})
    for f in IFACES:
        kinds, jown = Counter(), 0
        for i in sorted(one_ids):
            r = recs[f, ("onesided", 0, 1)].get(i)
            if r is None:
                continue
            ex, vals = exs[i], slot_contents(tok, exs[i], r, f == "variant")
            for (fname, p), (ii, jj) in onesided_pairs(tok, ex).items():
                v = vals[ii, p][0]
                desc = ex.function(fname)
                if value_ok(desc, p, v, ex.gold_calls[ii][1][p]):
                    k = "own"
                elif value_ok(desc, p, v, ex.gold_calls[jj][1][p]):
                    k = "sibling_fit"
                else:
                    k = classify_value(ex, ii, p, v, oracle[i][jj, p], oracle[i])
                    k = k if k == "overfill" else "other"
                kinds[k] += 1
                jown += value_ok(desc, p, vals[jj, p][0], ex.gold_calls[jj][1][p])
        tot = sum(kinds.values())
        if tot:
            rows_content_one.append({"interface": f, "n_slots": tot, **{k: kinds[k] / tot for k in ONESIDED},
                                     "jstar_own": jown / tot})
    for c in CONDS:
        verdicts = Counter()
        for i, r in sorted(recs["variant", c].items()):
            if r.get("trace"):
                verdicts.update(bare_vs_content(tok, exs[i], r))
        tot = sum(verdicts.values())
        if tot:
            rows_bare.append({"lengths": cond_label(c), "longer_closers": tot,
                              **{k: verdicts[k] / tot for k in ("bare", "content", "top5_closers")}})

    print("# Experiment C3: closer in the slot vs closer in the skeleton (Dream, one canvas)\n")
    print("Original: the skeleton writes each value's closer after its slot, the model ends a value early "
          "with its own closer and padding. Variant: no closer in the skeleton, one more position per slot, "
          "the model writes the closer and the rest of the slot becomes spaces. exact / +s: slot = value "
          "length + s (+1 for the variant's closer); swap / onesided: on the 165 items they change. Greedy, "
          "confidence order. Every row compares the two interfaces on the items both have.\n")
    print("## Item level\n")
    icols = ["lengths"] + [f"{k}_{f}" for k in ("n", "set_acc", "ccer", "syntax", "overfill") for f in IFACES]
    print(table(rows_items, icols))
    print("\n## Slot level (all slots; scripts/slot_errors.py classes)\n")
    scols = ["lengths", "interface", "n_items", "n_slots"] + se.KINDS
    print(table(rows_slots, scols))
    print(f"\n## Swapped slots (the slots the swap changes, {len(swap_ids)} items; the paper's Table 3 "
          "for the original)\n")
    print(table(rows_swap, ["lengths", "interface", "slots", "n_slots"] + se.KINDS))
    print(f"\n## One-sided lengthening ({len(one_ids)} items, k=1): the lengthened slot (i*, p)\n")
    print("own: its own value, ended early; overfill: its own value and more; sibling_fit: the value of j* "
          "(the longest sibling, whose slot stays exact), i.e. two calls with the same value; other = "
          "other_sibling + truncated + unparsed + wrong. jstar_own: j*'s slot holds j*'s own value.\n")
    print(table(rows_one, ["interface", "n_items", "n_slots"] + ONESIDED + OTHER_SUB + ["jstar_own"]))
    print("\n## Teacher-forced probe: the first position after the gold value\n")
    print("Original surplus s: s masks, then the skeleton's closer. Variant surplus s: 1 + s positions, then "
          "the next fixed text (s = 0: the one position must be the closer). P(closer): tokens that start "
          "with the slot's closer; pick: what the skeleton constraint would commit there.\n")
    pcols = ["surplus", "interface", "class", "n", "mean_p_close", "median_p_pad", "mean_p_content",
             "pick_closer", "pick_pad", "pick_content"]
    print(table([r for r in rows_probe if r["class"] == "all"], pcols))
    print("\nBy slot class:\n")
    print(table([r for r in rows_probe if r["class"] != "all"], pcols))
    print("\n## Variant closers\n")
    print("ended: the slot's value has a closer; bare_closer: of those, the closing token is just the closer "
          "('\"', ',' or '}'; for arrays / dicts the closer ends its token) rather than e.g. '\",' or '}}' (the "
          "extra text stays on the canvas, decoding keeps the closer only); no_closer: the slot was filled "
          "to the end without one.\n")
    print(table(rows_closers, ["lengths", "n_slots", "ended", "bare_closer", "no_closer"]))
    print("\nWhat a bare-closer-only constraint would have done where the variant closed a number / boolean / "
          "string slot with a longer token: the committing step's top-5 candidates rank a bare closer above "
          "every content token (bare), a content token above every bare closer (content: the restriction "
          "would likely have made it write more content instead of closing), or only longer closers (top5_closers).\n")
    print(table(rows_bare, ["lengths", "longer_closers", "bare", "content", "top5_closers"]))

    print("\n## Supplementary: slots read one by one\n")
    print("The tables above use the parsed output, as the paper does: in the variant one slot filled without a "
          "closer makes the whole output unparseable, and slot_errors.py then counts every slot of that item as "
          "unparsed. Here each slot's value is read from its own tokens (the text before its closer, or the whole "
          "slot if it has none) in both interfaces and classified the same way. unclosed: variant slots without "
          "a closer (they make the output fail to parse); unclosed_overfill: of all slots, the unclosed ones whose "
          "content is the own value followed by more.\n")
    print(table(rows_content, ["lengths", "interface", "n_slots"] + content_kinds + ["unclosed", "unclosed_overfill"]))
    print("\nSwapped slots, read one by one:\n")
    print(table(rows_content_swap, ["lengths", "interface", "n_slots"] + content_kinds))
    print("\nOne-sided lengthening, the lengthened slot read on its own (own / overfill / sibling_fit = j*'s "
          "value / other), and j*'s slot holding j*'s value:\n")
    print(table(rows_content_one, ["interface", "n_slots"] + ONESIDED + ["jstar_own"]))

    print("\n## Canvas samples (variant, k=1)\n")
    print("⟦ ⟧ mark the slots, ␣ a space token inside a slot, ∅ padding. The same item in the original "
          "interface below each.\n")
    shown = 0
    for c in [("oracle", 0, 1), ("oracle", 1, 1)]:
        cands = [(i, r) for i, r in sorted(recs["variant", c].items(), key=lambda kv: list(exs).index(kv[0]))
                 if i in recs["original", c]]
        # prefer an output that parses and, at +1, has every value closed early; else the first one
        good = [(i, r) for i, r in cands if r["syntax_ok"] and (c[1] == 0 or all(s[0] for s in closer_stats(tok, exs[i], r)))]
        for i, r in (good or cands)[:1]:
            orig = recs["original", c][i]
            print(f"### {i}, {cond_label(c)}: variant {'correct' if r['diagnosis']['correct'] else r['diagnosis']['labels']}, "
                  f"original {'correct' if orig['diagnosis']['correct'] else orig['diagnosis']['labels']}\n")
            print("```text")
            print("variant:  " + render(tok, exs[i], r, True))
            print("decoded:  " + r["text"])
            print("original: " + render(tok, exs[i], orig, False))
            print("decoded:  " + orig["text"])
            print("```\n")
            shown += 1
    if not shown:
        print("TBD (no variant records yet)\n")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        rows = ([{"table": "items", **r} for r in rows_items] + [{"table": "slots", **r} for r in rows_slots]
                + [{"table": "swap_slots", **r} for r in rows_swap] + [{"table": "onesided", **r} for r in rows_one]
                + [{"table": "probe", **r} for r in rows_probe] + [{"table": "variant_closers", **r} for r in rows_closers]
                + [{"table": "bare_vs_content", **r} for r in rows_bare]
                + [{"table": "slots_read_alone", **r} for r in rows_content]
                + [{"table": "swap_slots_read_alone", **r} for r in rows_content_swap]
                + [{"table": "onesided_read_alone", **r} for r in rows_content_one])
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
