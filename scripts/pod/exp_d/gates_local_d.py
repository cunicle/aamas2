"""Local checks of experiment D's gates on results pulled from the pod (EXP_D_PROMPT.md section 3).

  python gates_local_d.py gate2     <gate_dir> <release results dir>   # default path = the 10-03 records
  python gates_local_d.py tolerant  <gate_dir> <release results dir>   # filler in +1 slots; exact vs original
  python gates_local_d.py pos       <gate_dir>                         # position agents' outputs
  python gates_local_d.py sanity    <results dir> <release results dir> # Dream pos-anon = canvas k=16 cities
  python gates_local_d.py speed     <gate_dir>                         # seconds per record / agent
"""
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DREAM = "Dream-org/Dream-v0-Instruct-7B"
K1, K16 = "confidence_k1_tnone_bfull_T0.0", "confidence_k16_tnone_bfull_T0.0"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def bfcl_examples():
    from ptcdiag.data import load_examples
    return {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}


def gate2(gdir, rel):
    new = read(os.path.join(gdir, "default5.jsonl"))
    ref = {(r["id"], r["cfg_tag"]): r for r in read(os.path.join(rel, "dream", "bfcl_skel_k.jsonl"))}
    ok = 0
    for r in new:
        o = ref[(r["id"], r["cfg_tag"])]
        same_text, same_ids = r["text"] == o["text"], r["gen_ids"] == o["gen_ids"]
        ok += same_text
        print(f"{r['id']:22s} {r['cfg_tag']:34s} text {'SAME' if same_text else 'DIFF'}  gen_ids "
              f"{'SAME' if same_ids else 'DIFF'}  tolerant={'tolerant' in r}  active_call={'active_call' in r}")
        if not same_text:
            print("   new:", r["text"]); print("   old:", o["text"])
    print(f"gate 2: {ok}/{len(new)} texts identical to the 10-03 records")


def slot_values(tok, texts, ex, r):
    """[(slot, raw token texts of the slot, raw value text, normalized value)] of a canvas record,
    the value read as the format-tolerant decoder reads it (up to the closer, padding dropped)."""
    from ptcdiag.decoding.constraints import (build_skeleton, lengths_from_list, normalize_value, slot_class,
                                              slot_closers, value_end)
    lengths = lengths_from_list(r["lengths"]) if r.get("lengths") else None
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, r.get("surplus", 0), lengths)
    gen, special = r["gen_ids"], set(tok.all_special_ids)
    out = []
    for s in slots:
        raw = [gen[g] for g in s.positions]
        shown = ["<mask>" if t == tok.mask_token_id else "<pad>" if t in special else texts[t] for t in raw]
        tt = ["" if t in special else texts[t] for t in raw]
        end = value_end(tt, slot_closers(s.type), slot_class(s.type) == "generic")
        upto = raw[:end[0]] if end is not None else raw
        value = tok.decode([t for t in upto if t not in special])
        norm = normalize_value(value, s.type) if slot_class(s.type) != "generic" else value
        out.append((s, shown, value, norm))
    return out


def tolerant(gdir, rel):
    from transformers import AutoTokenizer
    from ptcdiag.decoding.constraints import token_texts
    exs = bfcl_examples()
    tok = AutoTokenizer.from_pretrained(DREAM, trust_remote_code=True)
    texts = token_texts(tok)
    recs = read(os.path.join(gdir, "tolerant20.jsonl"))
    ref = {(r["id"], r["cfg_tag"]): r for r in read(os.path.join(rel, "dream", "bfcl_skel_k.jsonl"))}
    sur = {(r["id"], r["cfg_tag"], r["surplus"]): r for r in read(os.path.join(rel, "dream", "bfcl_surplus.jsonl"))}
    by_s = defaultdict(list)
    for r in recs:
        by_s[r.get("surplus", 0)].append(r)
    for s, rs in sorted(by_s.items()):
        n_slots = n_fill = n_ws = n_text_ok = 0
        kinds, shown = Counter(), 0
        print(f"\n######## tolerant, surplus {s}: {len(rs)} records, tolerant flag {Counter(r.get('tolerant') for r in rs)}")
        for r in rs:
            ex = exs[r["id"]]
            vals = slot_values(tok, texts, ex, r)
            # filler: the normalization removed more than whitespace around the value
            filled = [(sl, sh, v, nv) for sl, sh, v, nv in vals if v.strip() != nv.strip()]
            n_slots += len(vals)
            n_fill += len(filled)
            n_ws += sum(v != nv and v.strip() == nv.strip() for _, _, v, nv in vals)
            calls = r["calls"] if r["syntax_ok"] else []
            for sl, sh, v, nv in filled:
                st, sn = v.strip(), nv.strip()
                kinds[repr(st[len(sn):]) if st.startswith(sn) else "changed: " + repr(v)] += 1
                # the decoded text holds the normalized value, not the raw one
                got = calls[sl.call]["arguments"].get(sl.param) if len(calls) > sl.call else "<no call>"
                try:
                    want = json.loads('"' + nv + '"') if sl.type in ("string", "any") else json.loads(nv)
                except ValueError:
                    want = "<unparsable>"
                n_text_ok += got == want
            if s == 1 and filled and shown < 3:
                shown += 1
                print(f"\n{r['id']}  correct={r['diagnosis']['correct']} labels={r['diagnosis']['labels']}")
                for sl, sh, v, nv in vals:
                    mark = "   <- filler removed" if v.strip() != nv.strip() else ""
                    print(f"  call {sl.call} {sl.param} ({sl.type}): slot {' | '.join(map(repr, sh))}\n"
                          f"      raw value {v!r} -> decoded {nv!r}{mark}")
                print("  decoded text:", r["text"])
                o = sur.get((r["id"], r["cfg_tag"], 1))
                if o:
                    print("  original interface, +1:", o["text"], f"(correct={o['diagnosis']['correct']})")
        print(f"\nsurplus {s}: {n_fill}/{n_slots} slots had filler or format tokens removed "
              f"(the parsed value in the decoded text is the normalized one in {n_text_ok} of them); "
              f"{n_ws} more slots lost only whitespace around the value")
        for k, v in kinds.most_common(15):
            print(f"   {v:4d}  removed {k}")
        acc = sum(r["diagnosis"]["correct"] for r in rs) / len(rs)
        if s == 0:
            orig = [ref[(r["id"], K1)] for r in rs]
        else:
            orig = [sur[(r["id"], K1, s)] for r in rs if (r["id"], K1, s) in sur]
        o = sum(x["diagnosis"]["correct"] for x in orig) / len(orig)
        print(f"set_acc tolerant k=1 surplus {s} on {len(rs)} items: {acc:.3f}; original interface on the same "
              f"items: {o:.3f} ({len(orig)} items); difference {100 * (o - acc):+.1f} points"
              + ("  (gate: stop if > 10)" if s == 0 else ""))
        if s == 0:
            for r, x in zip(rs, orig):
                if r["diagnosis"]["correct"] != x["diagnosis"]["correct"]:
                    print(f"  {r['id']}: tolerant {r['diagnosis']['labels']} {r['text']}\n"
                          f"  {' ' * len(r['id'])}  original {x['diagnosis']['labels']} {x['text']}")


def is_placeholder(v):
    return v is None or v == "..."


def check_agents(name, recs, exs):
    from ptcdiag.prompting import parse_tool_calls
    n_ag = bad_other = bad_own = 0
    for r in recs:
        if "error" in r:
            print("ERROR", r["id"], r["error"]); continue
        print(f"\n===== {name} {r['id']} n={r['n']} {r['protocol']} {r.get('decoding')}")
        if "team" in r:
            print(f"  team: correct={r['team']['diagnosis']['correct']} labels={r['team']['diagnosis']['labels']}")
            print("  gold:", json.dumps(exs[r["id"]].ground_truth))
        else:
            print(f"  cities={r['cities']}")
        for a in r["agents"]:
            n_ag += 1
            calls = parse_tool_calls(a["text"]).calls if a["syntax_ok"] else []
            own_ok = other_ok = False
            if len(calls) == r["n"]:
                own = calls[a["i"] - 1]["arguments"]
                own_ok = bool(own) and not any(is_placeholder(v) for v in own.values())
                other_ok = all(is_placeholder(v) for j, c in enumerate(calls) if j != a["i"] - 1
                               for v in c["arguments"].values())
            bad_own += not own_ok
            bad_other += not other_ok
            print(f"  -- agent {a['i']}: own call has values {own_ok}; other calls placeholders {other_ok}\n"
                  f"     {a['text']}")
        print("  user text of agent 1:", r["agents"][0]["user_text"])
    print(f"\n{name}: {n_ag} agents; own call without values: {bad_own}; other calls not placeholders: {bad_other}")


def pos(gdir):
    exs = bfcl_examples()
    for name in ("dream_pos_bfcl", "qwen_pos_bfcl", "dream_pos_choose", "qwen_pos_choose"):
        path = os.path.join(gdir, name + ".jsonl")
        if not os.path.exists(path):
            print("MISSING", path); continue
        print(f"\n#################### {name}")
        check_agents(name, read(path), exs)


def sanity(rdir, rel):
    canvas = {r["id"]: r for r in read(os.path.join(rel, "dream", "choose.jsonl")) if r["cfg_tag"] == K16}
    teams = [r for r in read(os.path.join(rdir, "dream", "agents_pos.jsonl")) if r["protocol"] == "pos-anon"]
    diff, err = [], [r for r in teams if "error" in r]
    for r in teams:
        if "error" in r:
            continue
        c = canvas[r["id"]]
        cities = [x["arguments"].get("city") for x in c["calls"]] if c["syntax_ok"] else None
        if cities != r["cities"]:
            diff.append((r["id"], r["variant"], r["cities"], cities))
    print(f"gate 4: {len(teams)} Dream pos-anon teams ({len(err)} errors), canvas k=16 records {len(canvas)}; "
          f"{len(teams) - len(err) - len(diff)} identical, {len(diff)} different")
    for d in diff[:20]:
        print("   ", d)


def speed(gdir):
    for name in ("default5", "tolerant20"):
        path = os.path.join(gdir, name + ".jsonl")
        if os.path.exists(path):
            by = defaultdict(list)
            for r in read(path):
                by[(r["cfg_tag"], r.get("surplus", 0))].append(r["seconds"])
            for k, v in sorted(by.items()):
                print(f"{name} {k}: {len(v)} records, {sum(v) / len(v):.2f} s per record (decode only)")
    for name in ("dream_pos_bfcl", "qwen_pos_bfcl", "dream_pos_choose", "qwen_pos_choose"):
        path = os.path.join(gdir, name + ".jsonl")
        if os.path.exists(path):
            rs = [r for r in read(path) if "error" not in r]
            na = sum(len(r["agents"]) for r in rs)
            print(f"{name}: {len(rs)} teams, {na} agents, {sum(r['seconds'] for r in rs) / max(na, 1):.2f} s per agent")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"gate2": gate2, "tolerant": tolerant, "pos": pos, "sanity": sanity, "speed": speed}[cmd](*rest)
