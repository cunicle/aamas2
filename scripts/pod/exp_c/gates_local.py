"""Local checks of experiment C's gates on results pulled from the pod.

  python gates_local.py gate2   <gate_dir> <release results dir>
  python gates_local.py smoke   <gate_dir>
  python gates_local.py closer  <gate_dir> <release results dir>
  python gates_local.py sanity  <results dir>
"""
import json
import os
import sys
from collections import defaultdict

REPO = r"C:\my-code\research\aamas2"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gate2(gdir, rel):
    new = read(os.path.join(gdir, "default5.jsonl"))
    ref = {(r["id"], r["cfg_tag"]): r for r in read(os.path.join(rel, "dream", "bfcl_skel_k.jsonl"))}
    ok = 0
    for r in new:
        o = ref[(r["id"], r["cfg_tag"])]
        same_text, same_ids = r["text"] == o["text"], r["gen_ids"] == o["gen_ids"]
        ok += same_text and same_ids
        print(f"{r['id']:22s} {r['cfg_tag']:34s} text {'SAME' if same_text else 'DIFF'}  gen_ids {'SAME' if same_ids else 'DIFF'}"
              f"  closer_in_slot={'closer_in_slot' in r}")
        if not same_text:
            print("   new:", r["text"]); print("   old:", o["text"])
    print(f"gate 2: {ok}/{len(new)} identical (text and gen_ids)")


def smoke(gdir):
    from transformers import AutoTokenizer
    from ptcdiag.data import load_examples
    from ptcdiag.decoding.constraints import oracle_lengths
    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    for name in ("qwen_sym", "dream_sym", "qwen_swap", "dream_swap", "dream_swap_swap"):
        path = os.path.join(gdir, name + ".jsonl")
        if not os.path.exists(path):
            print("MISSING", path); continue
        print(f"\n#################### {name}")
        for r in read(path):
            if "error" in r:
                print("ERROR", r["id"], r["protocol"], r["error"]); continue
            ex = exs[r["id"]]
            L = oracle_lengths(tok, ex)
            print(f"\n===== {r['id']} n={r['n']} {r['protocol']} lengths={r['length_mode']} {r['decoding']} "
                  f"team: correct={r['team']['diagnosis']['correct']} labels={r['team']['diagnosis']['labels']}")
            print("  gold:", json.dumps(ex.ground_truth))
            for a in r["agents"]:
                orc = [[0, p, L[a["i"] - 1, p]] for (ci, p) in sorted(L) if ci == a["i"] - 1]
                changed = "" if a["lengths"] == orc else f"  (oracle {orc})"
                proto = a["user_text"].split("\n\n", 1)[1] if "\n\n" in a["user_text"] else a["user_text"]
                print(f"  -- agent {a['i']} slots {a['lengths']}{changed}")
                print(f"     protocol text: {proto}")
                print(f"     output: {a['text']}")
            first = r["agents"][0]["user_text"]
            print("  request:", first.split("\n\n", 1)[0][:200])


def closer(gdir, rel):
    from transformers import AutoTokenizer
    from ptcdiag.data import load_examples
    from ptcdiag.decoding.constraints import build_skeleton, token_texts, STRING_TYPES
    from ptcdiag.prompting import parse_tool_calls
    import closer_analysis as ca
    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    texts = token_texts(tok)
    (space,) = tok(" ", add_special_tokens=False)["input_ids"]
    recs = read(os.path.join(gdir, "closer.jsonl"))
    ref = {(r["id"], r["cfg_tag"]): r for r in read(os.path.join(rel, "dream", "bfcl_skel_k.jsonl"))}
    by_s = defaultdict(list)
    for r in recs:
        by_s[r["surplus"]].append(r)
    for s, rs in sorted(by_s.items()):
        print(f"\n######## variant surplus {s}: {len(rs)} records")
        for r in rs[:5]:
            ex = exs[r["id"]]
            ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, s, None, closer_in_slot=True)
            # no closer written by the skeleton right after any slot
            pre = []
            for sl in slots:
                nxt = texts[ids[sl.positions[-1] + 1]]
                bad = nxt.startswith('"') if sl.type in STRING_TYPES else nxt.startswith((",", "}}"))
                pre.append(bad)
            forced = [p for st in r["trace"]["steps"] for p in st["forced"]]
            fill_ok = all(r["gen_ids"][p - r["trace"]["gen_start"]] == space for p in forced)
            print(f"\n{r['id']} s={s}: closer_in_slot={r.get('closer_in_slot')} skeleton closer after a slot: {any(pre)}; "
                  f"forced {len(forced)} positions, all spaces: {fill_ok}; syntax_ok {r['syntax_ok']}; "
                  f"correct {r['diagnosis']['correct']} {r['diagnosis']['labels']}")
            print("  canvas :", ca.render(tok, ex, r, True))
            print("  decoded:", r["text"])
            print("  parses :", parse_tool_calls(r["text"]).syntax_ok)
    rs = by_s.get(0, [])
    if rs:
        var = sum(r["diagnosis"]["correct"] for r in rs) / len(rs)
        orig = [ref[(r["id"], "confidence_k1_tnone_bfull_T0.0")] for r in rs]
        o = sum(x["diagnosis"]["correct"] for x in orig) / len(orig)
        print(f"\nvariant exact k=1 set_acc on {len(rs)} items: {var:.3f}; original k=1 on the same items: {o:.3f}; "
              f"difference {100 * (o - var):.1f} points (gate: stop if > 20)")
        for r, x in zip(rs, orig):
            if r["diagnosis"]["correct"] != x["diagnosis"]["correct"]:
                print(f"  {r['id']}: variant {r['diagnosis']['labels']} {r['text']}\n  {' ' * len(r['id'])}  original {x['diagnosis']['labels']} {x['text']}")


def sanity(rdir):
    for m in ("qwen", "dream"):
        path = os.path.join(rdir, m, "agents_c1.jsonl")
        if not os.path.exists(path):
            print("MISSING", path); continue
        teams = [r for r in read(path) if r["protocol"] == "sim-anon" and "error" not in r]
        bad, groups = [], 0
        from ptcdiag.data import load_examples
        exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}
        for r in teams:
            ex = exs[r["id"]]
            by_f = defaultdict(list)
            for a in r["agents"]:
                by_f[ex.gold_calls[a["i"] - 1][0]].append(a)
            for f, ags in by_f.items():
                if len(ags) < 2:
                    continue
                groups += 1
                if len({a["text"] for a in ags}) != 1 or len({a["user_text"] for a in ags}) != 1:
                    bad.append((r["id"], f, [a["text"] for a in ags]))
        print(f"{m}: {len(teams)} sim-anon teams, {groups} sibling groups, {len(bad)} with differing outputs")
        for b in bad[:10]:
            print("   ", b)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"gate2": gate2, "smoke": smoke, "closer": closer, "sanity": sanity}[cmd](*rest)
