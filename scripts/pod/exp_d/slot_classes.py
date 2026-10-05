"""Experiment D report helper (not the paper's analysis; python slot_classes.py <dir of swap_tolerant.py> <new results> <10-03 results>): per-slot classes (scripts/closer_analysis.py's
classify_value) of every slot under +s at k=1, tolerant (normalized as decoded) next to the original
(slot tokens as read by swap_direction.py), and what the tolerant normalization removed."""
import json, os, re, sys
from collections import Counter
S = sys.argv[1]; F, R = sys.argv[2], sys.argv[3]
sys.path.insert(0, S); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import swap_tolerant as st
from closer_analysis import classify_value, slot_contents
from transformers import AutoTokenizer
from ptcdiag.data import load_examples
from ptcdiag.decoding.constraints import (build_skeleton, normalize_value, oracle_lengths, slot_class, slot_closers,
                                          token_texts, value_end)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True); tt = token_texts(tok)
exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple")}
K1 = "confidence_k1_tnone_bfull_T0.0"
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
orig = {(r["id"], r["surplus"]): r for r in rd(f"{R}/dream/bfcl_surplus.jsonl") if r["cfg_tag"] == K1}
tol = [r for r in rd(f"{F}/dream/bfcl_tolerant.jsonl") if r["cfg_tag"] == K1 and r["surplus"] in (1, 2, 8)]
KINDS = ["own", "overfill", "truncated", "sibling_fit", "sibling", "other"]
cls = {}
removed = {1: Counter(), 2: Counter(), 8: Counter()}
nslots = Counter()
special = set(tok.all_special_ids)
def kind_of(raw, norm):
    rest = raw.strip()[len(norm.strip()):] if raw.strip().startswith(norm.strip()) else None
    if rest is None: return "other change"
    if rest.startswith("_"): return "digit group '_000'"
    if rest.replace(chr(92) + "n", "").strip() == "": return "escaped newline"
    if re.fullmatch(r"\.0*", rest): return "decimal point (+ zeros)"
    if re.fullmatch(r"[A-Za-z]+", rest): return "letter suffix (i, f, L, y, ...)"
    return "other suffix"
for r in tol:
    s = r["surplus"]; ex = exs[r["id"]]; oracle = oracle_lengths(tok, ex)
    o = orig[r["id"], s]
    tv = st.tolerant_contents(tok, ex, r)
    ov = {k: v for k, (v, _) in slot_contents(tok, ex, o, False).items()}
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, None, s, None)
    gen = r["gen_ids"]
    for sl in slots:
        L = oracle[sl.call, sl.param] + s
        for iface, vals in (("tolerant", tv), ("original", ov)):
            cls.setdefault((s, iface), Counter())[classify_value(ex, sl.call, sl.param, vals[sl.call, sl.param], L, oracle)] += 1
        if slot_class(sl.type) != "generic":
            nslots[s] += 1
            stx = ["" if gen[g] in special else tt[gen[g]] for g in sl.positions]
            end = value_end(stx, slot_closers(sl.type), False)
            upto = sl.positions[:end[0]] if end is not None else sl.positions
            raw = tok.decode([gen[g] for g in upto if gen[g] not in special]); norm = normalize_value(raw, sl.type)
            if raw.strip() != norm.strip():
                removed[s][kind_of(raw, norm)] += 1
print("Slot classes of every slot, Dream k=1 (% of slots):\n")
print("| surplus | interface | n slots | " + " | ".join(KINDS) + " |"); print("|---|---|---|" + "---|" * len(KINDS))
for (s, iface), c in sorted(cls.items(), key=lambda x: (x[0][0], x[0][1] != "original")):
    n = sum(c.values())
    print(f"| +{s} | {iface} | {n} | " + " | ".join(f"{100 * c[k] / n:.1f}" for k in KINDS) + " |")
print("\nWhat the tolerant normalization removed (string / number / boolean slots, k=1):\n")
print("| surplus | slots | with filler removed | " + " | ".join(sorted({k for c in removed.values() for k in c})) + " |")
keys = sorted({k for c in removed.values() for k in c})
print("|---|---|---|" + "---|" * len(keys))
for s in (1, 2, 8):
    print(f"| +{s} | {nslots[s]} | {sum(removed[s].values())} ({100 * sum(removed[s].values()) / nslots[s]:.1f}%) | " + " | ".join(str(removed[s][k]) for k in keys) + " |")
