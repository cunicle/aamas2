"""P0 checks on existing skeleton outputs (CPU only).

Per model / surplus / config:
  set_acc        strict (as reported)
  lenient_acc    a predicted value that starts with an acceptable gold value of its matched
                 call counts as that gold value (forgives text appended after a correct value)
  slot_acc       share of gold slots filled correctly (syntax-ok items; a slot is wrong if a
                 detail names it, or its call is omitted / duplicated)
  closed         share of slots in which the model wrote a closing token
  closed_at_L    share of slots whose closer sits right after the gold value's length
Error split (surplus > 0, error details on value slots): overfill (own gold + more) /
sibling (another call's gold for the same parameter, or a duplicated call) / other;
syntax-error items counted separately.

  python p0_check.py <model_id> <results.jsonl> [...]
"""
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/workspace/aamas2")
from transformers import AutoTokenizer

from ptcdiag.data import load_examples
from ptcdiag.decoding.constraints import SkeletonConstraint, value_text
from ptcdiag.eval.taxonomy import diagnose
from ptcdiag.prompting import parse_tool_calls


class Stub:  # what SkeletonConstraint needs from an adapter, without the model
    def __init__(self, tok, mask_id, pad_id, eos_ids):
        self.tokenizer, self.mask_id, self.pad_id, self.eos_ids = tok, mask_id, pad_id, eos_ids

    @property
    def special_ids(self):
        return set(self.eos_ids) | {self.pad_id, self.mask_id}


model_id = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if "llada2" in model_id.lower():
    stub = Stub(tok, 156895, 156892, [156892, tok.convert_tokens_to_ids("<|role_end|>")])
else:
    stub = Stub(tok, tok.mask_token_id, tok.eos_token_id, [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")])
exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple", "/workspace/aamas2/data/bfcl")}


def norm(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def acc_list(ex, gi, p):
    return [a for a in ex.gold_calls[gi][1].get(p, []) if a != ""]


def lenient_correct(ex, r):
    if not r["syntax_ok"]:
        return False
    m = dict((pi, gi) for pi, gi in r["diagnosis"]["matching"])
    calls = json.loads(json.dumps(r["calls"]))
    for pi, c in enumerate(calls):
        gi = m.get(pi)
        if gi is None:
            continue
        for p, v in list(c["arguments"].items()):
            for a in acc_list(ex, gi, p):
                if norm(v) != norm(a) and norm(v).startswith(norm(a)):
                    c["arguments"][p] = a
                    break
    return diagnose(ex, parse_tool_calls(json.dumps(calls))).correct


def slot_stats(ex, r):
    n = sum(len(acc_list(ex, gi, p)) > 0 for gi, (_, ps) in enumerate(ex.gold_calls) for p in ps)
    if not r["syntax_ok"]:
        return n, 0
    bad = set()
    m = dict((pi, gi) for pi, gi in r["diagnosis"]["matching"])
    for d in r["diagnosis"]["details"]:
        if "gold" in d and "param" in d:
            bad.add((d["gold"], d["param"]))
        elif d["label"] == "omitted_call":
            bad |= {(d["gold"], p) for p in ex.gold_calls[d["gold"]][1]}
        elif d["label"] == "duplicate_call":
            for pi in d.get("surplus", []):
                gi = m.get(pi)
                if gi is not None:
                    bad |= {(gi, p) for p in ex.gold_calls[gi][1]}
    bad = {(g, p) for g, p in bad if acc_list(ex, g, p)}
    return n, n - len(bad)


def classify(ex, d):
    if d["label"] == "duplicate_call":
        return "sibling"
    if "gold" not in d or "param" not in d:
        return "other"
    v = norm(d.get("value", ""))
    if any(v != norm(a) and v.startswith(norm(a)) for a in acc_list(ex, d["gold"], d["param"])):
        return "overfill"
    for gj in range(len(ex.gold_calls)):
        if gj != d["gold"] and any(v == norm(a) or v.startswith(norm(a)) for a in acc_list(ex, gj, d["param"])):
            return "sibling"
    return "other"


def closing(ex, r):
    s = r.get("surplus", 0)
    if not s:
        return 0, 0, 0
    c = SkeletonConstraint(stub, ex, surplus=s)
    gen = r["gen_ids"]
    n = closed = at_L = 0
    for sl in c.slots:
        end = c._end(gen, sl)
        n += 1
        if end is not None:
            closed += 1
            at_L += end[0] == len(sl.positions) - s and end[1] == 0
    return n, closed, at_L


groups = defaultdict(list)
for path in sys.argv[2:]:
    for line in open(path):
        r = json.loads(line)
        if r.get("mode") == "skeleton" and "diagnosis" in r and r["id"] in exs:
            groups[(r.get("surplus", 0), r["cfg_tag"])].append(r)

print(f"## {model_id}")
print("| surplus | cfg | n | set_acc | lenient_acc | slot_acc | closed | closed_at_L | overfill / sibling / other (of error slots) | syntax-error items |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (s, tag), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if not any(k in tag for k in ("_k1_", "_k4_", "_k16_")):
        continue
    n = len(rs)
    sa = sum(r["diagnosis"]["correct"] for r in rs) / n
    la = sum(lenient_correct(exs[r["id"]], r) for r in rs) / n
    tot = ok = 0
    for r in rs:
        a, b = slot_stats(exs[r["id"]], r)
        tot += a
        ok += b
    cl = [closing(exs[r["id"]], r) for r in rs]
    ns, nc, nl = (sum(x[i] for x in cl) for i in range(3))
    kinds = Counter(classify(exs[r["id"]], d) for r in rs if r["syntax_ok"] for d in r["diagnosis"]["details"]
                    if d["label"] not in ("inconsistent_shared_arg",))
    kt = sum(kinds.values()) or 1
    syn = sum(not r["syntax_ok"] for r in rs) / n
    split = " / ".join(f"{kinds[k] / kt:.2f}" for k in ("overfill", "sibling", "other")) + f" (n={sum(kinds.values())})"
    print(f"| {s} | {tag} | {n} | {sa:.3f} | {la:.3f} | {ok / max(tot, 1):.3f} | "
          f"{(nc / ns) if ns else float('nan'):.3f} | {(nl / ns) if ns else float('nan'):.3f} | {split} | {syn:.3f} |")
