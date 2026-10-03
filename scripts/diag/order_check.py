"""Do the models fill symmetric same-function slots in mention order?

Items: BFCL `parallel` (one function, every call has the same parameters) whose slots
are length-symmetric for this tokenizer, so the canvas does not tell the calls apart.
Gold calls in BFCL are listed in mention order. For syntax-ok records with all calls
matched, reports the share whose optimal matching is the identity (pred call i = gold
call i), i.e. the model wrote the i-th mentioned entity into the i-th slot.
"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "/workspace/aamas2")
from transformers import AutoTokenizer
from ptcdiag.data import load_examples
from ptcdiag.decoding.constraints import length_symmetric
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
exs = {e.id: e for e in load_examples("bfcl:parallel", "/workspace/aamas2/data/bfcl")}
sym = {i: all(length_symmetric(tok, e).values()) for i, e in exs.items()}
g = defaultdict(lambda: [0, 0, 0])
for path in sys.argv[2:]:
    for l in open(path):
        r = json.loads(l)
        if r["id"] not in exs or not sym[r["id"]] or not r.get("syntax_ok") or r.get("surplus", 0):
            continue
        m = r["diagnosis"]["matching"]
        n = len(exs[r["id"]].gold_calls)
        if len(m) != n:
            continue
        x = g[r["cfg_tag"]]
        x[0] += 1
        x[1] += all(p == q for p, q in m)
        x[2] += r["diagnosis"]["correct"]
print("| cfg | symmetric parallel items | identity order | correct |")
for tag, (n, ident, ok) in sorted(g.items()):
    print(f"| {tag} | {n} | {ident / n:.3f} | {ok / n:.3f} |")
