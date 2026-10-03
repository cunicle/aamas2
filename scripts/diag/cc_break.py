"""Diagnostic: cross-call errors split by label, and entity parameters vs defaulted ones."""
import json
import sys
from collections import Counter, defaultdict
sys.path.insert(0, "/workspace/aamas2")
from ptcdiag.eval.taxonomy import CROSS_CALL

ENTITY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
rs = [json.loads(l) for l in open(sys.argv[1])]
by_k = defaultdict(list)
for r in rs:
    by_k[r["cfg"]["k"]].append(r)
labels = ["chimera_value", "duplicate_call", "cross_binding", "omitted_call", "extra_call", "inconsistent_shared_arg"]
print("| k | n | any cross-call | " + (" entity params | other params | " if ENTITY else "") + " | ".join(labels) + " |")
for k in sorted(by_k):
    xs = by_k[k]
    n = len(xs)
    cc = lambda r, pred: any(d["label"] in CROSS_CALL and pred(d) for d in r["diagnosis"]["details"])  # noqa: E731
    row = [str(k), str(n), f"{sum(cc(r, lambda d: True) for r in xs) / n:.3f}"]
    if ENTITY:
        ent = lambda d: d.get("param") in ENTITY or (d["label"] in ("duplicate_call", "omitted_call", "extra_call"))  # noqa: E731
        row += [f"{sum(cc(r, ent) for r in xs) / n:.3f}", f"{sum(cc(r, lambda d: not ent(d)) for r in xs) / n:.3f}"]
    c = Counter(d["label"] for r in xs for d in r["diagnosis"]["details"])
    row += [str(c[l]) for l in labels]
    print("| " + " | ".join(row) + " |")
