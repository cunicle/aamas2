"""Diagnostic: accuracy and cross-call errors on length-symmetric items / parameters.

With per-value slot lengths, siblings of different lengths can be told apart by length;
cross-call errors are only informative where every call's slot for the parameter has the
same length. Reports, per file and config: all items, fully symmetric items, and the
share of cross-call error details that involve a symmetric parameter.

  python sym_check.py <model_id> <results.jsonl> [...]
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/workspace/aamas2")
from transformers import AutoTokenizer

from ptcdiag.data import load_examples
from ptcdiag.decoding.constraints import length_symmetric
from ptcdiag.eval.taxonomy import CROSS_CALL, is_cross_call

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
exs = {e.id: e for s in ("bfcl:parallel", "bfcl:parallel_multiple") for e in load_examples(s, "data/bfcl")}
exs.update({e.id: e for e in load_examples("probe:data/paraprobe.jsonl", "data/bfcl")})
_sym = {}


def sym(ex):
    if ex.id not in _sym:
        _sym[ex.id] = length_symmetric(tok, ex)
    return _sym[ex.id]


groups = defaultdict(list)
for path in sys.argv[2:]:
    for line in open(path):
        r = json.loads(line)
        if r["mode"] == "skeleton" and r["id"] in exs:
            groups[("/".join(path.split("/")[-2:]), r["cfg_tag"])].append(r)

acc = lambda xs: sum(r["diagnosis"]["correct"] for r in xs) / max(len(xs), 1)  # noqa: E731
cc = lambda xs: sum(is_cross_call(r["diagnosis"]["labels"]) for r in xs) / max(len(xs), 1)  # noqa: E731
print("| file | cfg | n | set_acc | ccer | n sym | set_acc sym | ccer sym | cross-call details on sym params |")
print("|---|---|---|---|---|---|---|---|---|")
for (f, tag), rs in sorted(groups.items()):
    sy = [r for r in rs if all(sym(exs[r["id"]]).values())]
    det = [(r, d) for r in rs for d in r["diagnosis"]["details"] if d["label"] in CROSS_CALL and "param" in d]
    on_sym = sum(sym(exs[r["id"]]).get(d["param"], False) for r, d in det)
    print(f"| {f} | {tag} | {len(rs)} | {acc(rs):.3f} | {cc(rs):.3f} | {len(sy)} | {acc(sy):.3f} | {cc(sy):.3f} | "
          f"{on_sym}/{len(det)} |")
