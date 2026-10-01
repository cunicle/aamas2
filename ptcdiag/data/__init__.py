import json

from ptcdiag.data import bfcl
from ptcdiag.types import Example


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [Example.from_dict(json.loads(line)) for line in f if line.strip()]


def load_examples(spec, bfcl_dir="data/bfcl"):
    """spec: "bfcl:parallel,parallel_multiple" or "probe:data/paraprobe.jsonl"."""
    kind, _, arg = spec.partition(":")
    if kind == "bfcl":
        bfcl.download(bfcl_dir)
        return [ex for c in arg.split(",") for ex in bfcl.load(c, bfcl_dir)]
    if kind == "probe":
        return load_jsonl(arg)
    raise ValueError(f"unknown dataset spec {spec!r}")
