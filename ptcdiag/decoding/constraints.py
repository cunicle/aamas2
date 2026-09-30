"""Decoding constraints.

Constraint interface used by the sampler:
  initial_gen(G, P)   -> initial generation region (list of ids, mask_id = undecided) or None
  filter(x, cand, logits) -> logits with disallowed tokens set to -inf, for candidate positions
  early_stop_ok       -> whether EOS early stopping is safe under this constraint

SkeletonConstraint implements the "oracle skeleton" setting (proposal §6.2): the call
count, function names, argument keys, brackets and quotes are fixed on the canvas;
only argument values are masked, each slot restricted to tokens of its JSON type.
Syntax is then guaranteed by construction up to the value level (a number slot can
still produce e.g. "-" alone; those cases are reported as syntax errors, not hidden).

A grammar constraint for free-form generation (e.g. wrapping the completability check
of eth-sri/constrained-diffusion) should implement the same interface; see README.
"""

import re
from dataclasses import dataclass, field

import torch

STRING_TYPES = {"string", "any"}
# Chosen from BFCL parallel-category gold values (Dream tokenizer): covers p99 of strings
# (11 tokens) and the max of integers (9) and floats (12); arrays/dicts p99 is 40.
# Too-short slots would truncate values and show up as spurious wrong_value errors.
SLOT_LENGTHS = {"string": 14, "any": 14, "integer": 10, "float": 12, "boolean": 2,
                "array": 48, "tuple": 48, "dict": 48}

_INT = re.compile(r"^\s?-?\d+$|^\s?-$")
_FLOAT = re.compile(r"^\s?[-+]?[\d.eE]+$|^\s?[-+]$")


def _classify(text):
    """Which slot types may contain a token that decodes to `text`."""
    out = set()
    if "\n" in text or "\r" in text:
        return out
    out.add("generic")
    if '"' not in text and "\\" not in text:
        out.add("string")
    if _INT.match(text):
        out.add("integer")
    if _FLOAT.match(text):
        out.add("float")
    st = text.strip()
    if st and st == text.lstrip() and (st in "true" or st in "false"):
        out.add("boolean")
    return out


_CLASS_CACHE = {}


def token_classes(tokenizer):
    """{slot_type: sorted list of allowed token ids}, cached per tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer))
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    n = len(tokenizer)
    texts = tokenizer.batch_decode([[i] for i in range(n)], skip_special_tokens=False)
    special = set(tokenizer.all_special_ids)
    special |= {i for i, t in enumerate(texts) if t.startswith("<|") and t.endswith("|>")}
    classes = {k: [] for k in ("generic", "string", "integer", "float", "boolean")}
    for i, t in enumerate(texts):
        if i in special or not t:
            continue
        for c in _classify(t):
            classes[c].append(i)
    _CLASS_CACHE[key] = classes
    return classes


def slot_class(bfcl_type):
    if bfcl_type in STRING_TYPES:
        return "string"
    if bfcl_type in ("integer", "float", "boolean"):
        return bfcl_type
    return "generic"


class Constraint:
    early_stop_ok = True

    def initial_gen(self, G, P):
        return None

    def filter(self, x, cand, logits):
        return logits


class NoConstraint(Constraint):
    pass


@dataclass
class Slot:
    call: int
    param: str
    type: str                 # BFCL type of the parameter
    positions: list = field(default_factory=list)  # gen-region indices


def _param_type(example, fname, param):
    f = example.function(fname)
    return f["parameters"]["properties"].get(param, {}).get("type", "string") if f else "string"


def build_skeleton(tokenizer, example, mask_id, slot_lengths=None):
    """Oracle skeleton from the ground truth: (gen_ids, slots).

    Includes every gold call in gold order and every parameter with at least one
    non-empty acceptable value. Parameters whose only acceptable value is "" are left out.
    """
    lengths = dict(SLOT_LENGTHS, **(slot_lengths or {}))
    ids, slots, buf = [], [], []

    def flush():
        if buf:
            ids.extend(tokenizer("".join(buf), add_special_tokens=False)["input_ids"])
            buf.clear()

    buf.append("[")
    for ci, (fname, params) in enumerate(example.gold_calls):
        if ci:
            buf.append(", ")
        buf.append('{"name": ' + '"' + fname + '", "arguments": {')
        first = True
        for p, acc in params.items():
            if all(a == "" for a in acc):
                continue
            t = _param_type(example, fname, p)
            q = '"' if t in STRING_TYPES else ""
            buf.append(("" if first else ", ") + '"' + p + '": ' + q)
            first = False
            flush()
            n = lengths.get(t, lengths["string"])
            slots.append(Slot(ci, p, t, list(range(len(ids), len(ids) + n))))
            ids.extend([mask_id] * n)
            buf.append(q)
        buf.append("}}")
    buf.append("]")
    flush()
    return ids, slots


class SkeletonConstraint(Constraint):
    early_stop_ok = False  # pad tokens inside slots must not end generation

    def __init__(self, adapter, example, slot_lengths=None):
        self.a = adapter
        self.gen_ids, self.slots = build_skeleton(adapter.tokenizer, example, adapter.mask_id, slot_lengths)
        self.classes = token_classes(adapter.tokenizer)
        self.P = None
        self._masks = {}
        self._pos_class = {}
        for s in self.slots:
            for g in s.positions:
                self._pos_class[g] = slot_class(s.type)

    def initial_gen(self, G, P):
        if len(self.gen_ids) > G:
            raise ValueError(f"skeleton needs {len(self.gen_ids)} tokens > gen_length {G}")
        self.P = P
        return self.gen_ids + [self.a.pad_id] * (G - len(self.gen_ids))

    def _mask(self, cls, V, device):
        key = (cls, V, str(device))
        if key not in self._masks:
            m = torch.zeros(V, dtype=torch.bool, device=device)
            ids = torch.tensor([i for i in self.classes[cls] if i < V], dtype=torch.long, device=device)
            m[ids] = True
            m[self.a.pad_id] = True  # unused slot positions are filled with padding
            self._masks[key] = m
        return self._masks[key]

    def filter(self, x, cand, logits):
        V = logits.shape[-1]
        rows = [self._mask(self._pos_class.get(int(p) - self.P, "generic"), V, logits.device) for p in cand]
        allowed = torch.stack(rows)
        return logits.masked_fill(~allowed, float("-inf"))

    def slot_positions(self):
        """{(call, param): canvas positions} (after initial_gen has been called)."""
        return {(s.call, s.param): [self.P + g for g in s.positions] for s in self.slots}
