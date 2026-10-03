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

A value ends where the model writes the slot's closing token, as in the AR baseline:
a quote for strings, ',' or '}' for everything else (for arrays and dicts only outside
brackets and string literals; see `value_end`). The closing token and everything after
it in the slot are dropped when decoding, and the slot's still-masked positions after
it are set to padding at once (`after_commit`). The models practically never put
padding inside a JSON value (P(pad) ~1e-5 even right after the gold value on Dream /
LLaDA2.0), so without closers they fill every slot to the end.

Slot length (`oracle_lengths`): masked diffusion models read the number of masks as the
length of the content (in training every mask stands for one real token), so a slot
longer than the value gets filled ("HSBC for home loan of $500,000", 5000000000). Each
slot therefore has the token length of its own reference gold value. This is part of
the oracle: siblings of different lengths (the same parameter in other calls) can be
told apart by length, so cross-call errors are analysed on length-symmetric sibling
groups (`length_symmetric`), and the length-swap run (`swapped_lengths`) gives each slot
a sibling's length instead. One shared length per parameter (the longest value) was
tried and rejected: the models then wrote the longer sibling's value into the shorter
slot (sea_level 0 -> 1000), producing artificial duplicates / cross-bindings (17 of 18
cross-call errors in a 20-item pilot sat in such slots).

Before a non-string value the skeleton stops at '":' and the model writes the space
itself, as in natural tokenization (' [', ' true'); number / boolean slots accept
whitespace-only tokens for it.

Closer-in-slot variant (`closer_in_slot=True`, experiment C3): with the closer written by the
skeleton after every slot, a value that ends early leaves value, closer, padding and the
skeleton's closer on the canvas, a form never seen in training; the model's reluctance to end
early could be reluctance to write two closers. In the variant the skeleton leaves the closer
out (`closer_in_slot_skeleton`), every slot has one more position for the model to write it,
and the positions after it get spaces, so ending early reads as plain JSON. Decoding keeps the
closer. The default (False) path is unchanged.

A grammar constraint for free-form generation (e.g. wrapping the completability check
of eth-sri/constrained-diffusion) should implement the same interface; see README.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

import torch

STRING_TYPES = {"string", "any"}
# Fixed per-type lengths of the first design, kept for `slot_lengths` overrides (tests).
# Chosen from BFCL parallel-category gold values (Dream tokenizer): covers p99 of strings
# (11 tokens) and the max of integers (9) and floats (12); arrays/dicts p99 is 40.
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
    if text and not st:  # the space before a value (the skeleton stops at '":')
        out |= {"integer", "float", "boolean"}
    return out


_TEXT_CACHE = {}
_CLASS_CACHE = {}


def token_texts(tokenizer):
    """Decoded text of every token id (index = id), cached per tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer))
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = tokenizer.batch_decode([[i] for i in range(len(tokenizer))],
                                                  skip_special_tokens=False)
    return _TEXT_CACHE[key]


def token_classes(tokenizer):
    """{slot_type: sorted list of allowed token ids}, cached per tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer))
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    texts = token_texts(tokenizer)
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


def slot_closers(bfcl_type):
    """Characters that end a value of this type (the token after it in the skeleton)."""
    return ('"',) if bfcl_type in STRING_TYPES else (",", "}")


def value_end(texts, closers, nested):
    """Where a slot's value ends: (token index, char offset) of its closer, or None.

    texts:   decoded text of each slot token in order; None = still masked, "" = padding.
    nested:  False for string / number / boolean slots. Their token classes cannot
             contain a closer, so the value ends at the first token that starts with
             one, whatever is still masked before it (that can only end it earlier).
             True for arrays / dicts, whose values contain ',' and '}' themselves: a
             closer counts only outside brackets and string literals, so every earlier
             token must be decided, and the value can end inside a token (e.g. '],').
    """
    if not nested:
        for i, t in enumerate(texts):
            if t and t.startswith(closers):
                return i, 0
        return None
    depth, in_str, esc = 0, False, False
    for i, t in enumerate(texts):
        if t is None:
            return None
        for j, ch in enumerate(t):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif depth == 0 and ch in closers:
                return i, j
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
    return None


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


def _concrete(v):
    """A concrete JSON value from a BFCL acceptable value (dicts hold lists of acceptable
    values; the first non-empty one is taken, as in the tests' gold predictions)."""
    if isinstance(v, dict) and v and all(isinstance(a, list) for a in v.values()):
        return {k: _concrete(next(a for a in acc if a != "")) for k, acc in v.items()
                if any(a != "" for a in acc)}
    if isinstance(v, list):
        return [_concrete(a) for a in v]
    return v


def value_text(value, bfcl_type):
    """Slot text of a gold value: string contents without quotes, otherwise ' ' + JSON."""
    if bfcl_type in STRING_TYPES:
        return value if isinstance(value, str) else json.dumps(value)
    return " " + json.dumps(_concrete(value))


def oracle_lengths(tokenizer, example):
    """{(call index, param): slot length} = tokens of the reference gold value, i.e. the
    first non-empty acceptable value (the one the tests' gold predictions use)."""
    out = {}
    for ci, (fname, params) in enumerate(example.gold_calls):
        for p, acc in params.items():
            vals = [a for a in acc if a != ""]
            if vals:
                t = _param_type(example, fname, p)
                out[ci, p] = max(1, len(tokenizer(value_text(vals[0], t), add_special_tokens=False)["input_ids"]))
    return out


def sibling_groups(example):
    """{(function, param): [call indices]} for parameters filled by several calls of the
    same function, in ground-truth order (parameters whose only acceptable value is ""
    are left out, as in the skeleton)."""
    groups = defaultdict(list)
    for ci, (fname, params) in enumerate(example.gold_calls):
        for p, acc in params.items():
            if any(a != "" for a in acc):
                groups[fname, p].append(ci)
    return {k: v for k, v in groups.items() if len(v) > 1}


def swapped_lengths(tokenizer, example):
    """Oracle lengths with every sibling group's lengths rotated by one call: the slot of
    the group's i-th call gets the length of the (i+1)-th call's value, the last one the
    first's. Groups whose values have equal lengths are unchanged. The length-swap run: a
    model that binds values to slots by length writes the sibling's value into such a
    slot, one that binds by position writes its own value (cut short or padded out)."""
    base = oracle_lengths(tokenizer, example)
    out = dict(base)
    for (_, p), cis in sibling_groups(example).items():
        for i, ci in enumerate(cis):
            out[ci, p] = base[cis[(i + 1) % len(cis)], p]
    return out


def lengths_to_list(lengths):
    """{(call, param): n} -> [[call, param, n], ...] for JSON records."""
    return sorted([ci, p, n] for (ci, p), n in lengths.items())


def lengths_from_list(rows):
    return {(ci, p): n for ci, p, n in rows}


def length_from_hazard(h):
    """Mode of a value-length distribution given per-position end probabilities.

    h[j] (j >= 1): probability that a value of length j ends there; h[0] is unused (values
    are never empty). P(length = j) = h[j] * prod_{i<j} (1 - h[i]) for j < len(h); the
    remaining mass is a value that fills all len(h) positions."""
    cap, surv, best, arg = len(h), 1.0, -1.0, len(h)
    for j in range(1, cap):
        pj = surv * h[j]
        if pj > best:
            best, arg = pj, j
        surv *= 1 - h[j]
    return cap if surv > best else arg


def length_symmetric(tokenizer, example):
    """{param: bool}: every call's slot for this parameter has the same length, so the
    canvas gives no hint which value belongs to which call (parameters used once: True)."""
    by_param = defaultdict(set)
    for (ci, p), n in oracle_lengths(tokenizer, example).items():
        by_param[p].add(n)
    return {p: len(ns) == 1 for p, ns in by_param.items()}


def build_skeleton(tokenizer, example, mask_id, slot_lengths=None, surplus=0, lengths=None,
                   closer_in_slot=False):
    """Oracle skeleton from the ground truth: (gen_ids, slots).

    Includes every gold call in gold order and every parameter with at least one
    non-empty acceptable value. Parameters whose only acceptable value is "" are left out.
    Slot lengths come from `oracle_lengths`; `lengths` ({(call, param): n}) replaces them
    slot by slot (length swap, estimated lengths), and `slot_lengths` ({bfcl_type: n})
    overrides both per type with fixed lengths. `surplus` adds that many masks to every
    slot (the length-prior experiment: how a slot longer than its value changes the errors).
    `closer_in_slot` builds the interface variant of `closer_in_slot_skeleton` instead.
    """
    if closer_in_slot:
        return closer_in_slot_skeleton(tokenizer, example, mask_id, slot_lengths, surplus, lengths)
    oracle = oracle_lengths(tokenizer, example)
    oracle.update(lengths or {})
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
            buf.append(("" if first else ", ") + '"' + p + '":' + (' "' if q else ""))
            first = False
            flush()
            n = (slot_lengths or {}).get(t, oracle[ci, p]) + surplus
            slots.append(Slot(ci, p, t, list(range(len(ids), len(ids) + n))))
            ids.extend([mask_id] * n)
            buf.append(q)
        buf.append("}}")
    buf.append("]")
    flush()
    return ids, slots


def closer_in_slot_skeleton(tokenizer, example, mask_id, slot_lengths=None, surplus=0, lengths=None):
    """The closer-in-slot interface variant of `build_skeleton` (experiment C3): (gen_ids, slots).

    The skeleton no longer writes the token that ends a value; the model writes it inside the
    slot. A string slot stops at '"param": "' with no closing quote after it, so the fixed text
    after it still starts with ', "next":' or '}}'. After a non-string slot the ',' (or, after
    the last parameter, the first '}' of '}}') is left out, so the fixed text after it starts
    with ' "next":' or '}'. Every slot has one more position, for its closer: the length of
    `build_skeleton` (oracle, `lengths` or `slot_lengths`) + surplus + 1. A value that ends
    early then reads as plain JSON once the rest of its slot holds spaces ('"HSBC"   , "amount":'),
    instead of value, closer, padding and the skeleton's own closer.
    """
    oracle = oracle_lengths(tokenizer, example)
    oracle.update(lengths or {})
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
        first, open_value = True, False  # open_value: the last slot's ',' or '}' is the model's
        for p, acc in params.items():
            if all(a == "" for a in acc):
                continue
            t = _param_type(example, fname, p)
            string = t in STRING_TYPES
            sep = "" if first else (" " if open_value else ", ")
            buf.append(sep + '"' + p + '":' + (' "' if string else ""))
            first = False
            flush()
            n = (slot_lengths or {}).get(t, oracle[ci, p]) + surplus + 1
            slots.append(Slot(ci, p, t, list(range(len(ids), len(ids) + n))))
            ids.extend([mask_id] * n)
            open_value = not string
        buf.append("}" if open_value else "}}")
    buf.append("]")
    flush()
    return ids, slots


class SkeletonConstraint(Constraint):
    early_stop_ok = False  # pad tokens inside slots must not end generation

    def __init__(self, adapter, example, slot_lengths=None, surplus=0, end_bias=0.0, lengths=None,
                 closer_in_slot=False):
        """end_bias: added to the logits of padding and closing tokens in every slot, a
        decoding-time counterweight to the length prior (0 = the plain constraint).
        lengths: per-slot lengths instead of the oracle ones (see `build_skeleton`).
        closer_in_slot: the interface variant of `closer_in_slot_skeleton`. The slots accept the
        same tokens, but a value's closer stays in the decoded text, and the masked positions of
        a slot after its closer get the space token instead of padding (the closer is not a
        space: writing spaces does not end a value)."""
        self.a = adapter
        self.end_bias = end_bias
        self.closer_in_slot = closer_in_slot
        if closer_in_slot:
            (self.space_id,) = adapter.tokenizer(" ", add_special_tokens=False)["input_ids"]
        self.gen_ids, self.slots = build_skeleton(adapter.tokenizer, example, adapter.mask_id,
                                                  slot_lengths, surplus, lengths, closer_in_slot)
        self.classes = token_classes(adapter.tokenizer)
        self.texts = token_texts(adapter.tokenizer)
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

    def _end_mask(self, cls, V, device):
        """Tokens that end a value of this class: its closers and padding."""
        key = ("end", cls, V, str(device))
        if key not in self._masks:
            m = torch.zeros(V, dtype=torch.bool, device=device)
            ends = [i for i, t in enumerate(self.texts[:V]) if t.startswith(slot_closers(cls))]
            m[torch.tensor(ends, dtype=torch.long, device=device)] = True
            m[self.a.pad_id] = True  # unused slot positions are filled with padding
            self._masks[key] = m
        return self._masks[key]

    def _mask(self, cls, V, device):
        key = (cls, V, str(device))
        if key not in self._masks:
            m = self._end_mask(cls, V, device).clone()  # a value may end anywhere
            ids = torch.tensor([i for i in self.classes[cls] if i < V], dtype=torch.long, device=device)
            m[ids] = True
            self._masks[key] = m
        return self._masks[key]

    def filter(self, x, cand, logits):
        V = logits.shape[-1]
        classes = [self._pos_class.get(int(p) - self.P, "generic") for p in cand]
        allowed = torch.stack([self._mask(c, V, logits.device) for c in classes])
        logits = logits.masked_fill(~allowed, float("-inf"))
        if self.end_bias:
            ends = torch.stack([self._end_mask(c, V, logits.device) for c in classes])
            logits = logits + self.end_bias * ends.to(logits.dtype)
        return logits

    def _slot_texts(self, gen, s):
        special, out = self.a.special_ids, []
        for g in s.positions:
            t = gen[g]
            if t == self.a.mask_id:
                out.append(None)
            else:
                out.append("" if t in special or t >= len(self.texts) else self.texts[t])
        return out

    def _end(self, gen, s):
        return value_end(self._slot_texts(gen, s), slot_closers(s.type), slot_class(s.type) == "generic")

    def after_commit(self, x, exclude=()):
        """Set the masked positions after every ended value to padding, in place.

        Returns the canvas positions it filled. `exclude`: positions to leave alone
        (DVS reveals the tokens of one step one at a time and must keep the rest masked).
        The closer-in-slot variant fills them with the space token instead.
        """
        if self.closer_in_slot:
            return self._space_after_closer(x, exclude)
        P = self.P
        gen = x[0, P:P + len(self.gen_ids)].tolist()
        forced = []
        for s in self.slots:
            end = self._end(gen, s)
            if end is None:
                continue
            forced += [P + g for g in s.positions[end[0] + 1:]
                       if gen[g] == self.a.mask_id and P + g not in exclude]
        if forced:
            x[0, forced] = self.a.pad_id
        return forced

    def cuts(self, gen):
        """{gen-region index: replacement token ids} that cut every value at its closer.

        The closing token and the rest of the slot map to []; a closer inside a token
        (arrays / dicts, e.g. '],') keeps that token's text before it, re-tokenized.
        The closer-in-slot variant keeps the closer instead (`_cuts_keep_closer`).
        """
        if self.closer_in_slot:
            return self._cuts_keep_closer(gen)
        out = {}
        for s in self.slots:
            texts = ["" if t is None else t for t in self._slot_texts(gen, s)]
            end = value_end(texts, slot_closers(s.type), slot_class(s.type) == "generic")
            if end is None:
                continue
            i, off = end
            out[s.positions[i]] = \
                self.a.tokenizer(texts[i][:off], add_special_tokens=False)["input_ids"] if off else []
            for g in s.positions[i + 1:]:
                out[g] = []
        return out

    def slot_positions(self):
        """{(call, param): canvas positions} (after initial_gen has been called)."""
        return {(s.call, s.param): [self.P + g for g in s.positions] for s in self.slots}

    # -- the closer-in-slot variant (experiment C3)

    def _space_after_closer(self, x, exclude=()):
        """`after_commit` of the closer-in-slot variant: the masked positions after every ended
        value get the space token, so the canvas reads '"HSBC"   , "amount":'."""
        P = self.P
        gen = x[0, P:P + len(self.gen_ids)].tolist()
        forced = []
        for s in self.slots:
            end = self._end(gen, s)
            if end is None:
                continue
            forced += [P + g for g in s.positions[end[0] + 1:]
                       if gen[g] == self.a.mask_id and P + g not in exclude]
        if forced:
            x[0, forced] = self.space_id
        return forced

    def _cuts_keep_closer(self, gen):
        """`cuts` of the closer-in-slot variant: each value keeps its closer, since the skeleton
        has none after the slot, and loses the rest of its slot (the space filling). Of a closing
        token with more text after the closer ('",' for a string, '}}' for the last parameter)
        only the text up to the closer is kept, re-tokenized, as `cuts` keeps the text before it.
        A value without a closer keeps the whole slot (the output then does not parse)."""
        out = {}
        for s in self.slots:
            texts = ["" if t is None else t for t in self._slot_texts(gen, s)]
            end = value_end(texts, slot_closers(s.type), slot_class(s.type) == "generic")
            if end is None:
                continue
            i, off = end
            if off + 1 < len(texts[i]):
                out[s.positions[i]] = self.a.tokenizer(texts[i][:off + 1], add_special_tokens=False)["input_ids"]
            for g in s.positions[i + 1:]:
                out[g] = []
        return out
