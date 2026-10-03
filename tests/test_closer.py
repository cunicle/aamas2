"""Experiment C3: the closer-in-slot skeleton variant, and one-sided lengthening.

In the variant the skeleton writes no closer after a slot; the model writes it inside the
slot (one more position per slot), the positions after it get spaces, and decoding keeps
it. The default path must stay as it was, token for token.
"""

import hashlib
import importlib.util
import json
import os

import pytest
import torch

from ptcdiag.decoding.adapters import ToyAdapter
from ptcdiag.decoding.constraints import (STRING_TYPES, SkeletonConstraint, build_skeleton, onesided_lengths,
                                          onesided_pairs, oracle_lengths, sibling_groups, swapped_lengths,
                                          value_text)
from ptcdiag.decoding.sampler import DecodeConfig
from ptcdiag.pipeline import record_constraint, run_example
from ptcdiag.prompting import parse_tool_calls, render_prompt
from ptcdiag.types import Example

HI = 15.0
FN = {"name": "f", "description": "Test function.",
      "parameters": {"type": "dict", "required": ["city", "tags", "days"],
                     "properties": {"city": {"type": "string", "description": "City."},
                                    "tags": {"type": "array", "items": {"type": "integer"}, "description": "Tags."},
                                    "days": {"type": "integer", "description": "Days."}}}}
EXAMPLE = Example(id="closers", category="parallel",
                  messages=[{"role": "user", "content": "f for Paris, tags 1 and 2, 3 days"}],
                  functions=[FN], ground_truth=[{"f": {"city": ["Paris"], "tags": [[1, 2]], "days": [3]}}])
TARGET = '[{"name": "f", "arguments": {"city": "Paris", "tags": [1, 2], "days": 3}}]'
# sha256 of the default skeletons (oracle, surplus 1, swapped lengths) of the 400 BFCL parallel +
# parallel_multiple items with the Dream tokenizer, computed with constraints.py as of b30fa10,
# before the variant was added
DEFAULT_DIGEST = "5bb4b1a7c3f87525759b1347be2c551188f8a76c0873c533e97e749a1678a505"


def one_token(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        pytest.skip(f"{s!r} is not a single token for this tokenizer")
    return ids[0]


def bfcl_items(bfcl_examples):
    return bfcl_examples["parallel"] + bfcl_examples["parallel_multiple"]


def test_default_skeletons_token_identical(bfcl_examples, dream_tokenizer):
    tok = dream_tokenizer
    h = hashlib.sha256()
    for ex in bfcl_items(bfcl_examples):
        for kw in ({}, {"surplus": 1}, {"lengths": swapped_lengths(tok, ex)}):
            ids, slots = build_skeleton(tok, ex, tok.mask_token_id, **kw)
            assert (ids, slots) == build_skeleton(tok, ex, tok.mask_token_id, closer_in_slot=False, **kw)
            h.update(json.dumps([ex.id, ids, [[s.call, s.param, s.type, s.positions] for s in slots]]).encode())
    assert h.hexdigest() == DEFAULT_DIGEST


def _filled(tok, ex, closer_in_slot):
    """The skeleton with every slot holding its gold value (+ its closer in the variant), decoded."""
    ids, slots = build_skeleton(tok, ex, tok.mask_token_id, closer_in_slot=closer_in_slot)
    last = {}  # (call) -> the call's last slot
    for s in slots:
        last[s.call] = s
    gold = {(ci, p): next(a for a in acc if a != "")
            for ci, (_, params) in enumerate(ex.gold_calls) for p, acc in params.items() if any(a != "" for a in acc)}
    out, pos = [], 0
    for s in slots:
        out += ids[pos:s.positions[0]]
        text = value_text(gold[s.call, s.param], s.type)
        if closer_in_slot:
            text += '"' if s.type in STRING_TYPES else ("}" if last[s.call] is s else ",")
        out += tok(text, add_special_tokens=False)["input_ids"]
        pos = s.positions[-1] + 1
    out += ids[pos:]
    return tok.decode(out)


def test_variant_layout_reads_as_the_gold_json(bfcl_examples, dream_tokenizer):
    """Filled with value + closer, the variant's skeleton is the same text as the default one
    filled with the values: the closers moved from the skeleton into the slots, nothing else."""
    tok = dream_tokenizer
    assert _filled(tok, EXAMPLE, True) == _filled(tok, EXAMPLE, False) == TARGET
    for ex in bfcl_items(bfcl_examples)[::7]:
        default = _filled(tok, ex, False)
        assert _filled(tok, ex, True) == default
        assert parse_tool_calls(default).syntax_ok


def test_variant_leaves_closers_out_of_the_skeleton(dream_tokenizer):
    tok = dream_tokenizer
    ids, slots = build_skeleton(tok, EXAMPLE, tok.mask_token_id, closer_in_slot=True)
    after = {s.param: tok.decode(ids[s.positions[-1] + 1:s.positions[-1] + 4]) for s in slots}
    assert after["city"].startswith(', "tags')     # no closing quote; the ',' stays
    assert after["tags"].startswith(' "days')      # no ','
    assert after["days"].startswith("}]")          # one '}' of '}}' is the model's
    before = {s.param: tok.decode(ids[:s.positions[0]]) for s in slots}
    assert before["city"].endswith('"city": "') and before["days"].endswith('"days":')
    assert all(t != tok.mask_token_id for t in ids[:slots[0].positions[0]])


@pytest.mark.parametrize("surplus", [0, 1, 3])
def test_variant_slot_lengths(bfcl_examples, dream_tokenizer, surplus):
    tok = dream_tokenizer
    for ex in bfcl_items(bfcl_examples)[::11]:
        orc = oracle_lengths(tok, ex)
        _, slots = build_skeleton(tok, ex, tok.mask_token_id, surplus=surplus, closer_in_slot=True)
        assert {(s.call, s.param): len(s.positions) for s in slots} == {k: n + 1 + surplus for k, n in orc.items()}
        sw = swapped_lengths(tok, ex)
        _, slots = build_skeleton(tok, ex, tok.mask_token_id, lengths=sw, closer_in_slot=True)
        assert {(s.call, s.param): len(s.positions) for s in slots} == {k: n + 1 for k, n in sw.items()}


class VariantToy:
    """Writes each value and its closer, then junk; on the closer-in-slot skeleton."""

    def __init__(self, tok, closers=('"', ",", "}"), surplus=0, closer_bonus=0.0, never_close=False):
        t = lambda s: one_token(tok, s)  # noqa: E731
        q, c, b = closers
        self.tok = tok
        self.scripts = {"city": [t("Paris")] + ([] if never_close else [t(q)]),
                        "tags": [t(" ["), t("1"), t(","), t(" "), t("2"), t("]")] + ([] if never_close else [t(c)]),
                        "days": [t(" "), t("3")] + ([] if never_close else [t(b)])}
        self.junk = {"city": t("x"), "tags": t("9"), "days": t("7")}
        self.closers = {t(q), t(c), t(b)}
        self.bonus = closer_bonus
        self.mask, self.eos = tok.mask_token_id, tok.eos_token_id
        (self.space,) = tok(" ", add_special_tokens=False)["input_ids"]
        self.adapter = ToyAdapter(self.fn, len(tok), mask_id=self.mask, pad_id=self.eos,
                                  eos_ids=[self.eos], tokenizer=tok)
        self.P = len(self.adapter.encode(render_prompt(tok, EXAMPLE)))
        gen, slots = build_skeleton(tok, EXAMPLE, self.mask, surplus=surplus, closer_in_slot=True)
        self.G = len(gen)
        self.slot_at = {g: (s.param, o) for s in slots for o, g in enumerate(s.positions)}

    def fn(self, x):
        logits = torch.zeros(len(x), len(self.tok))
        for g, (param, o) in self.slot_at.items():
            script = self.scripts[param]
            tid = script[o] if o < len(script) else self.junk[param]
            logits[self.P + g, tid] = HI + (self.bonus if tid in self.closers else 0.0)
        return logits

    def run(self, k=1, order="left_to_right", surplus=0):
        cfg = DecodeConfig(gen_length=self.G, k=k, order=order, eos_early_stop=False)
        return run_example(self.adapter, EXAMPLE, cfg, "skeleton", surplus=surplus, closer_in_slot=True)


@pytest.mark.parametrize("surplus,order,k,bonus", [(0, "left_to_right", 1, 0.0), (2, "left_to_right", 1, 0.0),
                                                   (3, "confidence", 1, 5.0), (3, "confidence", 4, 5.0)])
def test_variant_ends_early_with_spaces_and_parses(dream_tokenizer, surplus, order, k, bonus):
    toy = VariantToy(dream_tokenizer, surplus=surplus, closer_bonus=bonus)
    rec = toy.run(k=k, order=order, surplus=surplus)
    assert rec["text"] == TARGET and rec["syntax_ok"] and rec["diagnosis"]["correct"], rec["text"]
    assert rec["closer_in_slot"] is True and rec["surplus"] == surplus
    forced = [p for s in rec["trace"]["steps"] for p in s["forced"]]
    assert all(rec["gen_ids"][p - toy.P] == toy.space for p in forced)      # spaces, not padding
    assert toy.eos not in rec["gen_ids"]
    if surplus and order == "left_to_right":
        assert len(forced) == 3 * surplus  # every slot ends early by `surplus` positions
    elif surplus:
        assert forced
    if order == "left_to_right":
        # the canvas itself reads as JSON: value, closer, spaces, the skeleton's next text
        canvas = dream_tokenizer.decode(rec["gen_ids"])
        assert parse_tool_calls(canvas).calls == parse_tool_calls(TARGET).calls


def test_variant_keeps_only_the_closer_of_a_longer_token(dream_tokenizer):
    """'",' or '}}' as the closing token: the canvas has the extra character, the decoded text
    keeps the text up to the closer, as the default path keeps the text before it."""
    toy = VariantToy(dream_tokenizer, closers=('",', ",", "}}"), surplus=1)
    rec = toy.run(surplus=1)
    assert rec["text"] == TARGET and rec["diagnosis"]["correct"]
    assert '",' in dream_tokenizer.decode(rec["gen_ids"])


def test_variant_slot_filled_without_closer_does_not_parse(dream_tokenizer):
    toy = VariantToy(dream_tokenizer, surplus=1, never_close=True)
    rec = toy.run(surplus=1)
    assert not rec["syntax_ok"] and rec["diagnosis"]["labels"] == ["syntax_error"]
    assert not any(s["forced"] for s in rec["trace"]["steps"])


def test_variant_spaces_do_not_end_a_value(dream_tokenizer):
    tok = dream_tokenizer
    toy = VariantToy(tok)
    c = SkeletonConstraint(toy.adapter, EXAMPLE, closer_in_slot=True)
    c.initial_gen(toy.G, toy.P)
    city = next(s for s in c.slots if s.param == "city")
    x = torch.tensor([[0] * toy.P + c.gen_ids])
    x[0, toy.P + city.positions[0]] = toy.space  # a space as content
    assert c.after_commit(x) == []
    x[0, toy.P + city.positions[1]] = one_token(tok, '"')
    assert c.after_commit(x) == []  # the slot is full: nothing left to fill
    assert c.cuts(x[0, toy.P:].tolist()) == {}  # a bare closer is kept as it is


def test_variant_record_replays(dream_tokenizer):
    toy = VariantToy(dream_tokenizer, surplus=2)
    rec = toy.run(surplus=2)
    c = record_constraint(toy.adapter, EXAMPLE, rec)
    assert c.closer_in_slot and c.gen_ids == SkeletonConstraint(toy.adapter, EXAMPLE, surplus=2,
                                                                closer_in_slot=True).gen_ids


def test_default_records_carry_no_variant_field(dream_tokenizer):
    tok = dream_tokenizer
    flat = ToyAdapter(lambda x: torch.zeros(len(x), len(tok)), len(tok), mask_id=tok.mask_token_id,
                      pad_id=tok.eos_token_id, eos_ids=[tok.eos_token_id], tokenizer=tok)
    cfg = DecodeConfig(gen_length=64, k=1, order="left_to_right", eos_early_stop=False)
    rec = run_example(flat, EXAMPLE, cfg, "skeleton")
    assert "closer_in_slot" not in rec and rec["length_mode"] == "oracle"
    assert len(rec["gen_ids"]) == len(build_skeleton(tok, EXAMPLE, tok.mask_token_id)[0])


def load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("surplus", [0, 2])
def test_probe_variant_looks_right_after_the_value(dream_tokenizer, surplus):
    """length_prior.py --closer-in-slot: the gold value is teacher-forced, the query is the first
    of the 1 + surplus positions after it, where the toy wants the closer."""
    lp = load_script("length_prior")
    toy = VariantToy(dream_tokenizer, surplus=surplus)
    rows = lp.measure(toy.adapter, EXAMPLE, surplus, None, closer_in_slot=True)
    orc = oracle_lengths(dream_tokenizer, EXAMPLE)
    assert {r["param"]: r["value_len"] for r in rows} == {p: n for (_, p), n in orc.items()}
    assert all(r["constrained_kind"] == "closer" and r["p_close"] > 0.5 for r in rows)


def test_onesided_lengths(dream_tokenizer):
    """One slot per unequal sibling group gets the longest sibling's length: i* is the group's
    first call shorter than j* (the longest, first on ties); everything else stays exact."""
    tok = dream_tokenizer
    g = {"name": "g", "description": "", "parameters": {"type": "dict", "properties": {
        "city": {"type": "string"}, "n": {"type": "integer"}, "unit": {"type": "string"}}}}
    ex = Example(id="one", category="parallel", messages=[{"role": "user", "content": "x"}], functions=[g],
                 ground_truth=[{"g": {"city": ["Rio de Janeiro, Brazil"], "n": [5], "unit": ["celsius"]}},
                               {"g": {"city": ["Paris"], "n": [1234], "unit": ["kelvin"]}},
                               {"g": {"city": ["Rome"], "n": [56789], "unit": ["celsius"]}}])
    orc = oracle_lengths(tok, ex)
    assert orc[0, "city"] > orc[1, "city"] and orc[2, "n"] > orc[1, "n"] > orc[0, "n"]
    pairs = onesided_pairs(tok, ex)
    assert pairs[("g", "city")] == (1, 0)   # j* = Rio (call 0); i* = the first shorter call
    assert pairs[("g", "n")] == (0, 2)      # j* = 56789 (call 2); i* = call 0, before j*
    if orc[0, "unit"] == orc[1, "unit"] == orc[2, "unit"]:
        assert ("g", "unit") not in pairs
    one = onesided_lengths(tok, ex)
    changed = {k for k in one if one[k] != orc[k]}
    assert changed == {(i, p) for (_, p), (i, j) in pairs.items() if orc[i, p] != orc[j, p]}
    assert one[1, "city"] == orc[0, "city"] and one[0, "n"] == orc[2, "n"]
    assert one[0, "city"] == orc[0, "city"] and one[2, "n"] == orc[2, "n"]  # j* keeps its own length


def test_onesided_items_are_the_swap_items(bfcl_examples, dream_tokenizer):
    tok = dream_tokenizer
    for ex in bfcl_items(bfcl_examples):
        orc = oracle_lengths(tok, ex)
        assert (onesided_lengths(tok, ex) != orc) == (swapped_lengths(tok, ex) != orc)
        assert set(onesided_pairs(tok, ex)) <= set(sibling_groups(ex))
