"""Experiment D: format-tolerant slots and position agents.

Format-tolerant slots also accept filler and format tokens (an escaped newline in a string, a
decimal point after an integer) and normalize the value when decoding. Position agents decode one
call of the whole turn's skeleton; the other calls' slots stay masked and decode to placeholders.
The default path must stay as it was.
"""

import json
import os
import sys

import pytest
import torch

from ptcdiag.data.agents import position_agent_example, protocol_flags
from ptcdiag.decoding.adapters import ToyAdapter
from ptcdiag.decoding.constraints import (SkeletonConstraint, build_skeleton, normalize_value, token_classes,
                                          tolerant_classes)
from ptcdiag.decoding.sampler import DecodeConfig
from ptcdiag.pipeline import record_constraint, run_example
from ptcdiag.prompting import render_prompt
from ptcdiag.types import Example

HI = 15.0
FN = {"name": "g", "description": "Weather.", "parameters": {"type": "dict", "required": ["city", "days"], "properties": {
    "city": {"type": "string", "description": "City."}, "days": {"type": "integer", "description": "Days."}}}}
TWO = Example(id="two", category="parallel", messages=[{"role": "user", "content": "weather in Paris for 5 days "
                                                                                    "and Berlin for 7 days"}],
              functions=[FN], ground_truth=[{"g": {"city": ["Paris"], "days": [5]}},
                                            {"g": {"city": ["Berlin"], "days": [7]}}])


def one_token(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        pytest.skip(f"{s!r} is not a single token for this tokenizer")
    return ids[0]


def test_normalize_value():
    assert normalize_value("Taylor Swift\\n", "string") == "Taylor Swift"
    assert normalize_value("HSBC Bank", "string") == "HSBC Bank"          # content is not filler
    assert normalize_value('say \\"hi\\"', "string") == 'say \\"hi\\"'    # escapes survive
    assert normalize_value("a\\q", "string") == "a"                          # an invalid escape
    assert normalize_value(" 500000.0", "integer") == " 500000"
    assert normalize_value(" 500000L", "integer") == " 500000"
    assert normalize_value(" 5000000", "integer") == " 5000000"              # an extra digit stays
    assert normalize_value(" 2.5f", "float") == " 2.5"
    assert normalize_value(" true //", "boolean") == " true"
    assert normalize_value(" null", "integer") == " null"                    # no prefix: unchanged


def test_tolerant_classes_widen_the_typed_ones(dream_tokenizer):
    tok = dream_tokenizer
    typed, tol = token_classes(tok), tolerant_classes(tok)
    assert set(typed["string"]) <= set(tol["string"])
    assert set(typed["integer"]) | set(typed["float"]) <= set(tol["scalar"])
    esc, dot, comma, quote = (one_token(tok, s) for s in ("\\n", ".", ",", '"'))
    assert esc in tol["string"] and esc not in typed["string"]
    assert dot in tol["scalar"] and dot not in typed["integer"]
    assert comma not in tol["scalar"] and quote not in tol["string"]


class FillerToy:
    """Writes each value, then filler wherever a slot has room: an escaped newline in the city, a
    decimal point and zeros after the days. Never closes early."""

    def __init__(self, tok, surplus):
        t = lambda s: one_token(tok, s)  # noqa: E731
        self.tok = tok
        self.mask, self.eos = tok.mask_token_id, tok.eos_token_id
        self.adapter = ToyAdapter(self.fn, len(tok), mask_id=self.mask, pad_id=self.eos,
                                  eos_ids=[self.eos], tokenizer=tok)
        self.P = len(self.adapter.encode(render_prompt(tok, TWO)))
        gen, slots = build_skeleton(tok, TWO, self.mask, surplus=surplus)
        self.G = len(gen)
        value = {(0, "city"): [t("Paris")], (1, "city"): [t("Berlin")],
                 (0, "days"): [t(" "), t("5")], (1, "days"): [t(" "), t("7")]}
        filler = {"city": [t("\\n")], "days": [t("."), t("0")]}
        self.want = {}
        for s in slots:
            script = value[s.call, s.param] + filler[s.param] * 8
            for o, g in enumerate(s.positions):
                self.want[g] = script[o]
        self.second = {"city": t("x"), "days": t("9")}  # what the typed slot falls back to
        self.param = {g: s.param for s in slots for g in s.positions}

    def fn(self, x):
        logits = torch.zeros(len(x), len(self.tok))
        for g, tid in self.want.items():
            logits[self.P + g, tid] = HI
            logits[self.P + g, self.second[self.param[g]]] = HI - 1
        return logits

    def run(self, surplus, tolerant):
        cfg = DecodeConfig(gen_length=self.G, k=1, order="left_to_right", eos_early_stop=False)
        return run_example(self.adapter, TWO, cfg, "skeleton", surplus=surplus, tolerant=tolerant)


@pytest.mark.parametrize("surplus", [0, 1, 2])
def test_tolerant_slots_take_filler_and_normalize_it(dream_tokenizer, surplus):
    toy = FillerToy(dream_tokenizer, surplus)
    rec = toy.run(surplus, tolerant=True)
    assert rec["tolerant"] is True and rec["syntax_ok"] and rec["diagnosis"]["correct"], rec["text"]
    assert [c["arguments"] for c in rec["calls"]] == [{"city": "Paris", "days": 5}, {"city": "Berlin", "days": 7}]
    typed = toy.run(surplus, tolerant=False)
    assert "tolerant" not in typed
    assert typed["diagnosis"]["correct"] == (surplus == 0)  # the typed slot writes content instead


def test_tolerant_record_replays(dream_tokenizer):
    toy = FillerToy(dream_tokenizer, 1)
    rec = toy.run(1, tolerant=True)
    c = record_constraint(toy.adapter, TWO, rec)
    assert c.tolerant and c.active_call is None


class ValueToy:
    """Writes every slot's own value; records which slot positions were ever candidates."""

    def __init__(self, tok):
        t = lambda s: one_token(tok, s)  # noqa: E731
        self.tok = tok
        self.mask, self.eos = tok.mask_token_id, tok.eos_token_id
        self.adapter = ToyAdapter(self.fn, len(tok), mask_id=self.mask, pad_id=self.eos,
                                  eos_ids=[self.eos], tokenizer=tok)
        self.P = len(self.adapter.encode(render_prompt(tok, TWO)))
        gen, self.slots = build_skeleton(tok, TWO, self.mask)
        self.G = len(gen)
        value = {(0, "city"): [t("Paris")], (1, "city"): [t("Berlin")],
                 (0, "days"): [t(" "), t("5")], (1, "days"): [t(" "), t("7")]}
        self.want = {g: value[s.call, s.param][o] for s in self.slots for o, g in enumerate(s.positions)}

    def fn(self, x):
        logits = torch.zeros(len(x), len(self.tok))
        for g, tid in self.want.items():
            logits[self.P + g, tid] = HI
        return logits


@pytest.mark.parametrize("active,k", [(0, 1), (1, 1), (1, 16)])
def test_position_agent_decodes_only_its_call(dream_tokenizer, active, k):
    toy = ValueToy(dream_tokenizer)
    cfg = DecodeConfig(gen_length=toy.G, k=k, order="confidence", eos_early_stop=False)
    rec = run_example(toy.adapter, TWO, cfg, "skeleton", active_call=active)
    assert rec["active_call"] == active and rec["syntax_ok"], rec["text"]
    other = 1 - active
    for s in toy.slots:  # the other call's slots are never committed
        masked = all(rec["gen_ids"][g] == toy.mask for g in s.positions)
        assert masked == (s.call == other)
    calls = rec["calls"]
    assert calls[active]["arguments"] == {"city": ["Paris", "Berlin"][active], "days": [5, 7][active]}
    assert calls[other]["arguments"] == {"city": "...", "days": None}


def test_default_constraint_unchanged(dream_tokenizer):
    toy = ValueToy(dream_tokenizer)
    c = SkeletonConstraint(toy.adapter, TWO)
    assert c.tol is None and c.active_call is None and c.frozen is None
    c.initial_gen(toy.G, toy.P)
    assert c.frozen is None


def test_variants_do_not_combine(dream_tokenizer):
    toy = ValueToy(dream_tokenizer)
    with pytest.raises(ValueError):
        SkeletonConstraint(toy.adapter, TWO, closer_in_slot=True, tolerant=True)
    with pytest.raises(ValueError):
        SkeletonConstraint(toy.adapter, TWO, active_call=2)


class FakeAR:
    """An AR 'model' that always wants Paris; slots of length 1 stop after one token."""

    def __init__(self, tok):
        self.tokenizer, self.device, self.name = tok, "cpu", "fake-ar"
        self.paris = one_token(tok, "Paris")
        self.five = one_token(tok, "5")

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    def step(self, ids, past):
        logits = torch.zeros(len(self.tokenizer))
        logits[self.paris], logits[self.five] = HI, HI - 1  # an integer slot cannot take Paris
        return logits, past


@pytest.mark.parametrize("active", [0, 1])
def test_ar_position_agent_writes_placeholders(dream_tokenizer, active):
    from ptcdiag.decoding.ar import run_example_ar

    rec = run_example_ar(FakeAR(dream_tokenizer), TWO, mode="skeleton", active_call=active)
    assert rec["active_call"] == active and rec["syntax_ok"], rec["text"]
    other = 1 - active
    assert rec["calls"][other]["arguments"] == {"city": "...", "days": None}
    assert rec["calls"][active]["arguments"]["city"] == "Paris"


def test_position_protocol_and_example():
    assert protocol_flags("pos-anon") == (False, False)
    a = position_agent_example(TWO, 2, 2)
    assert a.messages == TWO.messages and a.ground_truth == TWO.ground_truth
    assert a.meta["agent"] == 2 and a.meta["protocol"] == "pos-anon"
    with pytest.raises(ValueError):
        position_agent_example(TWO, 3, 2)


def test_bfcl_team_of_position_agents(dream_tokenizer):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_agents_bfcl as rab

    seen = []

    def decode(aex, lengths, active_call=None):
        seen.append(active_call)
        calls = [{"name": "g", "arguments": {"city": c, "days": d}} for c, d in (("Paris", 5), ("Berlin", 7))]
        return json.dumps(calls), True, calls

    agents = rab.run_team(TWO, "pos-anon", decode, rab.lengths_fn(dream_tokenizer, TWO, "oracle"))
    assert seen == [0, 1]
    assert [a["call"]["arguments"]["city"] for a in agents] == ["Paris", "Berlin"]
    assert rab.assemble(TWO, agents)["diagnosis"]["correct"]
