"""End to end with a real tokenizer and a toy "model" whose logits we control.

The toy prefers a fixed target answer everywhere except the two city tokens, which
are symmetric: while both are masked each city is equally likely; once one is
committed, the other slot prefers the remaining city. Greedy decoding that commits
both in one step therefore duplicates a call, and sequentialising that step fixes it.
"""

import json

import pytest
import torch

from ptcdiag.analysis.counterfactual import attribute, placebo, reproduces
from ptcdiag.analysis.dependency import step_dvs
from ptcdiag.decoding.adapters import ToyAdapter
from ptcdiag.decoding.constraints import build_skeleton, token_classes
from ptcdiag.decoding.sampler import DecodeConfig, Trace
from ptcdiag.pipeline import run_example
from ptcdiag.prompting import render_prompt
from ptcdiag.types import Example

HI = 30.0
FN = {"name": "get_weather", "description": "Weather for a city.",
      "parameters": {"type": "dict", "required": ["city"],
                     "properties": {"city": {"type": "string", "description": "City name."}}}}


def single_token_cities(tok):
    out = []
    for c in ["Paris", "Tokyo", "Rome", "Oslo", "Berlin", "Madrid", "Lima"]:
        ids = tok(c, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out.append((c, ids[0]))
    assert len(out) >= 2
    return out[:2]


def make_example(c1, c2):
    return Example(id="e2e", category="probe_parallel",
                   messages=[{"role": "user", "content": f"Weather in {c1} and {c2}?"}],
                   functions=[FN],
                   ground_truth=[{"get_weather": {"city": [c1]}}, {"get_weather": {"city": [c2]}}])


def ids(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


class Toy:
    """Holds the layout so the logits function can find the two city slots."""

    def __init__(self, tok, symmetric=True):
        (self.c1, self.t1), (self.c2, self.t2) = single_token_cities(tok)
        self.tok = tok
        self.example = make_example(self.c1, self.c2)
        self.symmetric = symmetric
        self.mask, self.eos = tok.mask_token_id, tok.eos_token_id
        self.adapter = ToyAdapter(self.fn, len(tok), mask_id=self.mask, pad_id=self.eos,
                                  eos_ids=[self.eos], tokenizer=tok)
        self.P = len(self.adapter.encode(render_prompt(tok, self.example)))
        self.target, self.city_pos = None, None

    def use_free_layout(self):
        pre = ids(self.tok, '[{"name": "get_weather", "arguments": {"city": "')
        mid = ids(self.tok, '"}}, {"name": "get_weather", "arguments": {"city": "')
        post = ids(self.tok, '"}}]')
        self.target = pre + [self.t1] + mid + [self.t2] + post
        self.city_pos = (len(pre), len(pre) + 1 + len(mid))
        n = len(self.target) + 4
        return n + (n % 2)  # even, so k=2 steps end with the two city slots together

    def use_skeleton_layout(self):
        gen, slots = build_skeleton(self.tok, self.example, self.mask)
        self.target = [t if t != self.mask else self.eos for t in gen]
        self.city_pos = (slots[0].positions[0], slots[1].positions[0])
        return len(gen)

    def fn(self, x):
        L = len(x)
        logits = torch.zeros(L, len(self.tok), dtype=torch.bfloat16)
        a, b = self.city_pos
        for i in range(self.P, L):
            j = i - self.P
            if j in (a, b) and self.symmetric:
                other = int(x[self.P + (b if j == a else a)])
                if other == self.mask:
                    logits[i, [self.t1, self.t2]] = HI
                else:
                    logits[i, self.t2 if other == self.t1 else self.t1] = HI
            elif j in (a, b):
                logits[i, self.t1 if j == a else self.t2] = HI
            else:
                logits[i, self.target[j] if j < len(self.target) else self.eos] = HI
        return logits


@pytest.fixture(scope="module")
def toy(dream_tokenizer):
    return Toy(dream_tokenizer)


def test_free_mode_duplicate_then_counterfactual_fix(toy):
    G = toy.use_free_layout()
    cfg = DecodeConfig(gen_length=G, k=2, eos_early_stop=False)
    rec = run_example(toy.adapter, toy.example, cfg, "free")
    json.dumps(rec)  # records must be serialisable
    assert rec["syntax_ok"], rec["text"]
    assert rec["diagnosis"]["labels"] == ["duplicate_call"], rec["text"]

    tr = Trace.from_dict(rec["trace"])
    a, b = toy.city_pos
    last = tr.steps[-1]
    assert sorted(last.positions) == [toy.P + a, toy.P + b]  # co-committed

    assert reproduces(toy.adapter, toy.example, rec, step=2)
    res = attribute(toy.adapter, toy.example, rec)
    assert len(res) == 1 and res[0]["label"] == "duplicate_call"
    assert res[0]["fixed_by"] == last.step and res[0]["became_correct"]

    dvs = step_dvs(toy.adapter, toy.example, rec, last.step)
    assert dvs["m"] == 2 and dvs["dvs"] > 5  # the second city is ~impossible given the first


def test_one_token_per_step_is_correct(toy):
    G = toy.use_free_layout()
    rec = run_example(toy.adapter, toy.example, DecodeConfig(gen_length=G, k=1, eos_early_stop=False), "free")
    assert rec["diagnosis"]["correct"], rec["text"]


def test_placebo_does_not_break_correct_output(dream_tokenizer):
    t = Toy(dream_tokenizer, symmetric=False)
    G = t.use_free_layout()
    rec = run_example(t.adapter, t.example, DecodeConfig(gen_length=G, k=2, eos_early_stop=False), "free")
    assert rec["diagnosis"]["correct"]
    res = placebo(t.adapter, t.example, rec)
    assert res is not None and res["broken"] is False


def test_skeleton_mode(toy):
    G = toy.use_skeleton_layout()
    # k large enough that the two city slots are committed in the same step
    cfg = DecodeConfig(gen_length=G, k=64, eos_early_stop=False)
    rec = run_example(toy.adapter, toy.example, cfg, "skeleton")
    assert rec["syntax_ok"], rec["text"]
    assert rec["diagnosis"]["labels"] == ["duplicate_call"], rec["text"]
    res = attribute(toy.adapter, toy.example, rec)
    assert res[0]["fixed_by"] is not None and res[0]["became_correct"]
    rec1 = run_example(toy.adapter, toy.example, DecodeConfig(gen_length=G, k=1, eos_early_stop=False), "skeleton")
    assert rec1["diagnosis"]["correct"], rec1["text"]


def test_token_classes(dream_tokenizer):
    tok = dream_tokenizer
    cls = token_classes(tok)
    paris = ids(tok, "Paris")[0]
    quote = ids(tok, '"')[0]
    assert paris in cls["string"] and quote not in cls["string"]
    assert ids(tok, "42")[0] in cls["integer"] and paris not in cls["integer"]
    assert ids(tok, "true")[0] in cls["boolean"]
