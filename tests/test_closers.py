"""Value slots end at their closing token, for the dLLM skeleton and the AR baseline alike.

Real models put a closer ('",', ',', '}}', '],' ...) right after a value and practically
never padding, so without this the dLLM fills every fixed-length slot to the end.
"""

import pytest
import torch

from ptcdiag.analysis.counterfactual import reproduces
from ptcdiag.analysis.dependency import step_dvs
from ptcdiag.decoding.adapters import ToyAdapter
from ptcdiag.decoding.ar import generate_skeleton
from ptcdiag.decoding.constraints import (SLOT_LENGTHS, build_skeleton, length_symmetric, token_classes,
                                          token_texts, value_end)
from ptcdiag.decoding.sampler import DecodeConfig, Sampler, Trace
from ptcdiag.pipeline import decode_region, make_constraint, run_example
from ptcdiag.prompting import render_prompt
from ptcdiag.types import Example

HI = 15.0  # with ~150k tokens: p ~0.96, and ~0.9997 with a +5 closer bonus (not 1.0 in float32)
FN = {"name": "f", "description": "Test function.",
      "parameters": {"type": "dict", "required": ["city", "tags", "days"],
                     "properties": {"city": {"type": "string", "description": "City."},
                                    "tags": {"type": "array", "items": {"type": "integer"}, "description": "Tags."},
                                    "days": {"type": "integer", "description": "Days."}}}}
EXAMPLE = Example(id="closers", category="parallel",
                  messages=[{"role": "user", "content": "f for Paris, tags 1 and 2, 3 days"}],
                  functions=[FN], ground_truth=[{"f": {"city": ["Paris"], "tags": [[1, 2]], "days": [3]}}])
TARGET = '[{"name": "f", "arguments": {"city": "Paris", "tags": [1, 2], "days": 3}}]'
# the first design's fixed lengths: room for junk after each value (oracle lengths leave none)
FIXED = SLOT_LENGTHS


def test_value_end_scalar():
    q, c = ('"',), (",", "}")
    assert value_end(["Taylor", " Swift", '",', " x"], q, False) == (2, 0)
    assert value_end(["Taylor", None, '",', None], q, False) == (2, 0)  # masked before it: still the end
    assert value_end(["Taylor", "", " Swift"], q, False) is None        # padding is not a closer
    assert value_end(["2", "0", "}}"], c, False) == (2, 0)


def test_value_end_nested():
    c = (",", "}")
    assert value_end(["[", "3", ",", " ", "5", "],", " x"], c, True) == (5, 1)
    assert value_end(["[", "3", ",", " ", "5", "]", "}}"], c, True) == (6, 0)
    assert value_end(['["', "a", ",", " b", '",', ' "', "c", '"]', ","], c, True) == (8, 0)  # ',' in a string
    assert value_end(['{"', "x", '":', " 1", "}},"], c, True) == (4, 1)
    assert value_end(["[", None, "],"], c, True) is None  # depth unknown until every token is decided


def one_token(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        pytest.skip(f"{s!r} is not a single token for this tokenizer")
    return ids[0]


class SlotToy:
    """Writes each value, then its closer, then junk; closers can be made the most confident."""

    def __init__(self, tok, closer_bonus=0.0, slot_lengths=FIXED, surplus=0):
        t = lambda s: one_token(tok, s)  # noqa: E731
        self.tok = tok
        self.scripts = {"city": ([t("Paris"), t('",')], t("x")),
                        "tags": ([t(" ["), t("1"), t(","), t(" "), t("2"), t("],")], t("9")),
                        "days": ([t(" "), t("3"), t("}}")], t("7"))}
        self.closers = {t('",'), t("],"), t("}}")}
        self.bonus = closer_bonus
        self.mask, self.eos = tok.mask_token_id, tok.eos_token_id
        self.adapter = ToyAdapter(self.fn, len(tok), mask_id=self.mask, pad_id=self.eos,
                                  eos_ids=[self.eos], tokenizer=tok)
        self.P = len(self.adapter.encode(render_prompt(tok, EXAMPLE)))
        gen, slots = build_skeleton(tok, EXAMPLE, self.mask, slot_lengths, surplus)
        self.G = len(gen)
        self.slot_at = {g: (s.param, o) for s in slots for o, g in enumerate(s.positions)}

    def fn(self, x):
        logits = torch.zeros(len(x), len(self.tok))
        for g, (param, o) in self.slot_at.items():
            script, junk = self.scripts[param]
            t = script[o] if o < len(script) else junk
            logits[self.P + g, t] = HI + (self.bonus if t in self.closers else 0.0)
        return logits


@pytest.mark.parametrize("order,k,bonus", [("left_to_right", 1, 0.0), ("confidence", 4, 5.0),
                                           ("confidence", 1, 0.0)])
def test_dllm_slots_end_at_closer(dream_tokenizer, order, k, bonus):
    toy = SlotToy(dream_tokenizer, bonus)
    cfg = DecodeConfig(gen_length=toy.G, k=k, order=order, eos_early_stop=False)
    rec = run_example(toy.adapter, EXAMPLE, cfg, "skeleton", FIXED)
    assert rec["text"] == TARGET and rec["diagnosis"]["correct"], rec["text"]
    forced = [p for s in rec["trace"]["steps"] for p in s["forced"]]
    assert all(rec["gen_ids"][p - toy.P] == toy.eos for p in forced)
    if order == "left_to_right" or bonus:  # ties among equal logits may commit junk first otherwise
        assert forced
    assert not set(forced) & {p for s in rec["trace"]["steps"] for p in s["positions"]}
    assert reproduces(toy.adapter, EXAMPLE, rec, step=2)


def test_closers_first_pad_before_values(dream_tokenizer):
    """Closers most confident: scalar slots are padded behind them before the values exist."""
    # +15: closers at p = 1.0 in float32; values stay below even in small integer classes
    toy = SlotToy(dream_tokenizer, closer_bonus=15.0)
    rec = run_example(toy.adapter, EXAMPLE, DecodeConfig(gen_length=toy.G, k=1, eos_early_stop=False),
                      "skeleton", FIXED)
    assert rec["text"] == TARGET
    first3 = rec["trace"]["steps"][:3]
    assert all(s["tokens"][0] in toy.closers for s in first3)
    assert sum(bool(s["forced"]) for s in first3) == 2  # city and days; the array needs its prefix

    # DVS on a step whose closer pads part of the canvas: only the committed tokens count
    rec4 = run_example(toy.adapter, EXAMPLE, DecodeConfig(gen_length=toy.G, k=4, eos_early_stop=False),
                       "skeleton", FIXED)
    s0 = Trace.from_dict(rec4["trace"]).steps[0]
    assert s0.forced and len(s0.positions) == 4
    assert step_dvs(toy.adapter, EXAMPLE, rec4, 0)["m"] == 4


def test_sequentialized_step_skips_padded_positions(dream_tokenizer):
    toy = SlotToy(dream_tokenizer, closer_bonus=5.0)
    prompt = toy.adapter.encode(render_prompt(dream_tokenizer, EXAMPLE))
    cfg = DecodeConfig(gen_length=toy.G, k=10_000, eos_early_stop=False, sequentialize_steps=(0,))
    constraint = make_constraint(toy.adapter, EXAMPLE, "skeleton", FIXED)
    canvas, tr = Sampler(toy.adapter, cfg, constraint).generate(prompt)
    s0 = tr.steps[0]
    assert s0.sequentialized and len(tr.steps) == 1 and s0.forced
    assert len(s0.positions) + len(s0.forced) == sum(1 for t in constraint.gen_ids if t == toy.mask)
    assert decode_region(toy.adapter, canvas, tr, "skeleton", constraint)[0] == TARGET


class FakeAR:
    """Greedy 'model' that wants to write TARGET, preferring the longest matching token."""

    def __init__(self, tok, extra=(" [", "1", ",", " ", "2", "3", "Paris", "],", "}}]", "}}", '",')):
        self.tokenizer, self.device, self.fed = tok, "cpu", []
        self.texts = token_texts(tok)
        self.vocab = set(tok(TARGET, add_special_tokens=False)["input_ids"])
        self.vocab |= {i for s in extra for i in tok(s, add_special_tokens=False)["input_ids"]}

    def step(self, ids, past):
        if past is not None:
            self.fed += ids[0].tolist()
        done = self.tokenizer.decode(self.fed)
        rest = TARGET[len(done):] if TARGET.startswith(done) else ""
        best = max((i for i in self.vocab if self.texts[i] and rest.startswith(self.texts[i])),
                   key=lambda i: len(self.texts[i]), default=0)
        logits = torch.zeros(len(self.tokenizer))
        logits[best] = 1.0
        return logits, True


def test_ar_array_value_is_not_cut_at_inner_comma(dream_tokenizer):
    one_token(dream_tokenizer, "],")
    ar = FakeAR(dream_tokenizer)
    text, _ = generate_skeleton(ar, torch.tensor([[1, 2, 3]]), EXAMPLE)
    assert text == TARGET


def test_oracle_slot_lengths(dream_tokenizer):
    """Each slot has the length of its own reference value (first acceptable one);
    non-string values get their leading space inside the slot."""
    tok = dream_tokenizer
    fn = {"name": "g", "description": "", "parameters": {"type": "dict", "properties": {
        "city": {"type": "string"}, "n": {"type": "integer"}, "on": {"type": "boolean"}}}}
    ex = Example(id="len", category="parallel", messages=[{"role": "user", "content": "x"}], functions=[fn],
                 ground_truth=[{"g": {"city": ["Paris"], "n": [5], "on": [True, ""]}},
                               {"g": {"city": ["New York City", "NYC"], "n": [1234], "on": [False]}}])
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    gen, slots = build_skeleton(tok, ex, tok.mask_token_id)
    by = {(s.call, s.param): s for s in slots}
    assert len(by[0, "city"].positions) == n_tok("Paris")
    assert len(by[1, "city"].positions) == n_tok("New York City")  # first acceptable value, not "NYC"
    assert len(by[0, "n"].positions) == n_tok(" 5") and len(by[1, "n"].positions) == n_tok(" 1234")
    assert len(by[0, "on"].positions) == n_tok(" true")
    sym = length_symmetric(tok, ex)
    assert sym["on"] == (n_tok(" true") == n_tok(" false")) and not sym["n"] and not sym["city"]
    assert tok.decode(gen[:by[0, "n"].positions[0]]).endswith('"n":')       # space left to the model
    assert tok.decode(gen[:by[0, "city"].positions[0]]).endswith('"city": "')
    space = tok(" ", add_special_tokens=False)["input_ids"][0]
    cls = token_classes(tok)
    assert all(space in cls[c] for c in ("integer", "float", "boolean"))


def test_ar_ignores_surplus(dream_tokenizer):
    """The AR baseline stops at the closer, so longer slots change nothing."""
    text, _ = generate_skeleton(FakeAR(dream_tokenizer), torch.tensor([[1, 2, 3]]), EXAMPLE, surplus=3)
    assert text == TARGET


def test_surplus_adds_masks_to_every_slot(dream_tokenizer):
    tok = dream_tokenizer
    _, base = build_skeleton(tok, EXAMPLE, tok.mask_token_id)
    _, more = build_skeleton(tok, EXAMPLE, tok.mask_token_id, surplus=3)
    assert [len(s.positions) + 3 for s in base] == [len(s.positions) for s in more]


def test_surplus_run_records_and_replays(dream_tokenizer):
    """Oracle lengths + surplus: each value ends at its closer, the record carries the
    surplus, and replay rebuilds the same (longer) skeleton."""
    toy = SlotToy(dream_tokenizer, slot_lengths=None, surplus=2)
    cfg = DecodeConfig(gen_length=toy.G, k=1, order="left_to_right", eos_early_stop=False)
    rec = run_example(toy.adapter, EXAMPLE, cfg, "skeleton", surplus=2)
    assert rec["text"] == TARGET and rec["surplus"] == 2 and rec["end_bias"] == 0.0
    assert any(s["forced"] for s in rec["trace"]["steps"])
    assert reproduces(toy.adapter, EXAMPLE, rec, step=2)


def test_end_bias_raises_padding_and_closers(dream_tokenizer):
    tok = dream_tokenizer
    t = lambda s: one_token(tok, s)  # noqa: E731
    toy = SlotToy(tok)
    c = make_constraint(toy.adapter, EXAMPLE, "skeleton", FIXED, end_bias=3.0)
    c.initial_gen(toy.G, toy.P)
    city = next(s for s in c.slots if s.param == "city")
    x = torch.zeros(1, toy.P + toy.G, dtype=torch.long)
    out = c.filter(x, torch.tensor([toy.P + city.positions[0]]), torch.zeros(1, len(tok)))[0]
    assert out[toy.eos] == 3.0 and out[t('",')] == 3.0 and out[t('"')] == 3.0  # padding, closers
    assert out[t("Paris")] == 0.0                                                 # value tokens
    assert torch.isinf(out[t("\n")])                                              # not allowed at all
