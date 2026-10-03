"""Experiment C1 / C2: BFCL agent teams (subsets, agent examples, protocol texts, team metrics)."""

import importlib.util
import json
import os
import sys

import pytest

from ptcdiag.data.agents import ORDINALS, ordinal
from ptcdiag.data.agents_bfcl import agent_lengths, bfcl_agent_example, swap_changes, team_symmetric
from ptcdiag.decoding.constraints import build_skeleton, oracle_lengths, swapped_lengths
from ptcdiag.types import Example

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def load_script(name):
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)  # run_agents_bfcl imports run_agents
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = {"name": "g", "description": "Weather.", "parameters": {"type": "dict", "required": ["city", "days"], "properties": {
    "city": {"type": "string", "description": "City."}, "days": {"type": "integer", "description": "Days."}}}}
H = {"name": "h", "description": "Time.", "parameters": {"type": "dict", "required": ["city"], "properties": {
    "city": {"type": "string", "description": "City."}}}}


def example(calls, functions=(G, H), ex_id="toy"):
    return Example(id=ex_id, category="parallel_multiple", messages=[{"role": "user", "content": "the request"}],
                   functions=list(functions), ground_truth=[{f: {p: [v] for p, v in args.items()}} for f, args in calls])


# Paris / Berlin / London are one token each, New York City three (Dream tokenizer)
SYM = example([("g", {"city": "Paris", "days": 5}), ("g", {"city": "Berlin", "days": 7}),
               ("h", {"city": "London"})], ex_id="sym")
UNEQUAL = example([("g", {"city": "Paris", "days": 5}), ("g", {"city": "New York City", "days": 7})], ex_id="uneq")
NO_SIBS = example([("g", {"city": "Paris", "days": 5}), ("h", {"city": "Berlin"})], ex_id="nosib")


def call(f, **args):
    return {"name": f, "arguments": args}


def test_team_symmetric_cases(dream_tokenizer):
    tok = dream_tokenizer
    assert team_symmetric(tok, SYM)          # g's two calls: same parameters, same slot lengths
    assert not team_symmetric(tok, UNEQUAL)  # sibling slot lengths differ
    assert not team_symmetric(tok, NO_SIBS)  # no function called twice
    assert swap_changes(tok, UNEQUAL) and not swap_changes(tok, SYM) and not swap_changes(tok, NO_SIBS)


def test_team_symmetric_counts_only_skeleton_parameters(dream_tokenizer):
    """A parameter whose only acceptable value is "" is not in the skeleton, so it does not count."""
    ex = Example(id="opt", category="parallel", messages=[{"role": "user", "content": "x"}], functions=[G],
                 ground_truth=[{"g": {"city": ["Paris"], "days": [5]}},
                               {"g": {"city": ["Berlin"], "days": [7], "unit": [""]}}])
    assert team_symmetric(dream_tokenizer, ex)


def test_bfcl_agent_example_takes_the_ith_call(dream_tokenizer):
    tok = dream_tokenizer
    for i in (1, 2, 3):
        a = bfcl_agent_example(SYM, i, 3, "sim-label")
        assert a.ground_truth == [SYM.ground_truth[i - 1]]
        assert a.id == f"sym_a{i}" and a.functions == SYM.functions and a.category == SYM.category
        assert a.meta == {"agent": i, "protocol": "sim-label", "call": i - 1}
        (m,) = a.messages
        assert m["role"] == "user" and m["content"].startswith("the request\n\nYou are assistant")
        one = Example("x", SYM.category, SYM.messages, SYM.functions, [SYM.ground_truth[i - 1]])
        assert build_skeleton(tok, a, tok.mask_token_id) == build_skeleton(tok, one, tok.mask_token_id)
    with pytest.raises(ValueError):
        bfcl_agent_example(SYM, 1, 2, "sim-anon")  # a team has one agent per reference call


def fake_decoder(answers, fail=()):
    """Agent i answers answers[i-1]; agents in `fail` write text that does not parse."""
    def decode(aex, lengths):
        i = aex.meta["agent"]
        if i in fail:
            return f"garbled {i}", False, []
        return json.dumps([answers[i - 1]]), True, [answers[i - 1]]
    return decode


ANSWERS = [call("g", city="Paris", days=5), call("g", city="Berlin", days=7), call("h", city="London")]


def team(tok, ex, protocol, answers=ANSWERS, fail=(), mode="oracle"):
    rab = load_script("run_agents_bfcl")
    return rab.run_team(ex, protocol, fake_decoder(answers, fail), rab.lengths_fn(tok, ex, mode))


def test_sim_anon_sibling_texts_identical(dream_tokenizer):
    agents = team(dream_tokenizer, SYM, "sim-anon")
    assert len({a["user_text"] for a in agents}) == 1  # all three, siblings or not
    assert agents[0]["lengths"] == agents[1]["lengths"]  # g's agents: the same slots


def test_sim_rule_ordinals(dream_tokenizer):
    agents = team(dream_tokenizer, SYM, "sim-rule")
    for a, word in zip(agents, ["first", "second", "third"]):
        i = a["i"]
        assert a["user_text"].endswith(
            f"You are assistant {i} of 3 answering this request at the same time. Each assistant makes exactly "
            f"one of the 3 calls, and the assistants cannot see each other's calls. By convention, assistant {i} "
            f"makes the {word} of the calls, in the order in which the request mentions them. Make your one call.")
    assert [ordinal(i) for i in range(1, 9)] == list(ORDINALS) and ordinal(8) == "eighth"
    with pytest.raises(ValueError):
        ordinal(9)


@pytest.mark.parametrize("protocol", ["turn-anon", "turn-label"])
def test_turn_texts_carry_earlier_calls(dream_tokenizer, protocol):
    agents = team(dream_tokenizer, SYM, protocol, fail={2})
    assert "Calls made so far: none. Make your one call." in agents[0]["user_text"]
    assert f"Calls made so far: {json.dumps([ANSWERS[0]])}. Make your one call." in agents[1]["user_text"]
    assert f"Calls made so far: [{json.dumps(ANSWERS[0])}, garbled 2]. Make your one call." in agents[2]["user_text"]
    assert agents[1]["call"] is None and not agents[1]["syntax_ok"]


def test_agent_lengths_swap_mapping(dream_tokenizer):
    tok = dream_tokenizer
    sw, orc = swapped_lengths(tok, UNEQUAL), oracle_lengths(tok, UNEQUAL)
    assert agent_lengths(tok, UNEQUAL, 1, "oracle") is None
    for i in (1, 2):
        assert agent_lengths(tok, UNEQUAL, i, "swap") == {(0, p): sw[i - 1, p] for p in ("city", "days")}
    assert agent_lengths(tok, UNEQUAL, 1, "swap")[0, "city"] == orc[1, "city"]  # agent 1 gets call 2's length
    agents = team(tok, UNEQUAL, "sim-anon", mode="swap")
    assert agents[0]["lengths"] == [[0, "city", orc[1, "city"]], [0, "days", sw[0, "days"]]]
    with pytest.raises(ValueError):
        agent_lengths(tok, UNEQUAL, 1, "estimate")


def metrics_of(tok, ex, answers, protocol="sim-label", fail=()):
    rab, an = load_script("run_agents_bfcl"), load_script("agents_bfcl_analysis")
    agents = team(tok, ex, protocol, answers, fail)
    t = rab.assemble(ex, agents)
    calls, unparsed = an.team_calls({"agents": agents, "team": t})
    return t, an.team_metrics(ex, calls, t["diagnosis"], unparsed)


def test_team_assembly_and_metrics(dream_tokenizer):
    tok = dream_tokenizer
    t, m = metrics_of(tok, SYM, ANSWERS)
    assert t["syntax_ok"] and json.loads(t["text"]) == ANSWERS and t["diagnosis"]["correct"]
    assert m == {"set_acc": True, "ccer": False, "duplicate": False, "in_order": True,
                 "first_mention": (1, 2), "unparsed": False}  # g's group: agent 1 passes call 1's check

    # siblings swapped: still a correct set, not in order, nobody but agent 2 on the first mention
    t, m = metrics_of(tok, SYM, [ANSWERS[1], ANSWERS[0], ANSWERS[2]])
    assert m["set_acc"] and not m["in_order"] and not m["duplicate"] and m["first_mention"] == (1, 2)

    # both g agents take the first mentioned entity: a duplicate
    t, m = metrics_of(tok, SYM, [ANSWERS[0], ANSWERS[0], ANSWERS[2]])
    assert "duplicate_call" in t["diagnosis"]["labels"]
    assert m["duplicate"] and m["ccer"] and not m["set_acc"] and not m["in_order"] and m["first_mention"] == (2, 2)

    # an agent without exactly one call: the team is unparsed
    t, m = metrics_of(tok, SYM, ANSWERS, fail={3})
    assert not t["syntax_ok"] and t["text"] is None and t["diagnosis"]["labels"] == ["syntax_error"]
    assert m["unparsed"] and not m["set_acc"] and not m["duplicate"] and not m["in_order"]
    assert m["first_mention"] == (1, 2)


def test_slot_kinds(dream_tokenizer):
    an = load_script("agents_bfcl_analysis")
    tok = dream_tokenizer
    ex = example([("g", {"city": "Paris", "days": 5}), ("g", {"city": "New York City", "days": 7}),
                  ("g", {"city": "Berlin", "days": 9})], functions=[G])
    L = oracle_lengths(tok, ex)
    assert an.slot_kind(ex, 0, "city", call("g", city="Paris", days=5), L[0, "city"], L) == "own"
    assert an.slot_kind(ex, 0, "city", call("g", city="new york city", days=5), L[1, "city"], L) == "sibling_fit"
    assert an.slot_kind(ex, 0, "city", call("g", city="New York City", days=5), L[0, "city"], L) == "sibling_other"
    assert an.slot_kind(ex, 1, "city", call("g", city="Berlin", days=7), L[0, "city"], L) == "sibling_fit"  # Berlin fits
    assert an.slot_kind(ex, 0, "city", call("g", city="Boston", days=5), L[0, "city"], L) == "other"
    assert an.slot_kind(ex, 0, "city", None, L[0, "city"], L) == "other"
    assert an.slot_kind(ex, 0, "city", call("g", days=5), L[0, "city"], L) == "other"  # missing


def test_ar_refuses_swapped_lengths(monkeypatch):
    rab = load_script("run_agents_bfcl")
    monkeypatch.setattr(sys, "argv", ["run_agents_bfcl.py", "--backend", "ar", "--length-mode", "swap",
                                      "--subset", "swap", "--dry-run"])
    with pytest.raises(SystemExit, match="no per-slot lengths"):
        rab.main()
