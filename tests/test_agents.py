"""Experiment B: the agents' user texts under the four protocols, and the team metrics."""

import importlib.util
import json
import os

import pytest

from ptcdiag.data import choose
from ptcdiag.data.agents import PROTOCOLS, agent_example
from ptcdiag.decoding.constraints import build_skeleton

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ITEMS = choose.generate()
LIST3 = next(e for e in ITEMS if e.meta["variant"] == "list" and e.meta["n"] == 3)
OPEN4 = next(e for e in ITEMS if e.meta["variant"] == "open" and e.meta["n"] == 4)


def call(city):
    return {"name": "get_weather", "arguments": {"city": city}}


def fake_decoder(cities, fail=()):
    """Agent i answers cities[i-1]; agents in `fail` write text that does not parse."""
    def decode(aex):
        i = aex.meta["agent"]
        if i in fail:
            return f"garbled output {i}", False, []
        c = call(cities[i - 1])
        return json.dumps([c]), True, [c]
    return decode


def team(ex, protocol, cities=("Paris", "Berlin", "London", "Bern"), fail=()):
    return load_script("run_agents").run_team(ex, protocol, fake_decoder(cities, fail))


def test_agent_example_fields():
    a = agent_example(LIST3, 2, 3, "sim-label")
    assert a.id == f"{LIST3.id}_a2" and a.category == LIST3.category
    assert a.functions == LIST3.functions and a.ground_truth == LIST3.ground_truth[:1]
    assert a.meta == {**LIST3.meta, "agent": 2, "protocol": "sim-label"}
    (m,) = a.messages
    assert m["role"] == "user" and m["content"].startswith(LIST3.messages[0]["content"] + "\n\n")
    assert m["content"].endswith("Make your one call.")


@pytest.mark.parametrize("ex", [LIST3, OPEN4])
def test_sim_anon_texts_identical(ex):
    # the decoder's different answers must not reach simultaneous agents
    texts = [a["user_text"] for a in team(ex, "sim-anon")]
    assert len(texts) == ex.meta["n"] and len(set(texts)) == 1


@pytest.mark.parametrize("ex", [LIST3, OPEN4])
def test_sim_label_only_index_differs(ex):
    n = ex.meta["n"]
    texts = [a["user_text"] for a in team(ex, "sim-label")]
    assert len(set(texts)) == n
    for i, t in enumerate(texts, 1):
        assert f"You are assistant {i} of {n} answering" in t
        assert t.replace(f"assistant {i} of", "assistant # of") == texts[0].replace("assistant 1 of", "assistant # of")
    # and apart from the index it is the anonymous text's situation, not the turn-taking one
    assert "at the same time" in texts[0] and "Calls made so far" not in texts[0]


@pytest.mark.parametrize("protocol", ["turn-anon", "turn-label"])
@pytest.mark.parametrize("ex", [LIST3, OPEN4])
def test_turn_texts_carry_earlier_calls(ex, protocol):
    cities = ["Paris", "Berlin", "London", "Bern"]
    agents = team(ex, protocol, cities)
    assert "Calls made so far: none. Make your one call." in agents[0]["user_text"]
    for a in agents[1:]:
        i = a["i"]
        shown = json.dumps([call(c) for c in cities[:i - 1]])
        assert f"Calls made so far: {shown}. Make your one call." in a["user_text"]
        assert cities[i - 1] not in a["user_text"].split("Calls made so far:")[1]
    assert ("You are assistant 2 of" in agents[1]["user_text"]) == (protocol == "turn-label")


def test_turn_text_example_format():
    a = agent_example(LIST3, 2, 3, "turn-anon", [call("Chicago")])
    assert a.messages[0]["content"].endswith(
        'Calls made so far: [{"name": "get_weather", "arguments": {"city": "Chicago"}}]. Make your one call.')


def test_turn_unparsed_output_passed_as_raw_text():
    agents = team(LIST3, "turn-label", fail={1})
    assert agents[0]["city"] is None and not agents[0]["syntax_ok"]
    assert 'Calls made so far: [garbled output 1, {"name": "get_weather", "arguments": {"city": "Berlin"}}].' \
        in agents[2]["user_text"]


def test_simultaneous_agents_refuse_earlier_calls():
    with pytest.raises(ValueError):
        agent_example(LIST3, 2, 3, "sim-anon", [call("Paris")])


def test_protocol_texts_verbatim():
    assert [agent_example(OPEN4, 2, 4, p, [call("Miami")] if PROTOCOLS[p][0] else ()).messages[0]["content"]
            .split("\n\n", 1)[1] for p in PROTOCOLS] == [
        "You are one of 4 assistants answering this request at the same time. Each assistant makes exactly "
        "one of the 4 calls, and the assistants cannot see each other's calls. Make your one call.",
        "You are assistant 2 of 4 answering this request at the same time. Each assistant makes exactly one "
        "of the 4 calls, and the assistants cannot see each other's calls. Make your one call.",
        "You are one of 4 assistants answering this request one after another. Each assistant makes exactly "
        'one of the 4 calls. Calls made so far: [{"name": "get_weather", "arguments": {"city": "Miami"}}]. '
        "Make your one call.",
        "You are assistant 2 of 4 answering this request one after another. Each assistant makes exactly one "
        'of the 4 calls. Calls made so far: [{"name": "get_weather", "arguments": {"city": "Miami"}}]. '
        "Make your one call.",
    ]


def test_agent_skeleton_one_slot_of_original_length(dream_tokenizer):
    tok = dream_tokenizer
    for ex in ITEMS:
        _, orig = build_skeleton(tok, ex, tok.mask_token_id)
        n = ex.meta["n"]
        for p in PROTOCOLS:
            prev = [call("Paris")] * (n - 1) if PROTOCOLS[p][0] else ()
            ids, slots = build_skeleton(tok, agent_example(ex, n, n, p, prev), tok.mask_token_id)
            assert len(slots) == 1 and slots[0].param == "city"
            assert len(slots[0].positions) == len(orig[0].positions) == 1


def test_team_stats():
    aa = load_script("agents_analysis")
    a, b, c = LIST3.meta["listed"][:3]

    def rec(*cities):
        return {"agents": [{"i": i, "syntax_ok": x is not None, "city": x} for i, x in enumerate(cities, 1)]}

    assert aa.team_stats(LIST3, rec(a, b, c)) == {"ok": True, "duplicate": False, "all_same": False,
                                                  "invalid": False, "in_order": True}
    s = aa.team_stats(LIST3, rec(c, b, a))
    assert s["ok"] and not s["in_order"]
    s = aa.team_stats(LIST3, rec(a, a, a))
    assert s["duplicate"] and s["all_same"] and not s["ok"] and not s["invalid"]
    s = aa.team_stats(LIST3, rec(a, a, b))
    assert s["duplicate"] and not s["all_same"]
    s = aa.team_stats(LIST3, rec(a, None, a))  # unparsed agent: invalid, the other two still duplicate
    assert s["invalid"] and s["duplicate"] and not s["all_same"] and not s["ok"]
    s = aa.team_stats(LIST3, rec(a, b, "Tokyo"))
    assert s["invalid"] and not s["duplicate"]
    s = aa.team_stats(OPEN4, rec("Chicago", "chicago", "Boston", "Miami"))  # standardized
    assert s["duplicate"] and not s["in_order"]
