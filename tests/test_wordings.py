"""Experiment F: other wordings of sim-label and sim-rule (ptcdiag/data/agents.py, WORDINGS)."""

import pytest

from ptcdiag.data import choose
from ptcdiag.data.agents import (PROTOCOLS, RULE_PROTOCOLS, WORDINGS, agent_example, base_protocol,
                                 protocol_flags)
from ptcdiag.data.agents_bfcl import bfcl_agent_example
from ptcdiag.types import Example

ITEMS = choose.generate()
LIST3 = next(e for e in ITEMS if e.meta["variant"] == "list" and e.meta["n"] == 3)
REQUEST = LIST3.messages[0]["content"]


def text(protocol, i, n=3, ex=LIST3):
    return agent_example(ex, i, n, protocol).messages[0]["content"]


def test_wording_names_and_flags():
    assert list(WORDINGS) == ["sim-label@para", "sim-label@ordinal", "sim-label@first",
                              "sim-rule@para", "sim-rule@direct", "sim-rule@first"]
    for w in WORDINGS:
        assert base_protocol(w) in ("sim-label", "sim-rule")
        assert protocol_flags(w) == (False, True)
    assert base_protocol("turn-label") == "turn-label"
    with pytest.raises(KeyError):
        protocol_flags("sim-label@other")


def test_wording_texts_verbatim():
    assert [text(w, 2).replace(REQUEST, "<request>") for w in WORDINGS] == [
        "<request>\n\nThis request is handled by 3 assistants working in parallel, and you are number 2. "
        "Every assistant issues exactly one of the 3 calls, and no assistant can see what the others issue. "
        "Issue your one call.",
        "<request>\n\nYou are the second of 3 assistants answering this request at the same time. Each "
        "assistant makes exactly one of the 3 calls, and the assistants cannot see each other's calls. Make "
        "your one call.",
        "You are assistant 2 of 3. The 3 assistants answer the following request at the same time. Each "
        "assistant makes exactly one of the 3 calls, and the assistants cannot see each other's calls."
        "\n\n<request>\n\nMake your one call.",
        "<request>\n\nThis request is handled by 3 assistants working in parallel, and you are number 2. "
        "Every assistant issues exactly one of the 3 calls, and no assistant can see what the others issue. "
        "The assistants share one rule: the calls are taken in the order in which the request mentions them, "
        "so number 2 issues the second one. Issue your one call.",
        "<request>\n\nYou are assistant 2 of 3 answering this request at the same time. Each assistant makes "
        "exactly one of the 3 calls, and the assistants cannot see each other's calls. Make only the second "
        "of the calls that the request asks for, in the order in which it mentions them; the other assistants "
        "make the others.",
        "You are assistant 2 of 3. The 3 assistants answer the following request at the same time. Each "
        "assistant makes exactly one of the 3 calls, and the assistants cannot see each other's calls. By "
        "convention, assistant 2 makes the second of the calls, in the order in which the request mentions "
        "them.\n\n<request>\n\nMake your one call.",
    ]


@pytest.mark.parametrize("w", list(WORDINGS))
def test_wording_holds_the_request_once_and_differs_between_agents(w):
    texts = [text(w, i) for i in (1, 2, 3)]
    assert all(t.count(REQUEST) == 1 for t in texts)
    assert len(set(texts)) == 3  # the role is in the text: no two agents get the same one
    assert ("first" in texts[0]) == (w != "sim-label@para" and w != "sim-label@first")


def test_original_texts_unchanged_by_the_wordings():
    for p in [*PROTOCOLS, *RULE_PROTOCOLS]:
        observe = protocol_flags(p)[0]
        t = agent_example(LIST3, 2, 3, p, [{"name": "get_weather", "arguments": {"city": "Paris"}}] if observe else ())
        assert t.messages[0]["content"].startswith(REQUEST + "\n\nYou are ")


def test_wordings_refuse_earlier_calls():
    with pytest.raises(ValueError):
        agent_example(LIST3, 2, 3, "sim-rule@direct", [{"name": "get_weather", "arguments": {"city": "Paris"}}])


def test_bfcl_agent_of_a_wording_takes_the_ith_call():
    f = {"name": "g", "description": "", "parameters": {"type": "dict", "properties": {"x": {"type": "string"}},
                                                        "required": ["x"]}}
    ex = Example(id="toy", category="parallel", messages=[{"role": "user", "content": "Do g for a and for b."}],
                 functions=[f], ground_truth=[{"g": {"x": ["a"]}}, {"g": {"x": ["b"]}}], meta={})
    for w in WORDINGS:
        a = bfcl_agent_example(ex, 2, 2, w)
        base = bfcl_agent_example(ex, 2, 2, base_protocol(w))
        assert a.ground_truth == base.ground_truth == [{"g": {"x": ["b"]}}]
        assert a.meta["protocol"] == w and a.meta["call"] == 1 and a.id == "toy_a2"
        assert "Do g for a and for b." in a.messages[0]["content"]
