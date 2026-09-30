import json

from conftest import gold_prediction

from ptcdiag.data.paraprobe import CAPITALS, generate
from ptcdiag.eval.taxonomy import diagnose
from ptcdiag.prompting import build_messages, parse_tool_calls, tokens_in_span


def test_paraprobe_gold_is_correct_and_factors_covered():
    exs = generate(per_cell=2, seed=0)
    assert len(exs) == 2 * 168
    for e in exs:
        d = diagnose(e, parse_tool_calls(json.dumps(gold_prediction(e))))
        assert d.correct and d.bfcl_valid, (e.id, e.meta, d.labels)
        assert len(e.ground_truth) == e.meta["n"]
    for factor in ("n", "entities", "ambiguity", "shared", "mix"):
        assert len({e.meta[factor] for e in exs}) >= 2


def test_paraprobe_indirect_mentions_hide_names():
    for e in generate(per_cell=2, seed=1):
        text = e.messages[0]["content"]
        if e.meta["ambiguity"] == "derived":
            assert not any(c in text for c in e.meta["cities"]), text
            assert any(country in text for country in CAPITALS)
        else:
            assert all(c in text for c in e.meta["cities"]), text


def test_parse_with_spans():
    text = 'noise [{"name": "f", "arguments": {"city": "New York", "n": 3}}, {"name": "g", "arguments": {}}] tail'
    p = parse_tool_calls(text)
    assert p.syntax_ok and len(p.calls) == 2
    s, e = p.value_span(0, "city")
    assert text[s:e] == '"New York"'
    s, e = p.value_span(0, "n")
    assert text[s:e] == "3"
    assert p.value_span(1, "x") is None
    assert p.model_output() == [{"f": {"city": "New York", "n": 3}}, {"g": {}}]


def test_parse_failures():
    for bad in ["", "no json", '[{"name": 1, "arguments": {}}]', '[{"name": "f"}]',
                '[{"name": "f", "arguments": {"a": }}]', '{"name": "f", "arguments": {}}']:
        assert not parse_tool_calls(bad).syntax_ok, bad


def test_tokens_in_span():
    ends = [2, 5, 9, 10]  # tokens cover [0,2) [2,5) [5,9) [9,10)
    assert tokens_in_span(ends, (3, 6)) == [1, 2]
    assert tokens_in_span(ends, (9, 10)) == [3]


def test_system_message_merged():
    from ptcdiag.types import Example

    e = Example("x", "parallel", [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}],
                [{"name": "f", "parameters": {"type": "dict", "properties": {}}}], [])
    msgs = build_messages(e)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"].endswith("be nice") and '"name": "f"' in msgs[0]["content"]
