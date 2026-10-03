"""Choose-N probes: every city slot has the same length, and duplicates are counted."""

from ptcdiag.data import choose
from ptcdiag.decoding.constraints import build_skeleton


def test_generate_unique_and_balanced():
    exs = choose.generate()
    assert len({e.messages[0]["content"] for e in exs}) == len(exs) == 105
    for e in exs:
        assert len(e.ground_truth) == e.meta["n"]
        if e.meta["variant"] == "list":
            assert len(e.meta["listed"]) == e.meta["n"] + 3 and set(e.meta["listed"]) <= set(choose.ONE_TOKEN)


def test_city_slots_are_one_token(dream_tokenizer):
    tok = dream_tokenizer
    assert all(len(tok(c, add_special_tokens=False)["input_ids"]) == 1 for c in choose.ONE_TOKEN)
    for e in choose.generate()[:30] + choose.generate()[-15:]:
        _, slots = build_skeleton(tok, e, tok.mask_token_id)
        assert [len(s.positions) for s in slots] == [1] * e.meta["n"]


def test_choose_stats():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "choose_analysis.py")
    spec = importlib.util.spec_from_file_location("choose_analysis", path)
    ca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ca)
    ex = next(e for e in choose.generate() if e.meta["variant"] == "list" and e.meta["n"] == 2)
    a, b = ex.meta["listed"][:2]

    def rec(*cities, steps=()):
        return {"syntax_ok": True, "calls": [{"name": "get_weather", "arguments": {"city": c}} for c in cities],
                "trace": {"steps": [{"tokens": t} for t in steps]}}

    assert ca.item_stats(ex, rec(a, b)) == {"ok": True, "duplicate": False, "same_step": False,
                                            "invalid": False, "in_order": True}
    s = ca.item_stats(ex, rec(a, a, steps=[[7, 7]]))
    assert s["duplicate"] and s["same_step"] and not s["ok"]
    s = ca.item_stats(ex, rec(a, a, steps=[[7], [7]]))
    assert s["duplicate"] and not s["same_step"]
    assert ca.item_stats(ex, rec(a, "Tokyo"))["invalid"]
