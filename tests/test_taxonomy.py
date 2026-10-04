import json
import random

from conftest import gold_prediction

from ptcdiag.eval.taxonomy import chimera_kind, diagnose, extends_own, pair_diff
from ptcdiag.prompting import parse_tool_calls
from ptcdiag.types import Example

WEATHER = {
    "name": "get_weather",
    "parameters": {"type": "dict", "required": ["city"], "properties": {
        "city": {"type": "string"}, "unit": {"type": "string"}, "days": {"type": "integer"}}},
}
TIME = {"name": "get_time", "parameters": {"type": "dict", "required": ["city"],
                                           "properties": {"city": {"type": "string"}}}}


def ex(gt, fns=(WEATHER,), cat="parallel"):
    return Example(id="t", category=cat, messages=[], functions=list(fns), ground_truth=gt)


def diag(example, calls):
    return diagnose(example, parse_tool_calls(json.dumps(calls)))


def w(city, unit=None, days=None):
    args = {"city": city}
    if unit is not None:
        args["unit"] = unit
    if days is not None:
        args["days"] = days
    return {"name": "get_weather", "arguments": args}


GT2 = [{"get_weather": {"city": ["Paris"], "days": [3]}},
       {"get_weather": {"city": ["Tokyo"], "days": [5]}}]


def test_correct_any_order():
    e = ex(GT2)
    for calls in ([w("Paris", days=3), w("Tokyo", days=5)], [w("Tokyo", days=5), w("paris", days=3)]):
        d = diag(e, calls)
        assert d.correct and d.bfcl_valid and d.labels == []


def test_duplicate_call():
    d = diag(ex(GT2), [w("Paris", days=3), w("Paris", days=3)])
    assert d.labels == ["duplicate_call"] and not d.bfcl_valid


def test_omitted_and_extra():
    assert diag(ex(GT2), [w("Paris", days=3)]).labels == ["omitted_call"]
    d = diag(ex(GT2), [w("Paris", days=3), w("Tokyo", days=5), w("Rome", days=1)])
    assert "extra_call" in d.labels


def test_collision_vs_swap():
    # collision: both calls carry Paris, days are distinct so they are not duplicates
    d = diag(ex(GT2), [w("Paris", days=3), w("Paris", days=5)])
    assert d.labels == ["cross_binding"]
    assert d.details[0]["kind"] == "collision"
    # swap: days exchanged between the two calls
    d = diag(ex(GT2), [w("Paris", days=5), w("Tokyo", days=3)])
    assert "cross_binding" in d.labels
    assert all(x["kind"] == "swap" for x in d.details if x["label"] == "cross_binding")


def test_chimera():
    gt = [{"get_weather": {"city": ["New York"]}}, {"get_weather": {"city": ["Mexico City"]}}]
    d = diag(ex(gt), [w("New City"), w("Mexico City")])
    assert d.labels == ["chimera_value"]
    assert chimera_kind("New City", "New York", "Mexico City") == "word"
    assert chimera_kind("Tokoto", "Tokyo", "Kyoto") == "char"
    assert chimera_kind("Tokto", "Tokyo", "Kyoto") is None   # a 2-character piece matches by chance
    assert chimera_kind("Tokyo", "Tokyo", "Kyoto") is None
    assert chimera_kind("Berlin", "Tokyo", "Kyoto") is None
    assert chimera_kind("2022-001-01", "2022-01-01", "2022-02-01") is None    # the piece is its own too
    assert chimera_kind("Chicago, CA", "Chicago, IL", "San Francisco, CA") == "word"
    assert chimera_kind("Mar Swift", "Maroon 5", "Taylor Swift") == "char"
    assert chimera_kind("Mauryan", "Persian Empire", "Mauryan Empire") is None  # no piece of its own


def test_truncations_and_overfills_are_not_chimeras():
    assert extends_own("Sothe", "Sotheby") and extends_own("San Francisco CA", "San Francisco")
    assert extends_own("The God of War", "God of War")
    assert not extends_own("New City", "New York")
    gt = [{"get_weather": {"city": ["Boston"]}}, {"get_weather": {"city": ["Atlanta"]}},
          {"get_weather": {"city": ["Massachusetts"]}}]
    d = diag(ex(gt), [w("Boston MA"), w("Atlanta"), w("Massachusetts")])
    assert "chimera_value" not in d.labels and not d.correct


def test_inconsistent_shared_arg():
    gt = [{"get_weather": {"city": ["Paris"], "unit": ["celsius"]}},
          {"get_weather": {"city": ["Tokyo"], "unit": ["celsius"]}}]
    d = diag(ex(gt), [w("Paris", "celsius"), w("Tokyo", "fahrenheit")])
    assert "inconsistent_shared_arg" in d.labels and "wrong_value" in d.labels


def test_single_call_errors():
    gt = [{"get_weather": {"city": ["Paris"], "days": [3]}}, {"get_time": {"city": ["Tokyo"]}}]
    e = ex(gt, fns=(WEATHER, TIME), cat="parallel_multiple")
    d = diag(e, [w("Paris", days=3), {"name": "get_weather", "arguments": {"city": "Tokyo"}}])
    assert d.labels == ["wrong_function"]
    d = diag(e, [w("Paris", days="3"), {"name": "get_time", "arguments": {"city": "Tokyo"}}])
    assert d.labels == ["type_error"]
    d = diag(e, [w("Paris"), {"name": "get_time", "arguments": {"city": "Tokyo"}}])
    assert d.labels == ["missing_param"]
    d = diag(e, [w("Paris", days=3), {"name": "get_time", "arguments": {"city": "Tokyo", "tz": "x"}}])
    assert d.labels == ["unexpected_param"]


def test_syntax_and_no_call():
    e = ex(GT2)
    assert diagnose(e, parse_tool_calls('[{"name": "get_weather", "arguments": {')).labels == ["syntax_error"]
    assert diagnose(e, parse_tool_calls("[]")).labels == ["no_call"]


def test_gold_predictions_pass_on_bfcl(bfcl_examples):
    """Gold (shuffled) predictions: our verdict == official verdict on every item; all
    items of the four main categories pass; a handful of live items are unsatisfiable
    in BFCL itself (gold uses parameters missing from the schema)."""
    rng = random.Random(0)
    for cat, exs in bfcl_examples.items():
        n_valid = n = 0
        for e in exs:
            calls = gold_prediction(e)
            if calls is None:
                continue
            rng.shuffle(calls)
            d = diag(e, calls)
            assert d.correct == d.bfcl_valid, (cat, e.id, d.labels, d.bfcl_error_type)
            n += 1
            n_valid += d.bfcl_valid
        if cat in ("parallel", "parallel_multiple", "simple_python", "multiple"):
            assert n_valid == n, cat
        else:
            assert n_valid >= 0.98 * n, (cat, n_valid, n)


def test_perturbations_on_bfcl_parallel(bfcl_examples):
    """Duplicating / dropping a call is caught and labelled, never marked correct."""
    for e in bfcl_examples["parallel"]:
        calls = gold_prediction(e)
        if calls is None or len(calls) < 2 or pair_diff(e.functions, calls[0], e.gold_calls[1]).exact:
            continue  # the first call also satisfies the second gold call
        dup = [calls[0], calls[0]] + calls[2:]
        d = diag(e, dup)
        assert not d.correct and not d.bfcl_valid
        assert "duplicate_call" in d.labels or "cross_binding" in d.labels, (e.id, d.labels)
        d = diag(e, calls[1:])
        assert d.labels[0] == "omitted_call" and not d.bfcl_valid


def test_set_verdict_matches_bfcl_on_random_corruptions(bfcl_examples):
    rng = random.Random(1)
    n = 0
    for cat in ("parallel", "parallel_multiple", "simple_python", "multiple"):
        for e in bfcl_examples[cat]:
            calls = gold_prediction(e)
            if calls is None:
                continue
            c = rng.randrange(len(calls))
            args = calls[c]["arguments"]
            if args:
                p = rng.choice(sorted(args))
                if isinstance(args[p], str):
                    args[p] = args[p] + " x"
                elif isinstance(args[p], (int, float)) and not isinstance(args[p], bool):
                    args[p] = args[p] + 7
            d = diag(e, calls)
            assert d.correct == d.bfcl_valid, (cat, e.id, d.labels, d.bfcl_error_type)
            n += 1
    assert n > 900
