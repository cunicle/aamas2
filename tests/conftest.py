import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

BFCL_DIR = os.path.join(ROOT, "data", "bfcl")


def _first(value):
    """Concrete value from a BFCL acceptable value (dicts hold lists of acceptable values)."""
    if isinstance(value, dict) and all(isinstance(v, list) for v in value.values()):
        out = {}
        for k, acc in value.items():
            vals = [a for a in acc if a != ""]
            if vals:
                out[k] = _first(vals[0])
        return out
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return [_first(v) for v in value]
    return value


def gold_prediction(example):
    """A prediction built from the first acceptable value of every gold parameter."""
    calls = []
    for name, params in example.gold_calls:
        f = example.function(name)
        required = set(f["parameters"].get("required", [])) if f else set()
        args = {}
        for p, acc in params.items():
            if "" in acc and p not in required:
                continue  # optional and omittable
            vals = [a for a in acc if a != ""]
            if not vals:
                return None  # a few BFCL items are unsatisfiable (required param, no value)
            args[p] = _first(vals[0])
        calls.append({"name": name, "arguments": args})
    return calls


@pytest.fixture(scope="session")
def bfcl_examples():
    from ptcdiag.data import bfcl

    if not os.path.exists(os.path.join(BFCL_DIR, "BFCL_v4_parallel.json")):
        pytest.skip("BFCL data not downloaded (run scripts/prepare_data.py)")
    return {c: bfcl.load(c, BFCL_DIR) for c in bfcl.CATEGORIES}


@pytest.fixture(scope="session")
def dream_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")
