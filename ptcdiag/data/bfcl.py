"""Download and load BFCL v4 single-turn Python categories."""

import json
import os
import urllib.request

from ptcdiag.types import Example

BASE_URL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
            "berkeley-function-call-leaderboard/bfcl_eval/data")

CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]

PARALLEL_CATEGORIES = ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]


def _file(category):
    return f"BFCL_v4_{category}.json"


def download(cache_dir, categories=CATEGORIES):
    os.makedirs(os.path.join(cache_dir, "possible_answer"), exist_ok=True)
    for c in categories:
        for sub in ("", "possible_answer/"):
            dst = os.path.join(cache_dir, sub, _file(c))
            if not os.path.exists(dst):
                urllib.request.urlretrieve(f"{BASE_URL}/{sub}{_file(c)}", dst)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(category, cache_dir):
    questions = _read_jsonl(os.path.join(cache_dir, _file(category)))
    answers = {a["id"]: a["ground_truth"]
               for a in _read_jsonl(os.path.join(cache_dir, "possible_answer", _file(category)))}
    examples = []
    for q in questions:
        if q["id"] not in answers:
            continue
        # BFCL stores a list of turns; the single-turn categories have exactly one.
        messages = q["question"][0]
        examples.append(Example(
            id=q["id"],
            category=category,
            messages=messages,
            functions=q["function"],
            ground_truth=answers[q["id"]],
            meta={"source": "bfcl_v4"},
        ))
    return examples
