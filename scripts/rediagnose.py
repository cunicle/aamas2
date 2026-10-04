"""Recompute the stored diagnoses of result records with the current taxonomy, in place.

  python scripts/rediagnose.py results/dream/*.jsonl results/llada2/*.jsonl results/qwen/*.jsonl

The run scripts store `diagnosis` (labels, matching, ...) with each record; the analysis scripts read
it from there. After a change to ptcdiag/eval/taxonomy.py, this re-derives every diagnosis from the
record's own output text (one-canvas records: `text`; BFCL team records: `team.text`) and reports how
many records changed and which labels moved. Records without a BFCL request (choose-N, probes) and
unparsed teams are left as they are. --check only reports.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.eval.taxonomy import diagnose  # noqa: E402
from ptcdiag.prompting import ParsedOutput, parse_tool_calls  # noqa: E402


def redo(ex, text):
    parsed = parse_tool_calls(text) if text is not None else ParsedOutput("", False, error="no text")
    return json.loads(json.dumps(diagnose(ex, parsed).to_dict(), ensure_ascii=False))  # as stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--check", action="store_true", help="report the changes, do not write")
    args = ap.parse_args()

    exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple", args.bfcl_dir)}
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        changed, moved = 0, Counter()
        for r in recs:
            ex = exs.get(r.get("id"))
            if ex is None or "error" in r:
                continue
            if "team" in r:
                if r["team"].get("text") is None:
                    continue
                old, new = r["team"]["diagnosis"], redo(ex, r["team"]["text"])
                r["team"]["diagnosis"] = new
            elif "diagnosis" in r and "text" in r:
                old, new = r["diagnosis"], redo(ex, r["text"])
                r["diagnosis"] = new
            else:
                continue
            if old != new:
                changed += 1
                for lab in set(old["labels"]) ^ set(new["labels"]):
                    moved[("+" if lab in new["labels"] else "-") + lab] += 1
                if old["correct"] != new["correct"]:
                    moved["correct changed"] += 1
        print(f"{path}: {changed} of {len(recs)} records changed {dict(moved)}")
        if changed and not args.check:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, path)


if __name__ == "__main__":
    main()
