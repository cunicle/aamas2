"""Experiment B: did the n agents of a team pick n different allowed cities?

Per model x protocol x variant (list / open) x temperature, over the teams of
scripts/run_agents.py (cities standardized with `standardize_string`, as in
scripts/choose_analysis.py):
  ok          every agent has an allowed city and all n differ
  duplicate   two agents with the same city (among the agents whose output parsed)
  all_same    every agent parsed and all n picked the same city
  invalid     a city outside the allowed set, or an output that did not parse
  in_order    list variant: agent i takes the i-th listed city, for every i
Then the same per n, the one-canvas rows of results/summary/choose.csv next to them
(--canvas), and two example teams per protocol and model (--examples): the n agents'
user texts and outputs. As in choose_analysis.py, `invalid` in the open variant measures
the one-token slot ("New", "LA"), so only its duplicate rates are informative.

  python scripts/agents_analysis.py results/qwen/agents.jsonl results/dream/agents.jsonl \
      --csv results/summary/agents.csv --canvas results/summary/choose.csv \
      --examples results/summary/agents_examples.md
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_jsonl  # noqa: E402
from ptcdiag.data.agents import PROTOCOLS  # noqa: E402
from ptcdiag.eval.bfcl_checker import standardize_string  # noqa: E402
from ptcdiag.prompting import system_prompt  # noqa: E402

KEYS = ["ok", "duplicate", "all_same", "invalid", "in_order"]


def team_stats(ex, rec):
    n = ex.meta["n"]
    allowed = {standardize_string(c) for c in ex.ground_truth[0]["get_weather"]["city"]}
    vals = [standardize_string(a["city"]) if a["syntax_ok"] and isinstance(a["city"], str) else None
            for a in rec["agents"]]
    parsed = [v for v in vals if v is not None]
    valid = len(vals) == n and all(v in allowed for v in vals)
    dup = len(set(parsed)) < len(parsed)
    listed = [standardize_string(c) for c in ex.meta.get("listed", [])]
    return {"ok": valid and not dup, "duplicate": dup,
            "all_same": len(parsed) == n and len(set(parsed)) == 1, "invalid": not valid,
            "in_order": bool(listed) and vals == listed[:n]}


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        out.append("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) and c != "temperature"
                                     else str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def aggregate(groups, exs):
    rows = []
    for key, rs in sorted(groups.items()):
        st = [team_stats(exs[r["id"]], r) for r in rs]
        row = dict(key)
        row["teams"] = len(rs)
        row.update({k: sum(s[k] for s in st) / len(st) for k in KEYS})
        if row["variant"] != "list":
            row["in_order"] = "-"
        rows.append(row)
    return rows


def protocol_rank(p):
    return list(PROTOCOLS).index(p) if p in PROTOCOLS else len(PROTOCOLS)


def examples_md(recs, exs, per_variant=1):
    """Per model and protocol, the first list team and the first open team with n=3
    (the same items for every protocol and model, so they can be compared)."""
    fns = next(iter(exs.values())).functions
    out = ["# Experiment B: example teams", "",
           "Every agent gets the same system message (ptcdiag/prompting.py):", "", "```text",
           system_prompt(fns).rstrip("\n"), "```", "",
           "Per model and protocol, the first list item and the first open item with n=3 "
           "(T=0 is greedy). `user` is the agent's user message, `output` its decoded text.", ""]
    picked = defaultdict(list)
    for r in recs:
        if r["n"] != 3 or r["seed"] != 0:
            continue
        key = (r["model"], r["protocol"], r["temperature"])
        if sum(x["variant"] == r["variant"] for x in picked[key]) < per_variant:
            picked[key].append(r)
    for (m, p, t), rs in sorted(picked.items(), key=lambda kv: (kv[0][0], kv[0][2], protocol_rank(kv[0][1]))):
        for r in sorted(rs, key=lambda r: r["variant"]):
            st = team_stats(exs[r["id"]], r)
            flags = ", ".join(k for k in KEYS if st[k] is True) or "-"
            out += [f"## {m} / {p} / T={t} / {r['id']} ({r['variant']}, n={r['n']})", "",
                    f"cities: {r['cities']}  ({flags})", ""]
            for a in r["agents"]:
                out += [f"**agent {a['i']}** user:", "", "```text", a["user_text"], "```", "",
                        "output:", "", "```text", a["text"], "```", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default="data/choose.jsonl")
    ap.add_argument("--csv")
    ap.add_argument("--canvas", help="choose.csv of the one-canvas runs, shown next to the teams")
    ap.add_argument("--examples", help="markdown file for the example teams")
    args = ap.parse_args()

    exs = {e.id: e for e in load_jsonl(args.data)}
    recs, errors = [], defaultdict(int)
    for path in args.results:
        if not os.path.exists(path):
            print(f"(missing {path})", file=sys.stderr)
            continue
        with open(path) as f:
            for r in map(json.loads, f):
                r["model"] = r["model_id"].split("/")[-1]
                if "error" in r:
                    errors[r["model"], r["protocol"]] += 1
                elif r["id"] in exs:
                    recs.append(r)

    by_cell, by_n = defaultdict(list), defaultdict(list)
    for r in recs:
        cell = (("model", r["model"]), ("protocol", r["protocol"]), ("variant", r["variant"]),
                ("temperature", r["temperature"]))
        by_cell[cell].append(r)
        by_n[cell + (("n", r["n"]),)].append(r)

    def order(rows):
        return sorted(rows, key=lambda r: (r["model"], r["temperature"], protocol_rank(r["protocol"]),
                                           r["variant"], r.get("n", 0)))

    rows = order(aggregate(by_cell, exs))
    rows_n = order(aggregate(by_n, exs))
    cols = ["model", "protocol", "variant", "temperature", "teams"] + KEYS
    print("# Experiment B: teams of n agents on choose-N\n")
    print("Team level. ok: all cities allowed and different; duplicate: two agents with the same "
          "city; all_same: all n agents with one city; invalid: a city outside the allowed set or "
          "an unparsed output; in_order: agent i took the i-th listed city (list only). In the open "
          "variant `invalid` reflects the one-token slot; read its duplicate rates only.\n")
    print("## All teams\n")
    print(table(rows, cols))
    print("\n## By team size n\n")
    print(table(rows_n, cols[:4] + ["n"] + cols[4:]))
    if errors:
        print("\nTeams that raised an exception (left out above): "
              + ", ".join(f"{m} {p}: {c}" for (m, p), c in sorted(errors.items())))
    if args.canvas and os.path.exists(args.canvas):
        with open(args.canvas) as f:
            canvas = list(csv.DictReader(f))
        ccols = ["model", "cfg", "variant", "n", "ok", "duplicate", "same_step", "invalid", "in_order"]
        print(f"\n## For comparison: one canvas for all n calls (from {args.canvas}; n = items)\n")
        print(table([{c: (f"{float(r[c]):.3f}" if c in ccols[4:] else r[c]) for c in ccols} for r in canvas],
                    ccols))
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:  # n = "all": every team size together
            w = csv.DictWriter(f, fieldnames=cols[:4] + ["n"] + cols[4:])
            w.writeheader()
            w.writerows([{**r, "n": "all"} for r in rows] + rows_n)
    if args.examples:
        os.makedirs(os.path.dirname(args.examples) or ".", exist_ok=True)
        with open(args.examples, "w", encoding="utf-8") as f:
            f.write(examples_md(recs, exs) + "\n")


if __name__ == "__main__":
    main()
