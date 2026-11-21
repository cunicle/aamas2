"""Quick read-out of experiment F (EXP_F_PROMPT.md): simultaneous agents whose role is worded in
other ways, next to the original wordings and to the agents that get their role otherwise.

  # the old team records, as scripts/pod/exp_f/gate_f.sh unpacks them (agents.jsonl, agents_c1.jsonl,
  # agents_rule.jsonl, agents_d.jsonl, agents_pos.jsonl of both models, from release/)
  python scripts/exp_f_quick.py --results results > results/summary/exp_f_quick.md

Per model, one row per wording of the number (sim-label) and of the rule (sim-rule), by the
functions of experiments B and C (scripts/agents_bfcl_analysis.py, scripts/agents_analysis.py):
  BFCL, the team-symmetric requests   set accuracy with a 95% Wilson interval, duplicate,
      in order, first mention (agents of a sibling group that make the group's first reference
      call), unparsed; the difference in set accuracy to the original wording and to the same
      model's position agents (pos-anon), each with a 95% paired-bootstrap interval over requests
      and an exact McNemar test (scripts/stats.py); Holm's correction over the tests against the
      position agents.
  choose-N, the list requests   ok, duplicate with its Wilson interval, all same, in order, the
      share of agents that take the city listed first; the difference in duplicates to the
      original wording.
A condition with no records is printed as TBD.
"""

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agents_analysis import team_stats  # noqa: E402
from agents_bfcl_analysis import team_calls, team_metrics  # noqa: E402
from paper_tables import fmt  # noqa: E402
from ptcdiag.data import load_examples, load_jsonl  # noqa: E402
from ptcdiag.data.agents import WORDINGS  # noqa: E402
from stats import holm, paired, pfmt, wilson  # noqa: E402

MODELS = [("dream", "Dream"), ("qwen", "Qwen2.5")]
CUES = [("number", "sim-label"), ("rule", "sim-rule")]


def rows_spec(original_files):
    """(cue, wording, file, protocol) of every row: per cue the original wording, then experiment
    F's; then the reference rows. original_files: {protocol: file} of the earlier runs."""
    spec = []
    for cue, proto in CUES:
        spec.append((cue, "original", original_files[proto], proto))
        spec += [(cue, w.split("@")[1], original_files["new"], w) for w in WORDINGS if w.startswith(proto + "@")]
    spec.append(("position", "(reference)", original_files["pos-anon"], "pos-anon"))
    spec.append(("turns, numbered", "(reference)", original_files["turn-label"], "turn-label"))
    return spec


BFCL_FILES = {"sim-label": "agents_c1.jsonl", "sim-rule": "agents_c1.jsonl", "turn-label": "agents_c1.jsonl",
              "pos-anon": "agents_d.jsonl", "new": "agents_f1.jsonl"}
CHOOSE_FILES = {"sim-label": "agents.jsonl", "sim-rule": "agents_rule.jsonl", "turn-label": "agents.jsonl",
                "pos-anon": "agents_pos.jsonl", "new": "agents_f_choose.jsonl"}


def records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(line) for line in f if line.strip()) if "error" not in r]


def share(k, n):
    return fmt(Fraction(100 * k, n)) if n else "nan"


def interval(k, n):
    lo, hi = wilson(k, n)
    return f"{share(k, n)} [{fmt(lo)}, {fmt(hi)}]"


def diff_cell(a, b):
    """a - b in points with its paired-bootstrap interval and McNemar p; (cell, p) or ("TBD", None)
    when the two conditions do not cover the same requests."""
    if not a or not b or set(a) != set(b):
        return "TBD", None
    s = paired(a, b)
    return f"{fmt(s['diff'])} [{fmt(s['lo'])}, {fmt(s['hi'])}], p={pfmt(s['p'])}", s["p"]


def bfcl_section(results, exs):
    out = ["## BFCL: the team-symmetric requests (oracle slot lengths, greedy)", "",
           "set acc: the team's n calls are correct as a set, with a 95% Wilson interval. dup: a duplicated "
           "call. in order: agent i makes reference call i, every i. first: agents of a sibling group that make "
           "the group's first reference call. Differences are in points of set accuracy, with a 95% "
           "paired-bootstrap interval and the exact McNemar p.", ""]
    family = []  # (model, row index in `lines`, p) of the tests against the position agents
    for tag, name in MODELS:
        groups = {}
        for cue, wording, fname, proto in rows_spec(BFCL_FILES):
            mets = {}
            for r in records(f"{results}/{tag}/{fname}"):
                if r["protocol"] == proto and r.get("subset") == "sym" and r.get("length_mode") == "oracle" \
                        and r["id"] in exs:
                    calls, unparsed = team_calls(r)
                    mets[r["id"]] = team_metrics(exs[r["id"]], calls, r["team"]["diagnosis"], unparsed)
            groups[cue, wording] = mets
        pos = {i: m["set_acc"] for i, m in groups["position", "(reference)"].items()}
        out += [f"### {name}", "",
                "| role cue | wording | teams | set acc % | dup % | in order % | first % | unparsed % "
                "| minus original | minus position agents |", "|---|---|---|---|---|---|---|---|---|---|"]
        for (cue, wording), mets in groups.items():
            if not mets:
                out.append(f"| {cue} | {wording} | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
                continue
            n = len(mets)
            acc = {i: m["set_acc"] for i, m in mets.items()}
            k = lambda key: sum(m[key] for m in mets.values())
            fm_num = sum(m["first_mention"][0] for m in mets.values())
            fm_den = sum(m["first_mention"][1] for m in mets.values())
            new = wording not in ("original", "(reference)")
            d_orig = diff_cell(acc, {i: m["set_acc"] for i, m in groups[cue, "original"].items()})[0] if new else "-"
            d_pos, p = diff_cell(acc, pos) if cue in ("number", "rule") else ("-", None)
            if new and p is not None:
                family.append((len(out), p))
            out.append(f"| {cue} | {wording} | {n} | {interval(k('set_acc'), n)} | {share(k('duplicate'), n)} "
                       f"| {share(k('in_order'), n)} | {share(fm_num, fm_den)} | {share(k('unparsed'), n)} "
                       f"| {d_orig} | {d_pos} |")
        out.append("")
        for cue, _ in CUES:
            accs = [(100 * sum(m["set_acc"] for m in mets.values()) / len(mets), w)
                    for (c, w), mets in groups.items() if c == cue and mets]
            if len(accs) == 4:
                lo, hi = min(accs), max(accs)
                out.append(f"{name}, {cue}: set accuracy over the four wordings from {lo[0]:.1f}% ({lo[1]}) "
                           f"to {hi[0]:.1f}% ({hi[1]}).")
            else:
                out.append(f"{name}, {cue}: TBD ({len(accs)} of 4 wordings have records).")
        out.append("")
    if family:
        adj = holm([p for _, p in family])
        for (row, _), a in zip(family, adj):
            out[row] = out[row][:-2] + f", Holm p={pfmt(a)} |"
        out += [f"Holm's correction is over the {len(family)} tests of a new wording against the same model's "
                "position agents.", ""]
    return out


def choose_section(results, exs):
    out = ["## Choose-N: the 60 list requests (greedy)", "",
           "ok: all cities allowed and different. dup: two agents take one city, with a 95% Wilson interval. "
           "all same: every agent takes one city. in order: agent i takes the i-th listed city. first listed: "
           "share of agents that take the city listed first. The difference is in points of dup.", ""]
    for tag, name in MODELS:
        groups = {}
        for cue, wording, fname, proto in rows_spec(CHOOSE_FILES):
            recs = {}
            for r in records(f"{results}/{tag}/{fname}"):
                if r["protocol"] == proto and r["variant"] == "list" and r["temperature"] == 0.0 \
                        and r["id"] in exs:
                    recs[r["id"]] = r
            groups[cue, wording] = recs
        out += [f"### {name}", "",
                "| role cue | wording | teams | ok % | dup % | all same % | in order % | first listed % "
                "| dup minus original |", "|---|---|---|---|---|---|---|---|---|"]
        for (cue, wording), recs in groups.items():
            if not recs:
                out.append(f"| {cue} | {wording} | 0 | TBD | TBD | TBD | TBD | TBD | TBD |")
                continue
            st = {i: team_stats(exs[i], r) for i, r in recs.items()}
            n = len(st)
            k = lambda key: sum(s[key] for s in st.values())
            agents = sum(len(r["cities"]) for r in recs.values())
            first = sum(c == r["listed"][0] for r in recs.values() for c in r["cities"])
            new = wording not in ("original", "(reference)")
            orig = {i: team_stats(exs[i], r)["duplicate"] for i, r in groups[cue, "original"].items()} if new else {}
            d = diff_cell({i: s["duplicate"] for i, s in st.items()}, orig)[0] if new else "-"
            out.append(f"| {cue} | {wording} | {n} | {share(k('ok'), n)} | {interval(k('duplicate'), n)} "
                       f"| {share(k('all_same'), n)} | {share(k('in_order'), n)} | {share(first, agents)} | {d} |")
        out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--choose-data", default="data/choose.jsonl")
    a = ap.parse_args()
    exs = {e.id: e for e in load_examples(a.data, a.bfcl_dir)}
    choose_exs = {e.id: e for e in load_jsonl(a.choose_data)}
    out = ["# Experiment F: quick read-out", "",
           "Simultaneous agents that get their role in the prompt, as a number (sim-label) or as a number and "
           "the stated rule (sim-rule), each in its original wording (experiments B and C) and in three others "
           "(ptcdiag/data/agents.py, WORDINGS). Reference rows: position agents (experiment D) and numbered "
           "agents that take turns (experiments B and C).", ""]
    out += bfcl_section(a.results, exs)
    out += choose_section(a.results, choose_exs)
    print("\n".join(out))


if __name__ == "__main__":
    main()
