"""Experiments C1 / C2: tables of the BFCL agent teams, next to the one-canvas runs on the same items.

  python scripts/agents_bfcl_analysis.py --teams results/qwen/agents_c1.jsonl results/dream/agents_c1.jsonl \
      results/qwen/agents_c2.jsonl results/dream/agents_c2.jsonl results/dream/agents_c2_swap.jsonl \
      --rule results/qwen/agents_rule.jsonl results/dream/agents_rule.jsonl \
      --canvas-root <extracted release_2026-10-03>/results --agents-csv results/summary/agents.csv \
      --csv results/summary/agents_bfcl.csv --examples results/summary/agents_bfcl_examples.md

Team level, per model x subset x slot lengths x protocol, and for the one-canvas runs on the same
requests by the same functions (canvas call i = agent i):
  set_acc        the team's output is correct (set-level diagnosis against the whole request)
  ccer           a cross-call error label
  duplicate      the duplicate_call label
  in_order       agent i's call passes reference call i's check (simple_function_checker), every i
  first_mention  agent level, sibling groups only (the calls of a function called more than once):
                 the share of agents whose call passes the check of the group's first reference call
  unparsed       some agent did not make exactly one call (the team is then not correct and has
                 only the syntax_error label)
  order_correct  in_order among the correct teams (in_order implies correct), the measure behind the
                 paper's "slots filled in mention order 95-97%": on the C1 requests the one-canvas
                 runs give 94.6% (Dream k=1), 95.7% (LLaDA2.0 k=1), 95.8% (Qwen)
  in_order_clean C1 only: in_order on the requests whose reference calls follow the request's
                 mention order (all but ORDER_NOISE); every correct one-canvas output there is in order
Slot level (C2): the slots the length swap changes (514 with the Dream / Qwen tokenizer); each
agent's value for them is
  own            passes its own reference call's check (value_ok)
  sibling_fit    passes a sibling's check (same function and parameter, another call) whose oracle
                 length equals the slot's length in this condition
  sibling_other  passes another sibling's check, of another length
  other          anything else, missing values and unparsed outputs included
`dir` splits these slots by whether the swap makes them longer or shorter than their own value
(in the oracle-length rows too, so the rows compare the same slots).
The choose-N rows (sim-rule on the list items) use experiment B's team metrics
(scripts/agents_analysis.py), with experiment B's greedy list rows next to them.
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples, load_jsonl  # noqa: E402
from ptcdiag.data.agents import PROTOCOLS, RULE_PROTOCOLS  # noqa: E402
from ptcdiag.data.agents_bfcl import swap_changes, team_symmetric  # noqa: E402
from ptcdiag.decoding.constraints import (lengths_from_list, oracle_lengths, sibling_groups,  # noqa: E402
                                          swapped_lengths)
from ptcdiag.eval.bfcl_checker import simple_function_checker  # noqa: E402
from ptcdiag.eval.taxonomy import is_cross_call, value_ok  # noqa: E402
from ptcdiag.prompting import system_prompt  # noqa: E402

TEAM_KEYS = ["set_acc", "ccer", "duplicate", "in_order", "first_mention", "unparsed"]
ORDER_KEYS = ["order_correct", "in_order_clean"]
SLOT_KINDS = ["own", "sibling_fit", "sibling_other", "other"]
# C1 requests (checked by hand) whose reference order is not the request's mention order: the
# reference lists the calls out of mention order (parallel_14 "10, 20 and 30 years" -> 20, 30, 10;
# parallel_152 "first 3^5, then 2^3" -> 2^3, 3^5; parallel_168 the 2nd and 3rd scenarios swapped;
# parallel_multiple_111 core beliefs "of both these religions" -> Hinduism, Buddhism after
# Buddhism, Hinduism), or the request is a cross product with no single mention order
# (parallel_46, 74, 137, 178, 180). Every correct but out-of-order one-canvas output (Dream k=1 / 16,
# LLaDA2.0 k=1, Qwen) on the 104 requests is one of these.
ORDER_NOISE = {"parallel_14", "parallel_152", "parallel_168", "parallel_multiple_111",
               "parallel_46", "parallel_74", "parallel_137", "parallel_178", "parallel_180"}
ORDER = [*PROTOCOLS, *RULE_PROTOCOLS]
DREAM, QWEN = "Dream-org/Dream-v0-Instruct-7B", "Qwen/Qwen2.5-7B-Instruct"
# one-canvas comparison rows: (label, file under --canvas-root, cfg_tag, length mode)
CANVAS = {
    "sym": [("Dream canvas k=1", "dream/bfcl_skel_k.jsonl", "confidence_k1_tnone_bfull_T0.0", "oracle"),
            ("Dream canvas k=16", "dream/bfcl_skel_k.jsonl", "confidence_k16_tnone_bfull_T0.0", "oracle"),
            ("LLaDA2.0 canvas k=1", "llada2/bfcl_skel_k.jsonl", "confidence_k1_tnone_b32_T0.0", "oracle"),
            ("LLaDA2.0 canvas k=16", "llada2/bfcl_skel_k.jsonl", "confidence_k16_tnone_b32_T0.0", "oracle"),
            ("Qwen canvas (AR)", "qwen/bfcl_skeleton.jsonl", "ar_greedy", "oracle")],
    "swap": [("Dream canvas k=1", "dream/bfcl_skel_k.jsonl", "confidence_k1_tnone_bfull_T0.0", "oracle"),
             ("Dream canvas k=1 swap", "dream/bfcl_swap.jsonl", "confidence_k1_tnone_bfull_T0.0", "swap"),
             ("Qwen canvas (AR)", "qwen/bfcl_skeleton.jsonl", "ar_greedy", "oracle")],
}


def short(model_id):
    return (model_id or "?").split("/")[-1]


def call_ok(ex, ci, call):
    """`call` ({"name", "arguments"} or None) passes reference call ci's check."""
    if call is None:
        return False
    fname, _ = ex.gold_calls[ci]
    try:
        return bool(simple_function_checker(ex.function(fname), {call["name"]: call["arguments"]},
                                            ex.ground_truth[ci])["valid"])
    except Exception:
        return False


def function_groups(ex):
    """{function: [call indices]} for the functions called more than once (the sibling agents)."""
    groups = defaultdict(list)
    for ci, (fname, _) in enumerate(ex.gold_calls):
        groups[fname].append(ci)
    return {f: cis for f, cis in groups.items() if len(cis) > 1}


def team_metrics(ex, calls, diagnosis, unparsed):
    """calls[ci]: agent ci+1's call (the canvas's call ci), None when it made none."""
    labels = diagnosis["labels"]
    fm = [call_ok(ex, cis[0], calls[ci]) for cis in function_groups(ex).values() for ci in cis]
    return {"set_acc": bool(diagnosis["correct"]), "ccer": is_cross_call(labels),
            "duplicate": "duplicate_call" in labels,
            "in_order": all(call_ok(ex, ci, c) for ci, c in enumerate(calls)),
            "first_mention": (sum(fm), len(fm)), "unparsed": bool(unparsed)}


def team_calls(rec):
    """(calls by agent, unparsed) of a team record."""
    return [a["call"] for a in rec["agents"]], not rec["team"]["syntax_ok"]


def canvas_calls(ex, rec):
    """(calls in skeleton order = reference order, unparsed) of a one-canvas record."""
    calls = rec.get("calls") or []
    if not rec.get("syntax_ok") or len(calls) != len(ex.ground_truth):
        return [None] * len(ex.ground_truth), True
    return calls, False


def slot_kind(ex, ci, p, call, slot_len, oracle):
    """own / sibling_fit / sibling_other / other for the value `call` gives parameter p of call ci."""
    fname, params = ex.gold_calls[ci]
    if call is None or call.get("name") != fname or p not in call.get("arguments", {}):
        return "other"
    v, desc = call["arguments"][p], ex.function(fname)
    if value_ok(desc, p, v, params[p]):
        return "own"
    sibs = [cj for cj in sibling_groups(ex).get((fname, p), []) if cj != ci
            and value_ok(desc, p, v, ex.gold_calls[cj][1][p])]
    if any(oracle[cj, p] == slot_len for cj in sibs):
        return "sibling_fit"
    return "sibling_other" if sibs else "other"


def swap_slots(tok, ex):
    """[(call, param, direction)] for the slots the length swap changes."""
    L, S = oracle_lengths(tok, ex), swapped_lengths(tok, ex)
    return [(ci, p, "longer" if S[ci, p] > L[ci, p] else "shorter") for (ci, p) in sorted(L) if S[ci, p] != L[ci, p]]


def mean_row(key, items):
    """items: [(request id, team_metrics)] of one group -> its row (TEAM_KEYS and ORDER_KEYS)."""
    mets = [m for _, m in items]
    row = dict(key)
    row["teams"] = len(mets)
    for k in TEAM_KEYS:
        if k == "first_mention":
            num, den = sum(m[k][0] for m in mets), sum(m[k][1] for m in mets)
            row[k] = num / den if den else float("nan")
        else:
            row[k] = sum(m[k] for m in mets) / len(mets)
    n_correct = sum(m["set_acc"] for m in mets)
    row["order_correct"] = sum(m["in_order"] for m in mets) / n_correct if n_correct else float("nan")
    if dict(key).get("subset") == "sym":
        clean = [m for i, m in items if i not in ORDER_NOISE]
        row["in_order_clean"] = sum(m["in_order"] for m in clean) / len(clean) if clean else float("nan")
    return row


def slot_rows(key, kinds):
    """kinds: [(direction, kind)] -> rows for all slots and per direction."""
    rows = []
    for d in ("all", "longer", "shorter"):
        ks = [k for dd, k in kinds if d == "all" or dd == d]
        if not ks:
            continue
        row = {**dict(key), "dir": d, "n_slots": len(ks)}
        row.update({k: ks.count(k) / len(ks) for k in SLOT_KINDS})
        rows.append(row)
    return rows


def fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    if not path or not os.path.exists(path):
        print(f"(missing {path})", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def protocol_rank(p):
    return ORDER.index(p) if p in ORDER else len(ORDER)


def examples_md(recs, exs, rule_recs, choose_exs):
    """One team per model x subset x slot lengths x protocol: the first request with three calls of
    each subset (the same request for every protocol, model and length mode)."""
    out = ["# Experiment C: example teams", "",
           "Every agent gets the system message of ptcdiag/prompting.py with the request's functions "
           "(shown once per request) and its own user message (`user`). `slots` are the agent's slot "
           "lengths [call, parameter, tokens], `output` its decoded text. Greedy decoding; Dream k=1, "
           "confidence order.", ""]
    first = {}
    for r in sorted(recs, key=lambda r: r["_idx"]):
        if r["n"] == 3 and r["subset"] not in first:
            first[r["subset"]] = r["id"]
    shown = set()
    picked = sorted((r for r in recs if r["id"] == first.get(r["subset"])),
                    key=lambda r: (r["subset"] != "sym", r["length_mode"], r["model"], protocol_rank(r["protocol"])))
    for r in picked:
        ex = exs[r["id"]]
        if r["id"] not in shown:
            shown.add(r["id"])
            out += [f"## Request {r['id']} ({r['subset']} subset, {r['category']}, n={r['n']})", "",
                    "System message:", "", "```text", system_prompt(ex.functions).rstrip("\n"), "```", "",
                    "Reference calls: `" + json.dumps(ex.ground_truth) + "`", ""]
        calls, unparsed = team_calls(r)
        tm = team_metrics(ex, calls, r["team"]["diagnosis"], unparsed)
        flags = ", ".join(k for k in TEAM_KEYS if k != "first_mention" and tm[k]) or "-"
        out += [f"### {r['model']} / {r['subset']} / lengths {r['length_mode']} / {r['protocol']}", "",
                f"team output: `{r['team']['text']}`", "",
                f"labels: {r['team']['diagnosis']['labels'] or ['(none)']}; flags: {flags}", ""]
        for a in r["agents"]:
            out += [f"**agent {a['i']}** slots {a['lengths']}, user:", "", "```text", a["user_text"], "```", "",
                    "output:", "", "```text", a["text"], "```", ""]
    if rule_recs:
        aa = load_script("agents_analysis")
        out += ["## Choose-N, sim-rule (list items)", ""]
        seen = set()
        for r in rule_recs:
            if r["n"] != 3 or r["model"] in seen:
                continue
            seen.add(r["model"])
            st = aa.team_stats(choose_exs[r["id"]], r)
            flags = ", ".join(k for k in aa.KEYS if st[k] is True) or "-"
            out += [f"### {r['model']} / {r['id']} (list, n=3)", "", f"cities: {r['cities']}  ({flags})", ""]
            for a in r["agents"]:
                out += [f"**agent {a['i']}** user:", "", "```text", a["user_text"], "```", "",
                        "output:", "", "```text", a["text"], "```", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", nargs="+", default=[], help="team JSONL of scripts/run_agents_bfcl.py")
    ap.add_argument("--rule", nargs="*", default=[], help="choose-N sim-rule teams of scripts/run_agents.py")
    ap.add_argument("--canvas-root", help="results/ directory of the one-canvas runs (release 2026-10-03)")
    ap.add_argument("--agents-csv", help="experiment B's results/summary/agents.csv")
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--choose-data", default="data/choose.jsonl")
    ap.add_argument("--csv")
    ap.add_argument("--examples")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    exs = {e.id: e for e in load_examples(args.data, args.bfcl_dir)}
    toks = {m: AutoTokenizer.from_pretrained(m, trust_remote_code=True) for m in (DREAM, QWEN)}
    subset_ids = {"sym": {i for i, e in exs.items() if team_symmetric(toks[DREAM], e)},
                  "swap": {i for i, e in exs.items() if swap_changes(toks[DREAM], e)}}
    for m in (QWEN,):  # the two tokenizers give the same subsets (checked on 2026-10-03)
        assert subset_ids["sym"] == {i for i, e in exs.items() if team_symmetric(toks[m], e)}
        assert subset_ids["swap"] == {i for i, e in exs.items() if swap_changes(toks[m], e)}

    recs, errors = [], defaultdict(int)
    for path in args.teams:
        for r in read_jsonl(path):
            r["model"], r["_idx"] = short(r["model_id"]), len(recs)
            if "error" in r:
                errors[r["model"], r["subset"], r["length_mode"], r["protocol"]] += 1
                continue
            assert r["id"] in subset_ids[r["subset"]], (r["id"], r["subset"])
            recs.append(r)

    def tok_of(model):
        return toks[QWEN] if model.startswith("Qwen") else toks[DREAM]

    team_groups, slot_kinds = defaultdict(list), defaultdict(list)
    for r in recs:
        ex = exs[r["id"]]
        key = (("model", r["model"]), ("subset", r["subset"]), ("lengths", r["length_mode"]),
               ("protocol", r["protocol"]))
        calls, unparsed = team_calls(r)
        team_groups[key].append((r["id"], team_metrics(ex, calls, r["team"]["diagnosis"], unparsed)))
        if r["subset"] == "swap":
            L = oracle_lengths(tok_of(r["model"]), ex)
            for ci, p, d in swap_slots(tok_of(r["model"]), ex):
                slot_len = lengths_from_list(r["agents"][ci]["lengths"])[0, p]
                slot_kinds[key].append((d, slot_kind(ex, ci, p, r["agents"][ci]["call"], slot_len, L)))

    canvas_team, canvas_slots = defaultdict(list), defaultdict(list)
    if args.canvas_root:
        cache = {}
        for subset, specs in CANVAS.items():
            for label, rel, tag, lm in specs:
                path = os.path.join(args.canvas_root, rel)
                if path not in cache:
                    cache[path] = read_jsonl(path)
                for r in cache[path]:
                    if r["id"] not in subset_ids[subset] or r.get("cfg_tag") != tag or "diagnosis" not in r \
                            or r.get("mode") != "skeleton" or r.get("length_mode", "oracle") != lm \
                            or r.get("surplus", 0) or r.get("end_bias", 0.0):
                        continue
                    ex = exs[r["id"]]
                    key = (("model", label), ("subset", subset), ("lengths", lm), ("protocol", "one canvas"))
                    calls, unparsed = canvas_calls(ex, r)
                    canvas_team[key].append((r["id"], team_metrics(ex, calls, r["diagnosis"], unparsed)))
                    if subset == "swap" and not label.startswith("LLaDA"):
                        tok = toks[QWEN] if label.startswith("Qwen") else toks[DREAM]
                        L = oracle_lengths(tok, ex)
                        lens = dict(L)
                        lens.update(lengths_from_list(r["lengths"]) if r.get("lengths") else {})
                        for ci, p, d in swap_slots(tok, ex):
                            canvas_slots[key].append((d, slot_kind(ex, ci, p, calls[ci], lens[ci, p], L)))

    def order(rows):
        return sorted(rows, key=lambda r: (r["subset"] != "sym", r["model"], r["lengths"], protocol_rank(r["protocol"])))

    team_rows = order([mean_row(k, v) for k, v in team_groups.items()])
    canvas_rows = [mean_row(k, v) for k, v in canvas_team.items()]
    s_rows = order([row for k, v in slot_kinds.items() for row in slot_rows(k, v)])
    cs_rows = [row for k, v in canvas_slots.items() for row in slot_rows(k, v)]
    tcols = ["model", "lengths", "protocol", "teams"] + TEAM_KEYS[:4] + ORDER_KEYS + TEAM_KEYS[4:]
    tcols2 = [c for c in tcols if c != "in_order_clean"]  # C2: mention order not checked there
    scols = ["model", "lengths", "protocol", "dir", "n_slots"] + SLOT_KINDS
    n_items = {s: len(ids) for s, ids in subset_ids.items()}

    print("# Experiment C1 / C2: teams of agents on the BFCL parallel requests\n")
    print("One agent per reference call, each with its call's single-call skeleton; greedy decoding "
          "(Dream: k=1, confidence order). Team metrics: set_acc (the n calls as one array, set-level "
          "diagnosis), ccer (cross-call error label), duplicate (duplicate_call label), in_order "
          "(agent i passes reference call i, all i), order_correct (in_order among the correct teams: "
          "the paper's 95-97% measure), in_order_clean (C1: in_order on the 95 requests whose reference "
          "order is the request's mention order), first_mention (agents of a sibling group that pass "
          "the group's first reference call), unparsed (some agent did not make exactly one call). "
          "The one-canvas rows are the 10-03 runs on the same requests, by the same functions "
          "(canvas call i = agent i).\n")
    print(f"## C1: order cue, team-symmetric requests ({n_items['sym']} items), oracle slot lengths\n")
    print(table([r for r in team_rows if r["subset"] == "sym"], tcols))
    print("\nOne canvas, same requests:\n")
    print(table([r for r in canvas_rows if r["subset"] == "sym"], tcols))

    aa = load_script("agents_analysis")
    choose_exs = {e.id: e for e in load_jsonl(args.choose_data)} if os.path.exists(args.choose_data) else {}
    rule_recs, rule_rows = [], []
    for path in args.rule:
        for r in read_jsonl(path):
            r["model"] = short(r["model_id"])
            if "error" not in r and r["id"] in choose_exs:
                rule_recs.append(r)
    if rule_recs:
        groups = defaultdict(list)
        for r in rule_recs:
            groups[(("model", r["model"]), ("protocol", r["protocol"]), ("variant", r["variant"]),
                    ("temperature", r["temperature"]))].append(r)
        rule_rows = aa.aggregate(groups, choose_exs)
    b_rows = []
    if args.agents_csv and os.path.exists(args.agents_csv):
        with open(args.agents_csv) as f:
            b_rows = [{**r, **{k: float(r[k]) for k in aa.KEYS if r[k] not in ("-", "")}}
                      for r in csv.DictReader(f) if r["variant"] == "list" and r["n"] == "all"
                      and float(r["temperature"]) == 0.0]
    print("\n### Choose-N positive control: sim-rule on the 60 list items (experiment B's metrics)\n")
    print("ok: all cities allowed and different; duplicate: two agents with one city; in_order: agent i "
          "took the i-th listed city. Experiment B's greedy list rows (results/summary/agents.csv) below "
          "for comparison.\n")
    ccols = ["model", "protocol", "variant", "teams"] + aa.KEYS
    print(table(sorted(rule_rows, key=lambda r: r["model"]) + sorted(b_rows, key=lambda r: (r["model"], protocol_rank(r["protocol"]))),
                ccols) if rule_rows or b_rows else "TBD (no sim-rule teams yet)")

    print(f"\n## C2: length cue, requests where the swap changes a slot ({n_items['swap']} items)\n")
    print(table([r for r in team_rows if r["subset"] == "swap"], tcols2))
    print("\nOne canvas, same requests:\n")
    print(table([r for r in canvas_rows if r["subset"] == "swap"], tcols2))
    print("\n### C2 slot level: the slots the swap changes\n")
    print("own / sibling_fit (a sibling's value of exactly the slot's length) / sibling_other / other "
          "(incl. missing and unparsed); dir: the swap makes the slot longer or shorter than its own "
          "value.\n")
    print(table(s_rows, scols))
    print("\nOne canvas, same slots:\n")
    print(table(cs_rows, scols))
    if errors:
        print("\nTeams that raised an exception (left out above): "
              + ", ".join(f"{'/'.join(map(str, k))}: {c}" for k, c in sorted(errors.items())))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        rows = ([{"table": "team", **r} for r in team_rows] + [{"table": "team_canvas", **r} for r in canvas_rows]
                + [{"table": "slots", **r} for r in s_rows] + [{"table": "slots_canvas", **r} for r in cs_rows]
                + [{"table": "choose_rule", **r} for r in rule_rows] + [{"table": "choose_b", **r} for r in b_rows])
        cols = ["table", "model", "subset", "lengths", "protocol", "variant", "temperature", "teams", "dir",
                "n_slots"] + TEAM_KEYS + ORDER_KEYS + SLOT_KINDS + [k for k in aa.KEYS if k not in TEAM_KEYS]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if args.examples:
        os.makedirs(os.path.dirname(args.examples) or ".", exist_ok=True)
        with open(args.examples, "w", encoding="utf-8") as f:
            f.write(examples_md(recs, exs, rule_recs, choose_exs) + "\n")


if __name__ == "__main__":
    main()
