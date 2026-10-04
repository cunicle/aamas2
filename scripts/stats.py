"""Uncertainty for the paper's main comparisons, from the raw result records (CPU).

Paired comparisons (the same requests under two conditions): the difference of the two
rates (A - B, in points) with a 95% paired-bootstrap interval over requests (10,000
resamples, seed 0) and an exact McNemar test (two-sided binomial test on the discordant
requests). Single rates: 95% Wilson interval. Rates are set accuracy (acc), the cross-call
error rate (ccer) and, on choose-N, the share of requests with a duplicated city (dup).

  tar xzf release/results_2026-10-03.tar.gz; tar xzf release/agents_2026-10-03.tar.gz
  python scripts/stats.py --results results --out results/summary/stats.md
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from paper_tables import DREAM_K, DREAM_TAU, LLADA_K, fmt, runs_by  # noqa: E402
from choose_analysis import item_stats  # noqa: E402
from agents_analysis import team_stats  # noqa: E402
from ptcdiag.data import load_jsonl  # noqa: E402
from ptcdiag.eval.taxonomy import is_cross_call  # noqa: E402

METRIC = {"acc": lambda r: bool(r["diagnosis"]["correct"]),
          "ccer": lambda r: is_cross_call(r["diagnosis"]["labels"])}


def paired(a, b, resamples=10000, seed=0):
    """a, b: {id: bool} on the same requests -> stats of A - B in points."""
    ids = sorted(set(a) & set(b))
    assert ids and len(ids) == len(a) == len(b), (len(ids), len(a), len(b))
    x = np.array([a[i] for i in ids], float)
    y = np.array([b[i] for i in ids], float)
    d = x - y
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(1)
    only_a, only_b = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"n": len(ids), "a": 100 * x.mean(), "b": 100 * y.mean(), "diff": 100 * d.mean(),
            "lo": 100 * np.percentile(boot, 2.5), "hi": 100 * np.percentile(boot, 97.5),
            "only_a": only_a, "only_b": only_b, "p": p}


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def pfmt(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--agents-results", default=None, help="directory of the experiment B records "
                    "(default: --results)")
    ap.add_argument("--choose", default="data/choose.jsonl")
    ap.add_argument("--c-results", default=None, help="directory of the experiment C records "
                    "(default: --results)")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res, ares = args.results, args.agents_results or args.results

    def runs(model, files):
        return runs_by([f"{res}/{model}/{f}.jsonl" for f in files if os.path.exists(f"{res}/{model}/{f}.jsonl")])

    d = runs("dream", ["bfcl_skel_k", "bfcl_skel_tau", "bfcl_surplus", "bfcl_swap", "bfcl_estimate", "bfcl_endbias"])
    l_ = runs("llada2", ["bfcl_skel_k", "bfcl_surplus", "bfcl_swap", "bfcl_estimate"])

    def cond(runs_, mode, s, tag, metric, beta=0.0, ids=None):
        rs = runs_[mode, s, beta, tag]
        out = {r["id"]: METRIC[metric](r) for r in rs}
        return {i: v for i, v in out.items() if ids is None or i in ids}

    lines = ["| section | model | metric | A | B | n | A % | B % | A - B | 95% CI | A only / B only | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def add(sec, model, metric, name_a, a, name_b, b):
        s = paired(a, b)
        lines.append(f"| {sec} | {model} | {metric} | {name_a} | {name_b} | {s['n']} | {fmt(s['a'])} | {fmt(s['b'])} "
                     f"| {fmt(s['diff'])} | [{fmt(s['lo'])}, {fmt(s['hi'])}] | {s['only_a']} / {s['only_b']} "
                     f"| {pfmt(s['p'])} |")

    for name, runs_, K, sub_mode in [("Dream", d, DREAM_K, None), ("LLaDA2.0", l_, LLADA_K, None)]:
        k1, k16 = K[1], K[16]
        add("4", name, "acc", "exact k=1", cond(runs_, "oracle", 0, k1, "acc"), "exact k=16", cond(runs_, "oracle", 0, k16, "acc"))
        add("4", name, "ccer", "exact k=1", cond(runs_, "oracle", 0, k1, "ccer"), "exact k=16", cond(runs_, "oracle", 0, k16, "ccer"))
        if name == "Dream":
            add("4", name, "acc", "exact k=1", cond(runs_, "oracle", 0, k1, "acc"), "exact tau=0.9",
                cond(runs_, "oracle", 0, DREAM_TAU, "acc"))
        sub = set(cond(runs_, "oracle", 1, k1, "acc"))  # the requests of the surplus runs (all 400 or the subset)
        add("6", name, "acc", "exact k=1", cond(runs_, "oracle", 0, k1, "acc", ids=sub), "+1 k=1", cond(runs_, "oracle", 1, k1, "acc"))
        add("6", name, "acc", "+1 k=1", cond(runs_, "oracle", 1, k1, "acc"), "+1 k=16", cond(runs_, "oracle", 1, k16, "acc"))
        for label, mode, s in [("estimate k=1", "length_estimate", 0), ("+1 k=1", "oracle", 1), ("+8 k=1", "oracle", 8),
                               ("swap k=1", "swap", 0)]:
            a = cond(runs_, mode, s, k1, "ccer")
            add("7", name, "ccer", label, a, "exact k=16", cond(runs_, "oracle", 0, k16, "ccer", ids=set(a)))
        est = cond(runs_, "length_estimate", 0, k1, "acc")
        add("8", name, "acc", "exact k=1", cond(runs_, "oracle", 0, k1, "acc", ids=set(est)), "estimate k=1", est)
    k4 = DREAM_K[4]
    add("8", "Dream", "acc", "+8 k=4 beta=2", cond(d, "oracle", 8, k4, "acc", beta=2.0), "+8 k=4 beta=0",
        cond(d, "oracle", 8, k4, "acc"))
    add("8", "Dream", "acc", "+2 k=4 beta=8", cond(d, "oracle", 2, k4, "acc", beta=8.0), "+2 k=4 beta=0",
        cond(d, "oracle", 2, k4, "acc"))

    # choose-N: one canvas and teams of agents, duplicates per request (greedy runs)
    exs = {e.id: e for e in load_jsonl(args.choose)}

    def canvas(model, tag, variant):
        out = {}
        with open(f"{res}/{model}/choose.jsonl") as f:
            for r in map(json.loads, f):
                if r.get("cfg_tag") == tag and r["id"] in exs and exs[r["id"]].meta["variant"] == variant:
                    out[r["id"]] = item_stats(exs[r["id"]], r)["duplicate"]
        return out

    def team(model, protocol, variant):
        out = {}
        with open(f"{ares}/{model}/agents.jsonl") as f:
            for r in map(json.loads, f):
                if r["protocol"] == protocol and r["variant"] == variant and float(r["temperature"]) == 0.0:
                    out[r["id"]] = team_stats(exs[r["id"]], r)["duplicate"]
        return out

    for v in ("list", "open"):
        add("5", "Dream", f"dup ({v})", "canvas k=16", canvas("dream", DREAM_K[16], v), "canvas k=1",
            canvas("dream", DREAM_K[1], v))
    add("5", "Dream", "dup (list)", "team same time, numbered", team("dream", "sim-label", "list"),
        "canvas k=16", canvas("dream", DREAM_K[16], "list"))
    add("5", "Dream", "dup (list)", "team same time, anonymous", team("dream", "sim-anon", "list"),
        "team same time, numbered", team("dream", "sim-label", "list"))
    for m in ("qwen", "dream"):
        add("5", m.capitalize(), "dup (list)", "team same time, numbered", team(m, "sim-label", "list"),
            "team turns, numbered", team(m, "turn-label", "list"))

    # experiment C: teams of agents against one canvas on the same BFCL requests (set accuracy)
    cres = args.c_results or res

    def teams(model, fname, protocol, mode="oracle"):
        out = {}
        path = f"{cres}/{model}/{fname}.jsonl"
        if not os.path.exists(path):
            return out
        with open(path) as f:
            for r in map(json.loads, f):
                if r["protocol"] == protocol and (r.get("length_mode") or "oracle") == mode:
                    out[r["id"]] = bool(r["team"]["diagnosis"]["correct"])
        return out

    c1 = teams("dream", "agents_c1", "sim-label")
    if c1:
        ids = set(c1)
        k16 = cond(d, "oracle", 0, DREAM_K[16], "acc", ids=ids)
        k1 = cond(d, "oracle", 0, DREAM_K[1], "acc", ids=ids)
        add("4C", "Dream", "acc", "canvas k=16", k16, "team same time, numbered", c1)
        add("4C", "Dream", "acc", "team same time, rule", teams("dream", "agents_c1", "sim-rule"),
            "team same time, numbered", c1)
        add("4C", "Dream", "acc", "canvas k=16", k16, "team same time, rule", teams("dream", "agents_c1", "sim-rule"))
        add("4C", "Dream", "acc", "canvas k=1", k1, "team turns, numbered", teams("dream", "agents_c1", "turn-label"))
        add("4C", "Dream", "acc", "canvas k=1", k1, "team turns, anonymous", teams("dream", "agents_c1", "turn-anon"))
        # the team-symmetric requests with one tool: the whole skeleton is n copies of one call's
        # skeleton, so a slot knows no more than a numbered agent (its index among n calls)
        from ptcdiag.data import load_examples
        exs = {e.id: e for e in load_examples("bfcl:parallel,parallel_multiple", args.bfcl_dir)}
        one = {i for i in ids if len({f for f, _ in exs[i].gold_calls}) == 1}
        sub = lambda m: {i: v for i, v in m.items() if i in one}  # noqa: E731
        add("4C", "Dream", "acc", "canvas k=16, one tool", sub(k16), "team same time, numbered, one tool", sub(c1))
        add("4C", "Dream", "acc", "canvas k=16, one tool", sub(k16), "team same time, rule, one tool",
            sub(teams("dream", "agents_c1", "sim-rule")))
        ar = {}
        if os.path.exists(f"{res}/qwen/bfcl_skeleton.jsonl"):
            with open(f"{res}/qwen/bfcl_skeleton.jsonl") as f:
                ar = {r["id"]: bool(r["diagnosis"]["correct"]) for r in map(json.loads, f) if r["id"] in ids}
        if ar:
            for proto, name in [("turn-anon", "anonymous"), ("turn-label", "numbered")]:
                add("4C", "Qwen", "acc", "canvas (AR)", ar, f"team turns, {name}", teams("qwen", "agents_c1", proto))
        add("4C", "Qwen", "acc", "team same time, rule", teams("qwen", "agents_c1", "sim-rule"),
            "team same time, numbered", teams("qwen", "agents_c1", "sim-label"))
        add("6C", "Dream", "acc", "team same time, anonymous, exact (C2)", teams("dream", "agents_c2", "sim-anon"),
            "Qwen team same time, anonymous, exact (C2)", teams("qwen", "agents_c2", "sim-anon"))

    # single rates on LLaDA2.0's subsets: 95% Wilson intervals
    lines += ["", "| model | condition | metric | n | rate % | 95% Wilson interval |", "|---|---|---|---|---|---|"]
    for label, runs_, mode, s, tag, metric in [
            ("exact k=1", l_, "oracle", 0, LLADA_K[1], "acc"), ("+1 k=1", l_, "oracle", 1, LLADA_K[1], "acc"),
            ("estimate k=1", l_, "length_estimate", 0, LLADA_K[1], "acc"), ("+8 k=1", l_, "oracle", 8, LLADA_K[1], "acc"),
            ("swap k=1", l_, "swap", 0, LLADA_K[1], "ccer"), ("estimate k=1", l_, "length_estimate", 0, LLADA_K[1], "ccer")]:
        sub = set(cond(l_, "oracle", 1, LLADA_K[1], "acc"))
        c = cond(runs_, mode, s, tag, metric, ids=sub if mode == "oracle" and s == 0 else None)
        k, n = sum(c.values()), len(c)
        lo, hi = wilson(k, n)
        lines.append(f"| LLaDA2.0 | {label} | {metric} | {n} | {fmt(100 * k / n)} | [{fmt(lo)}, {fmt(hi)}] |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write("# Uncertainty for the main comparisons (scripts/stats.py)\n\n" + text)


if __name__ == "__main__":
    main()
