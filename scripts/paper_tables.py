"""LaTeX tables for the paper, from the raw result records and the summary tables.

Rates are computed from the raw records (counts, not the rounded summary values) where
the records carry what is needed: set accuracy and the cross-call error rate (CCER)
from each record's diagnosis, forward passes from `nfe`. Quantities that need a
tokenizer (length-symmetric items, mention order) come from results/summary/symmetry_*.md,
which scripts/symmetry.py wrote from the same records.

  tar xzf release/results_2026-10-03.tar.gz      # raw records into results/
  python scripts/paper_tables.py --results results --out paper/tables
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.eval.taxonomy import is_cross_call  # noqa: E402

DREAM_K = {k: f"confidence_k{k}_tnone_bfull_T0.0" for k in (1, 2, 4, 8, 16)}
LLADA_K = {k: f"confidence_k{k}_tnone_b32_T0.0" for k in (1, 2, 4, 8, 16)}
DREAM_LTR = "left_to_right_k1_tnone_bfull_T0.0"
DREAM_TAU = "confidence_k1_t0.9_bfull_T0.0"


def exact_runs(paths):
    """{cfg_tag: [records]} of the skeleton runs with exact (oracle, surplus 0) slot lengths."""
    runs = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for r in map(json.loads, f):
                if r.get("mode") == "skeleton" and "diagnosis" in r and not r.get("surplus") \
                        and (r.get("length_mode") or "oracle") == "oracle":
                    runs[r["cfg_tag"]].append(r)
    return runs


def stats(rs):
    """Exact percentages (Fractions), so that rounding happens once, half up, in fmt."""
    n = len(rs)
    return {"n": n,
            "acc": Fraction(100 * sum(bool(r["diagnosis"]["correct"]) for r in rs), n),
            "ccer": Fraction(100 * sum(is_cross_call(r["diagnosis"]["labels"]) for r in rs), n),
            "nfe": Fraction(sum(r["nfe"] for r in rs), n)}


def md_tables(path):
    """Every markdown table in a file, as a list of row dicts per table."""
    tables, cols, rows = [], None, []
    with open(path) as f:
        for line in list(f) + [""]:
            line = line.strip()
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cols is None:
                    cols = cells
                elif not set(line) <= set("|-"):
                    rows.append(dict(zip(cols, cells)))
            elif cols is not None:
                tables.append(rows)
                cols, rows = None, []
    return tables


def symmetry(path):
    """From scripts/symmetry.py output: {cfg: {"n", "acc", "ccer"}} on the equal-length requests and
    {cfg: (n, identity-order %)} on the equal-length `parallel` requests, counting outputs that do not
    parse or leave a call unmatched as not in order. Rates are turned back into exact counts."""
    sym, order = {}, {}
    for rows in md_tables(path):
        for r in rows:
            if "ccer sym" in r:
                n = int(r["n sym"])
                sym[r["cfg"]] = {"n": n, "acc": pct(r["set_acc sym"], n), "ccer": pct(r["ccer sym"], n)}
            if "identity order over all" in r:
                n = int(r["all symmetric parallel items"])
                order[r["cfg"]] = (n, pct(r["identity order over all"], n))
    return sym, order


def fmt(v, d=1):
    """Round half up (89.75 -> 89.8, 86.25 -> 86.3); summary-table floats go through str()."""
    if v is None:
        return "--"
    if isinstance(v, Fraction):
        v = Decimal(v.numerator) / Decimal(v.denominator)
    elif not isinstance(v, Decimal):
        v = Decimal(str(v))
    return str(v.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table_ksweep(res, summ):
    dream = exact_runs([f"{res}/dream/bfcl_skel_k.jsonl", f"{res}/dream/bfcl_skel_ltr.jsonl",
                        f"{res}/dream/bfcl_skel_tau.jsonl"])
    llada = exact_runs([f"{res}/llada2/bfcl_skel_k.jsonl"])
    qwen = exact_runs([f"{res}/qwen/bfcl_skeleton.jsonl"])
    d_sym, d_ord = symmetry(f"{summ}/symmetry_dream.md")
    l_sym, l_ord = symmetry(f"{summ}/symmetry_llada2.md")

    d_cfgs = [DREAM_K[k] for k in (1, 2, 4, 8, 16)] + [DREAM_LTR, DREAM_TAU]
    l_cfgs = [LLADA_K[k] for k in (1, 2, 4, 8, 16)] + [None, None]
    d = {c: stats(dream[c]) for c in d_cfgs}
    l_ = {c: stats(llada[c]) for c in l_cfgs if c}
    q = stats(qwen["ar_greedy"])
    for s in list(d.values()) + list(l_.values()):
        assert s["n"] == 400, s

    def row(label, model, cfgs, get):
        cells = [get(model, c) if c else None for c in cfgs]
        return f"{label} & " + " & ".join(fmt(v) if not isinstance(v, str) else v for v in cells) + r" \\"

    nsym_d, nsym_l = d_sym[DREAM_K[1]]["n"], l_sym[LLADA_K[1]]["n"]
    nord_d, nord_l = d_ord[DREAM_K[1]][0], l_ord[LLADA_K[1]][0]
    lines = [
        r"\begin{table}[t]",
        r"\caption{One canvas, ordered requests, exact slot lengths (400 BFCL requests; set accuracy and \ccer in \%). "
        rf"(eq.): the equal-length requests ({nsym_d} for \dream, {nsym_l} for \llada). Order: share of the "
        rf"equal-length \texttt{{parallel}} requests ({nord_d}, {nord_l}) whose $i$-th call takes the $i$-th mentioned "
        r"entity (outputs that do not parse count as out of order). Fwd: forward passes per request. LTR: left to "
        r"right; $\tau$: threshold 0.9; --: not run. "
        rf"\qwen (AR) in the same skeleton: {fmt(q['acc'])}\% set accuracy, {fmt(q['ccer'])}\% \ccer.}}",
        r"\label{tab:ksweep}",
        r"\small\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & $k{=}1$ & 2 & 4 & 8 & 16 & LTR & $\tau$ \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{\dream (full attention)}} \\",
        row("Set acc.", d, d_cfgs, lambda m, c: m[c]["acc"]),
        row(r"\ccer", d, d_cfgs, lambda m, c: m[c]["ccer"]),
        row("Set acc. (eq.)", d, d_cfgs, lambda m, c: d_sym[c]["acc"] if c in d_sym else None),
        row("Order", d, d_cfgs, lambda m, c: d_ord[c][1] if c in d_ord else None),
        row("Fwd", d, d_cfgs, lambda m, c: m[c]["nfe"]),
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{\llada (32-token blocks)}} \\",
        row("Set acc.", l_, l_cfgs, lambda m, c: m[c]["acc"]),
        row(r"\ccer", l_, l_cfgs, lambda m, c: m[c]["ccer"]),
        row("Set acc. (eq.)", l_, l_cfgs, lambda m, c: l_sym[c]["acc"] if c in l_sym else None),
        row("Order", l_, l_cfgs, lambda m, c: l_ord[c][1] if c in l_ord else None),
        row("Fwd", l_, l_cfgs, lambda m, c: m[c]["nfe"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def pct(rate, n):
    """An exact percentage from a rate that the summaries store as count / n."""
    return Fraction(100 * round(float(rate) * int(n)), int(n))


def csv_rows(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def table_swap(summ):
    """Swapped slots only, against the same slots with exact lengths (scripts/slot_errors.py --swap-slots)."""
    rows = []
    for model, tag, label, cfgs in [("dream", "\\dream", "dream", [("oracle", 1), ("swap", 1), ("swap", 4), ("swap", 16)]),
                                    ("llada2", "\\llada", "llada2", [("oracle", 1), ("swap", 1), ("swap", 4)])]:
        recs = {(r["length_mode"], int(re.search(r"_k(\d+)_", r["cfg"]).group(1))): r
                for r in csv_rows(f"{summ}/swap_{model}.csv") if r["slots"] == "all"}
        first = True
        for mode, k in cfgs:
            r = recs[mode, k]
            n, items = int(r["n_slots"]), int(r["n_items"])
            other = pct(float(r["other"]) + float(r["unparsed"]) + float(r["sibling"]), n)
            name = (f"{tag}, " if first else "") + ("exact" if mode == "oracle" else "swap") + f", $k{{=}}{k}$"
            cells = [pct(r["own"], n), pct(r["sibling_fit"], n), pct(r["overfill"], n), pct(r["truncated"], n),
                     other, pct(r["set_acc"], items), pct(r["ccer"], items)]
            rows.append(name + " & " + " & ".join(fmt(c) for c in cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    rows.pop()
    d = {r["length_mode"]: r for r in csv_rows(f"{summ}/swap_dream.csv") if r["slots"] == "all" and "_k1_" in r["cfg"]}
    l_ = {r["length_mode"]: r for r in csv_rows(f"{summ}/swap_llada2.csv") if r["slots"] == "all" and "_k1_" in r["cfg"]}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Swapped slot lengths. Only the slots whose length the swap changes "
        rf"({d['swap']['n_slots']} slots in {d['swap']['n_items']} requests for \dream, "
        rf"{l_['swap']['n_slots']} in {l_['swap']['n_items']} for \llada), next to the same slots with exact lengths. "
        r"Slot shares in \%: the slot's own value, a sibling's value whose length equals the slot's, "
        r"the own value followed by more, a prefix of it, anything else or nothing. "
        r"Set accuracy and \ccer in \% of the same requests.}",
        r"\label{tab:swap}",
        r"\small\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & Own & Sibling & Over- & Trun- & Other & Set & \ccer \\",
        r" & & that fits & fill & cated & & acc. & \\",
        r"\midrule",
    ] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_teams(summ, res, cres):
    """One canvas against teams of agents (one call each) on ordered and symmetric requests.

    Ordered: set accuracy on the 104 BFCL requests whose sibling calls share one single-call
    skeleton (experiment C1; agents_bfcl.csv, plus the canvas threshold run on the same
    requests). Symmetric: choose-N duplicates and list order (choose.csv, choose_sample.csv,
    choose_tau_ltr.csv for the canvas; agents.csv and the sim-rule rows of agents_bfcl.csv for
    the teams), greedy and at T=0.7 over five seeds."""
    canvas = {}
    for f in ("choose.csv", "choose_sample.csv", "choose_tau_ltr.csv"):
        canvas.update({(r["model"], r["cfg"], r["variant"]): r for r in csv_rows(f"{summ}/{f}")})
    teams = {(r["model"], r["protocol"], r["variant"], float(r["temperature"])): r
             for r in csv_rows(f"{summ}/agents.csv") if r["n"] == "all"}
    bf = csv_rows(f"{summ}/agents_bfcl.csv")
    for r in bf:
        if r["table"] == "choose_rule":
            teams[r["model"], "sim-rule", "list", 0.0] = dict(r, n="all")
    ordered = {(r["model"], r["protocol"]): r for r in bf
               if r["table"] in ("team", "team_canvas") and r["subset"] == "sym" and r["lengths"] == "oracle"}
    D, L, Q = "Dream-v0-Instruct-7B", "LLaDA2.0-mini", "Qwen2.5-7B-Instruct"

    ids = set()
    with open(f"{cres}/dream/agents_c1.jsonl") as f:
        ids = {json.loads(line)["id"] for line in f}
    tau = [r for r in runs_by([f"{res}/dream/bfcl_skel_tau.jsonl"])["oracle", 0, 0.0, DREAM_TAU] if r["id"] in ids]
    assert len(tau) == len(ids) == 104, (len(tau), len(ids))
    tau_acc = Fraction(100 * sum(bool(r["diagnosis"]["correct"]) for r in tau), len(tau))

    def c(r, col, n_col):
        return fmt(pct(r[col], r[n_col])) if r and r.get(col) not in (None, "", "-", "nan") else "--"

    def line(label, ordered_acc, greedy, sample):
        cells = [ordered_acc, c(greedy.get("list"), "duplicate", "n"), c(greedy.get("list"), "in_order", "n"),
                 c(greedy.get("open"), "duplicate", "n"),
                 c(sample.get("list"), "duplicate", "n"), c(sample.get("open"), "duplicate", "n")]
        return f"{label} & " + " & ".join(cells) + r" \\"

    def cv(model, cfg):
        return {v: canvas.get((model, cfg, v)) for v in ("list", "open")}

    def oa(key):
        r = ordered.get(key)
        return fmt(pct(r["set_acc"], r["teams"])) if r else "--"

    rows = [r"\multicolumn{7}{@{}l}{\emph{One model writes all $n$ calls}} \\",
            line(r"\dream, $k{=}16$", oa(("Dream canvas k=16", "one canvas")),
                 cv(D, "confidence_k16_tnone_bfull_T0.0"), cv(D, "confidence_k16_tnone_bfull_T0.7")),
            line(r"\dream, $\tau{=}0.9$", fmt(tau_acc), cv(D, "confidence_k1_t0.9_bfull_T0.0"), {}),
            line(r"\dream, $k{=}1$", oa(("Dream canvas k=1", "one canvas")),
                 cv(D, "confidence_k1_tnone_bfull_T0.0"), cv(D, "confidence_k1_tnone_bfull_T0.7")),
            line(r"\llada, $k{=}16$", oa(("LLaDA2.0 canvas k=16", "one canvas")), cv(L, "confidence_k16_tnone_b32_T0.0"), {}),
            line(r"\llada, $k{=}1$", oa(("LLaDA2.0 canvas k=1", "one canvas")), cv(L, "confidence_k1_tnone_b32_T0.0"), {}),
            line(r"\qwen (AR)", oa(("Qwen canvas (AR)", "one canvas")), cv(Q, "ar_greedy"), {})]
    names = {"sim-anon": "same time, anonymous", "sim-label": "same time, numbered",
             "sim-rule": "same time, told the rule", "turn-anon": "turns, anonymous", "turn-label": "turns, numbered"}
    for model, tag in [(Q, r"\qwen"), (D, r"\dream")]:
        rows += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{A team of $n$ {tag} agents, one call each}}}} \\"]
        for p, name in names.items():
            g = {v: (dict(teams[model, p, v, 0.0], n=teams[model, p, v, 0.0]["teams"])
                     if (model, p, v, 0.0) in teams else None) for v in ("list", "open")}
            smp = {v: (dict(teams[model, p, v, 0.7], n=teams[model, p, v, 0.7]["teams"])
                       if (model, p, v, 0.7) in teams else None) for v in ("list", "open")}
            rows.append(line(r"\quad " + name, oa((model, p)), g, smp))
    nl = teams[(D, "sim-anon", "list", 0.0)]["teams"]
    no = teams[(D, "sim-anon", "open", 0.0)]["teams"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{One canvas against teams of agents. Ordered: set accuracy (\%) on the "
        rf"{len(ids)} team-symmetric BFCL requests. Symmetric (choose-N): "
        r"Dup.\ is the share of requests (\%) in which two calls take the same city; Order, the share of list "
        r"requests whose $i$-th call takes the $i$-th listed city. Agents act at the same time or in turns "
        r"and are anonymous, numbered (``assistant $i$ of $n$''), or numbered and "
        r"told that assistant $i$ makes the $i$-th call; \dream agents decode one token per step. Greedy decoding on "
        rf"{nl} list and {no} open requests; $T{{=}}0.7$: sampling at temperature 0.7, five seeds. Open requests "
        r"hold one-token slots, so only duplicates are reported; --: not run.}",
        r"\label{tab:teams}",
        r"\small\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & Ordered & \multicolumn{3}{c}{Symmetric, greedy} & \multicolumn{2}{c}{$T{=}0.7$} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(l){6-7}",
        r" & Set & \multicolumn{2}{c}{List} & Open & List & Open \\",
        r"\cmidrule(lr){3-4}",
        r" & acc. & Dup. & Order & Dup. & Dup. & Dup. \\",
        r"\midrule",
    ] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_length(summ):
    """Slot length as a signal: what slots of the wrong length hold, on one canvas with the closing
    token after the slot (the main interface) or inside it (experiment C3), and for single Dream
    agents (experiment C2). Slot values are read from each slot's own tokens (closer.csv
    *_read_alone; agents_bfcl.csv slots), set accuracy from the parsed outputs."""
    cl = csv_rows(f"{summ}/closer.csv")
    items = {r["lengths"]: r for r in cl if r["table"] == "items"}

    def get(table, lengths, interface):
        (r,) = [r for r in cl if r["table"] == table and r.get("lengths", "") == lengths
                and r["interface"] == interface]
        return r

    def cells(own, over, sib, n_slots, acc):
        own, over, sib = (pct(v, n_slots) if v is not None else None for v in (own, over, sib))
        rest = 100 - own - (over or 0) - sib
        return [fmt(own), fmt(over) if over is not None else "--", fmt(sib), fmt(rest), fmt(acc)]

    rows = []
    for interface, title in [("original", "One canvas, closing token after the slot"),
                             ("variant", "One canvas, closing token in the slot")]:
        rows.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for lengths, label, table in [("exact k=1", "exact", "slots_read_alone"),
                                      ("+1 k=1", "$+1$", "slots_read_alone"),
                                      ("swap k=1", "swap", "swap_slots_read_alone")]:
            r = get(table, lengths, interface)
            it = items[lengths]
            acc = pct(it[f"set_acc_{interface}"], it[f"n_{interface}"])
            rows.append(f"\\quad {label} & " + " & ".join(cells(r["own"], r["overfill"], r["sibling_fit"],
                                                                r["n_slots"], acc)) + r" \\")
        r = [r for r in cl if r["table"] == "onesided_read_alone" and r["interface"] == interface][0]
        it = items["onesided k=1"]
        acc = pct(it[f"set_acc_{interface}"], it[f"n_{interface}"])
        rows.append(r"\quad one slot longer & " + " & ".join(cells(r["own"], r["overfill"], r["sibling_fit"],
                                                                 r["n_slots"], acc)) + r" \\")
        rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\emph{One \dream agent per call (the slots the swap changes)}} \\")
    bf = csv_rows(f"{summ}/agents_bfcl.csv")
    team = {(r["lengths"], r["protocol"]): r for r in bf if r["table"] == "team" and r["subset"] == "swap"
            and r["model"].startswith("Dream")}
    slots = {(r["lengths"], r["protocol"]): r for r in bf if r["table"] == "slots" and r["dir"] == "all"
             and r["model"].startswith("Dream")}
    for lengths, proto, label in [("oracle", "sim-anon", "same time, exact"), ("swap", "sim-anon", "same time, swap"),
                                  ("swap", "turn-anon", "turns, swap")]:
        r, t = slots[lengths, proto], team[lengths, proto]
        rows.append(f"\\quad {label} & " + " & ".join(cells(r["own"], None, r["sibling_fit"], r["n_slots"],
                                                            pct(t["set_acc"], t["teams"]))) + r" \\")
    n_all = get("slots_read_alone", "exact k=1", "original")["n_slots"]
    n_sw = get("swap_slots_read_alone", "swap k=1", "original")["n_slots"]
    n_one = [r for r in cl if r["table"] == "onesided_read_alone"][0]["n_slots"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Slot length steers values (\dream, one token per step). Share of slots (\%) that hold their own "
        r"value, their own value followed by more (overfill), the value of a sibling whose length fits the slot, "
        rf"or anything else: over all {int(n_all):,} slots (exact, $+1$), the {n_sw} slots whose length the swap changes, "
        rf"and the {n_one} slots lengthened to a sibling's length while the sibling keeps its exact slot (one slot "
        r"longer). Each slot's value is read from its own tokens. Set accuracy (\%) on the 400 requests, or on "
        r"the 165 whose slots the swap changes. Agents are anonymous; for them, overfill is counted under other (--).}",
        r"\label{tab:length}",
        r"\small\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & Own & Over- & Sibling & Other & Set \\",
        r" & & fill & that fits & & acc. \\",
        r"\midrule",
    ] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


MASQ_KEEP = ["exact, k=16", "estimate, k=1", "estimate, k=16", "surplus +1, k=1", "surplus +8, k=1", "swap, k=1"]


def table_masquerade(summ):
    """Cross-call labels: 16 tokens per step with exact lengths vs one token per step with wrong ones."""
    out, model_prev = [], None
    names = {"exact, k=16": "exact, $k{=}16$", "estimate, k=1": "estimate, $k{=}1$",
             "estimate, k=16": "estimate, $k{=}16$",
             "surplus +1, k=1": "$+1$, $k{=}1$", "surplus +2, k=1": "$+2$, $k{=}1$",
             "surplus +4, k=1": "$+4$, $k{=}1$", "surplus +8, k=1": "$+8$, $k{=}1$", "swap, k=1": "swap, $k{=}1$"}
    for r in csv_rows(f"{summ}/masquerade.csv"):
        if r["condition"] not in MASQ_KEEP:
            continue
        model = "\\dream" if r["model"].startswith("Dream") else "\\llada"
        if model != model_prev:
            if model_prev:
                out.append(r"\midrule")
            out.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{model}}}}} \\")
            model_prev = model
        n = r["n"]
        cells = [pct(r[c], n) for c in ("duplicate_call", "cross_binding", "chimera_value",
                                         "inconsistent_shared_arg", "ccer")]
        ref = r["condition"].startswith("exact")
        name = names[r["condition"]]
        line = f"{name} & {n} & " + " & ".join(fmt(c) for c in cells) + r" \\"
        out.append(r"\rowcolor{refrow}" + line if ref else line)
    lines = [
        r"\begin{table}[t]",
        r"\caption{What an evaluator sees. Requests (\%) with each kind of cross-call error, when the agent "
        r"commits 16 tokens per step with exact slot lengths (shaded) and when it commits one token per step "
        r"with wrong slot lengths (for estimated lengths also 16), on the same requests. Estimate: lengths "
        r"estimated by the model (Section~\ref{sec:design}). Req.: requests. Dup.: a duplicated "
        r"call; Bound: a value that belongs to another call; Chim.: a string that puts a piece of another call's "
        r"value in place of a piece of its own; Shared: an argument that the reference shares across calls and "
        r"the prediction does not. A request can have several kinds, so \ccer is not their sum.}",
        r"\label{tab:masquerade}",
        r"\small\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Slot lengths & Req. & Dup. & Bound & Chim. & Shared & \ccer \\",
        r"\midrule",
    ] + out + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def runs_by(paths):
    """{(length_mode, surplus, end_bias, cfg_tag): [records]} of the skeleton runs."""
    runs = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for r in map(json.loads, f):
                if r.get("mode") == "skeleton" and "diagnosis" in r:
                    runs[r.get("length_mode") or "oracle", int(r.get("surplus") or 0),
                         float(r.get("end_bias") or 0), r["cfg_tag"]].append(r)
    return runs


def estimate_slots(path):
    """Estimated minus reference length, one entry per slot (scripts/length_estimate.py output)."""
    diffs, items, exact_items = [], set(), 0
    with open(path) as f:
        for r in map(json.loads, f):
            orc = {(c, p): n for c, p, n in r["oracle"]}
            d = [n - orc[c, p] for c, p, n in r["lengths"]]
            diffs += d
            items.add(r["id"])
            exact_items += all(x == 0 for x in d)
    return diffs, len(items), exact_items


def table_mitigation(res):
    """(a) closing bias x surplus, Dream k=4; (b) one-forward length estimate vs exact lengths."""
    d = runs_by([f"{res}/dream/{f}.jsonl" for f in ("bfcl_skel_k", "bfcl_surplus", "bfcl_endbias", "bfcl_estimate")])
    l_ = runs_by([f"{res}/llada2/{f}.jsonl" for f in ("bfcl_skel_k", "bfcl_estimate")])

    def acc(rs):
        return Fraction(100 * sum(bool(r["diagnosis"]["correct"]) for r in rs), len(rs))

    def parses(rs):
        return Fraction(100 * sum(bool(r["syntax_ok"]) for r in rs), len(rs))

    k4, betas = DREAM_K[4], (0.0, 2.0, 4.0, 8.0)
    for b in betas:
        assert len(d["oracle", 2, b, k4]) == len(d["oracle", 8, b, k4]) == 400, b
    bias = [label + " & " + " & ".join(fmt(f(d["oracle", s, b, k4])) for b in betas) + r" & & \\"
            for label, s, f in [(r"Set acc., $s{=}2$", 2, acc), (r"Set acc., $s{=}8$", 8, acc),
                                (r"Parses, $s{=}8$", 8, parses)]]
    exact4 = d["oracle", 0, 0.0, k4]

    est_rows = []
    for tag, runs, ks, tags, model in [(r"\dream", d, (1, 4, 16), DREAM_K, "dream"),
                                       (r"\llada", l_, (1, 4), LLADA_K, "llada2")]:
        diffs, n_items, _ = estimate_slots(f"{res}/{model}/length_estimate.jsonl")
        n = len(diffs)
        slot = [Fraction(100 * sum(x == 0 for x in diffs), n), Fraction(100 * sum(x > 0 for x in diffs), n),
                Fraction(100 * sum(x < 0 for x in diffs), n)]
        est = {k: runs["length_estimate", 0, 0.0, tags[k]] for k in ks}
        ids = {r["id"] for r in est[1]}
        assert len(ids) == n_items, (model, len(ids), n_items)
        ex = {k: [r for r in runs["oracle", 0, 0.0, tags[k]] if r["id"] in ids] for k in ks}
        cells = [fmt(v) for v in slot] + [fmt(acc(est[k])) if k in est else "--" for k in (1, 4, 16)]
        est_rows.append(f"{tag} ({len(ids)}) & " + " & ".join(cells) + r" \\")
        cells = ["--"] * 3 + [fmt(acc(ex[k])) if k in ex else "--" for k in (1, 4, 16)]
        est_rows.append(r"\rowcolor{refrow}\quad exact lengths & " + " & ".join(cells) + r" \\")
    lines = [
        r"\begin{table}[t]",
        r"\caption{Coping with unknown value lengths. (a) A logit bias $\beta$ toward the closing and padding "
        rf"tokens of every slot, \dream at $k{{=}}4$ with $s$ surplus masks per slot (exact lengths: "
        rf"{fmt(acc(exact4))}\% set accuracy). Parses: share of requests whose output parses. (b) Slot lengths that the model "
        r"predicts in one forward pass: share of slots whose estimate is exact, too long, or too short, and set "
        r"accuracy when the agent decodes with the estimates, next to exact lengths on the same requests (shaded); "
        r"in parentheses, the number of requests. All values in \%; --: not run.}",
        r"\label{tab:mitigation}",
        r"\small\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\multicolumn{7}{@{}l}{\emph{(a) Closing bias, \dream, $k{=}4$}} \\",
        r"$\beta$ & 0 & 2 & 4 & 8 & & \\",
        r"\cmidrule(lr){2-5}",
    ] + bias + [
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{(b) Length estimate from one forward pass}} \\",
        r" & \multicolumn{3}{c}{Slots} & \multicolumn{3}{c}{Set acc.} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r" & Exact & Long & Short & $k{=}1$ & 4 & 16 \\",
        r"\midrule",
    ] + est_rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def estimate_errors(res):
    """How far the one-forward estimates miss, per model. Markdown, for the numbers quoted in the text."""
    out = ["| model | slots | requests | requests all exact | exact | +1 | +2 | +3..7 | >=+8 | -1 | <=-2 | "
           "median overshoot |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in ("dream", "llada2"):
        diffs, n_items, exact_items = estimate_slots(f"{res}/{model}/length_estimate.jsonl")
        n = len(diffs)
        bins = [sum(x == 0 for x in diffs), sum(x == 1 for x in diffs), sum(x == 2 for x in diffs),
                sum(3 <= x <= 7 for x in diffs), sum(x >= 8 for x in diffs), sum(x == -1 for x in diffs),
                sum(x <= -2 for x in diffs)]
        over = sorted(x for x in diffs if x > 0)
        med = (over[(len(over) - 1) // 2] + over[len(over) // 2]) / 2
        out.append(f"| {model} | {n} | {n_items} | {exact_items} ({fmt(Fraction(100 * exact_items, n_items))}%) | "
                   + " | ".join(f"{fmt(Fraction(100 * b, n))}%" for b in bins) + f" | {med:g} |")
    return "\n".join(out) + "\n"


def kcost_breakdown(res):
    """Why requests fail at k=1 and k=16 (exact lengths): output that does not parse, a
    cross-call error, or single-call errors only. Markdown, for the numbers quoted in the text."""
    out = ["| model | k | requests | wrong | does not parse | cross-call | single-call only |",
           "|---|---|---|---|---|---|---|"]
    for model, tags in [("dream", DREAM_K), ("llada2", LLADA_K)]:
        runs = exact_runs([f"{res}/{model}/bfcl_skel_k.jsonl"])
        for k in (1, 16):
            rs = runs[tags[k]]
            wrong = [r for r in rs if not r["diagnosis"]["correct"]]
            syn = sum(not r["syntax_ok"] for r in wrong)
            cc = sum(r["syntax_ok"] and is_cross_call(r["diagnosis"]["labels"]) for r in wrong)
            out.append(f"| {model} | {k} | {len(rs)} | {len(wrong)} | {syn} | {cc} | {len(wrong) - syn - cc} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--c-results", default=None, help="directory of the experiment C records (default: --results)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in [("ksweep", lambda: table_ksweep(args.results, args.summary)),
                     ("teams", lambda: table_teams(args.summary, args.results, args.c_results or args.results)),
                     ("length", lambda: table_length(args.summary)),
                     ("masquerade", lambda: table_masquerade(args.summary)),
                     ("mitigation", lambda: table_mitigation(args.results))]:
        with open(f"{args.out}/{name}.tex", "w") as f:
            f.write(fn())
        print(f"wrote {args.out}/{name}.tex")
    for name, fn in [("kcost", kcost_breakdown), ("estimate", estimate_errors)]:
        with open(f"{args.out}/{name}.md", "w") as f:
            f.write(fn(args.results))
        print(f"wrote {args.out}/{name}.md")


if __name__ == "__main__":
    main()
