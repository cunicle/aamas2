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
    """{cfg: (n sym, ccer sym %)} and {cfg: identity-order %} from scripts/symmetry.py output."""
    sym, order = {}, {}
    for rows in md_tables(path):
        for r in rows:
            if "ccer sym" in r:
                sym[r["cfg"]] = (int(r["n sym"]), Decimal(r["ccer sym"]) * 100)
            if "identity order" in r:
                order[r["cfg"]] = Decimal(r["identity order"]) * 100
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

    nsym_d = d_sym[DREAM_K[1]][0]
    nsym_l = l_sym[LLADA_K[1]][0]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Committing more tokens per step, with exact slot lengths (BFCL parallel and "
        r"parallel\_multiple, 400 requests). Set accuracy and \ccer in \%; \ccer{} (sym.) on the "
        rf"requests whose slot lengths cannot tell the calls apart ({nsym_d} for \dream, {nsym_l} for "
        r"\llada); order: share of symmetric \texttt{parallel} requests whose $i$-th call took the "
        r"$i$-th mentioned entity; fwd: forward passes per request. LTR: left to right, one token per step; "
        r"$\tau$: commit every position above confidence 0.9. "
        rf"\qwen (autoregressive) in the same skeleton: {fmt(q['acc'])}\% set accuracy, {fmt(q['ccer'])}\% \ccer.}}",
        r"\label{tab:ksweep}",
        r"\small\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & $k{=}1$ & 2 & 4 & 8 & 16 & LTR & $\tau$ \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{\dream (full attention)}} \\",
        row("Set acc.", d, d_cfgs, lambda m, c: m[c]["acc"]),
        row(r"\ccer", d, d_cfgs, lambda m, c: m[c]["ccer"]),
        row(r"\ccer (sym.)", d, d_cfgs, lambda m, c: d_sym[c][1] if c in d_sym else None),
        row("Order", d, d_cfgs, lambda m, c: d_ord.get(c)),
        row("Fwd", d, d_cfgs, lambda m, c: m[c]["nfe"]),
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{\llada (32-token blocks)}} \\",
        row("Set acc.", l_, l_cfgs, lambda m, c: m[c]["acc"]),
        row(r"\ccer", l_, l_cfgs, lambda m, c: m[c]["ccer"]),
        row(r"\ccer (sym.)", l_, l_cfgs, lambda m, c: l_sym[c][1] if c in l_sym else None),
        row("Order", l_, l_cfgs, lambda m, c: l_ord.get(c)),
        row("Fwd", l_, l_cfgs, lambda m, c: m[c]["nfe"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in [("ksweep", table_ksweep)]:
        with open(f"{args.out}/{name}.tex", "w") as f:
            f.write(fn(args.results, args.summary))
        print(f"wrote {args.out}/{name}.tex")
    with open(f"{args.out}/kcost.md", "w") as f:
        f.write(kcost_breakdown(args.results))
    print(f"wrote {args.out}/kcost.md")


if __name__ == "__main__":
    main()
