"""Figures for the paper, from the summary tables (CPU, no model).

  surplus.pdf  (a, b) set accuracy against the surplus masks per slot, one line per k,
               for Dream (400 requests) and LLaDA2.0 (its 100-request subset; surplus 0
               on the same subset), with the autoregressive agent as a reference;
               (c) the teacher-forced probe: mean probability of a closing token right
               after the reference value, with s masks left in the slot.

Colors: hue = model (Dream blue, LLaDA2.0 orange), lightness = k (darkest = one token
per step); marker shape repeats k so the figure survives grayscale print. Both ramps
pass the data-viz skill's ordinal checks (monotone lightness, visible steps, light end
>= 2:1 on white).

  python scripts/paper_figures.py --summary results/summary --out paper/figures
"""

import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

S = [0, 1, 2, 4, 8]
KS = [1, 4, 16]
RAMP = {"dream": ["#0d366b", "#2a78d6", "#86b6ef"],   # blue 700 / 450 / 250
        "llada2": ["#9a3a10", "#eb6834", "#f29a6f"]}  # orange, same three steps
MARKER = {1: "o", 4: "s", 16: "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL = {"dream": "Dream-v0-Instruct-7B", "llada2": "LLaDA2.0-mini", "qwen": "Qwen2.5-7B-Instruct"}


def set_acc(summary):
    """{(model, s, k): set accuracy %} for the exact-length (oracle) runs without end bias."""
    out = {}
    with open(f"{summary}/length.csv") as f:
        for r in csv.DictReader(f):
            if r["length_mode"] != "oracle" or float(r["end_bias"]):
                continue
            m = next((k for k, v in MODEL.items() if v == r["model"]), None)
            k = int(re.search(r"_k(\d+)_", r["cfg"]).group(1)) if r["cfg"] != "ar_greedy" else 0
            out[m, int(r["surplus"]), k] = 100 * float(r["set_acc"])
    return out


def p_close(summary):
    """{(model, s): mean P(closing token)} from the 'all' rows of the probe tables."""
    out, cur = {}, None
    with open(f"{summary}/length_prior.md") as f:
        for line in f:
            m = re.match(r"\S+/(Dream-v0-Instruct-7B|LLaDA2\.0-mini), surplus (\d+): next token", line)
            if m:
                cur = ("dream" if m.group(1).startswith("Dream") else "llada2", int(m.group(2)))
            elif cur and line.startswith("| all |"):
                out[cur] = float(line.split("|")[4])
                cur = None
    return out


def style(ax, ylabel=None):
    ax.set_xticks(range(len(S)), [str(s) for s in S])
    ax.set_xlim(-0.3, len(S) - 0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5)
    ax.set_xlabel("surplus masks per slot, $s$", color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)


def surplus_figure(summary, out):
    acc, pc = set_acc(summary), p_close(summary)
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.titlesize": 7.5,
                         "pdf.fonttype": 42, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.32})
    for ax, model, title in [(axes[0], "dream", "(a) Dream"),
                             (axes[1], "llada2", "(b) LLaDA2.0 (100 requests)")]:
        handles = []
        for i, k in enumerate(KS):
            ys = [acc.get((model, s, k)) for s in S]
            xs = [x for x, y in zip(range(len(S)), ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if len(xs) < 2:
                continue
            # break the line where a surplus was not run (LLaDA2.0: s=4; k=16 only s<=1)
            seg_x, seg_y = [xs[0]], [ys[0]]
            for x, y in zip(xs[1:], ys[1:]):
                if x != seg_x[-1] + 1:
                    ax.plot(seg_x, seg_y, color=RAMP[model][i], linewidth=1.5)
                    seg_x, seg_y = [], []
                seg_x.append(x)
                seg_y.append(y)
            ax.plot(seg_x, seg_y, color=RAMP[model][i], linewidth=1.5)
            ax.plot(xs, ys, linestyle="none", marker=MARKER[k], markersize=4.2, color=RAMP[model][i],
                    markeredgecolor="white", markeredgewidth=0.6)
            handles.append(Line2D([], [], color=RAMP[model][i], linewidth=1.5, marker=MARKER[k], markersize=4.2,
                                  markeredgecolor="white", markeredgewidth=0.6, label=f"$k={k}$"))
        if model == "dream":
            ar = [(x, acc[("qwen", s, 0)]) for x, s in enumerate(S) if ("qwen", s, 0) in acc]
            ax.plot([x for x, _ in ar], [y for _, y in ar], linestyle="none", marker="D", markersize=3.6,
                    color=MUTED)
            handles.append(Line2D([], [], linestyle="none", marker="D", markersize=3.6, color=MUTED,
                                  label="AR (Qwen2.5)"))
        if model == "llada2":
            ax.text(3, 4, "not run", color=MUTED, ha="center", fontsize=6.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", color=INK)
        style(ax, "set accuracy (%)" if model == "dream" else None)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.6, 0.93), frameon=False, ncol=1, handlelength=1.2,
                  labelcolor=INK, borderaxespad=0)
    ax = axes[2]
    for model, label, mk in [("dream", "Dream", "o"), ("llada2", "LLaDA2.0", "s")]:
        pts = [(x, 100 * pc[model, s]) for x, s in enumerate(S) if (model, s) in pc]
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=RAMP[model][1], linewidth=1.5, marker=mk,
                markersize=4.2, markeredgecolor="white", markeredgewidth=0.6, label=label)
    ax.set_ylim(0, 100)
    ax.set_title("(c) Probe: closing right after the value", loc="left", color=INK)
    style(ax, "P(closing token) (%)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, labelcolor=INK, borderaxespad=0.2)
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/surplus.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out}/surplus.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"wrote {out}/surplus.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    surplus_figure(args.summary, args.out)


if __name__ == "__main__":
    main()
