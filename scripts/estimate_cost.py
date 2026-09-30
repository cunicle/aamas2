"""Estimate A100 GPU-hours for a run plan (README "算力估计").

Forward-pass counts are exact for fixed-k configs: they are computed from the real
prompts and skeletons (per-block mask counts, absolute blocks for LLaDA2.0). Threshold
configs, error rates and replay counts are assumptions, listed in ASSUMPTIONS.

Timing model per forward pass over a canvas of L tokens:  overhead + L * ms_per_ktok / 1000.
Defaults come from FLOPs (2 * active params per token) at ~40% MFU of an A100's 312 TFLOPs
bf16 peak; LLaDA2.0-mini's MoE is given a pessimistic value because HF MoE kernels are slow.
After the smoke test, replace them with measured values:

  python scripts/estimate_cost.py --plan minimal
  python scripts/estimate_cost.py --plan full --timing dream=110:20 --timing llada2=45:30
"""

import argparse
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import bfcl, load_jsonl  # noqa: E402
from ptcdiag.decoding.constraints import build_skeleton  # noqa: E402
from ptcdiag.prompting import render_prompt  # noqa: E402

DLLM = {  # name: (hf id, ms per 1k canvas tokens, ms overhead per forward, absolute blocks)
    "dream": ("Dream-org/Dream-v0-Instruct-7B", 125, 15, False),
    "llada": ("GSAI-ML/LLaDA-8B-Instruct", 130, 15, False),
    "llada2": ("inclusionAI/LLaDA2.0-mini", 60, 25, True),
}
AR = {  # name: (tokenizer proxy, prefill ms per 1k tokens, ms per decode step)
    "qwen2.5-7b": ("dream", 125, 30),
    "ling-mini-2.0": ("llada2", 60, 35),
    "llama-3.1-8b": ("llada", 130, 30),
}
ASSUMPTIONS = {
    "tokens_per_step_at_threshold": {0.99: 1.5, 0.95: 2.0, 0.9: 2.5, 0.7: 3.5, 0.5: 5.0},
    "free_output_tokens": "skeleton length of the example (proxy for the answer length)",
    "error_rate": 0.5, "labels_per_error": 1.5, "replays_per_label": 3, "placebo_replays": 4,
    "max_errors": 200, "max_placebo": 200,
    "ar_value_tokens_per_slot": 4,
    "model_load_min": 3, "rerun_buffer": 1.3,
}

PLANS = {
    "minimal": {
        "dllms": ["dream", "llada2"],
        "ar": ["qwen2.5-7b", "ling-mini-2.0"],
        "bfcl": ["parallel", "parallel_multiple"],
        "probe_scale": 1,  # data/paraprobe.jsonl as generated with --per-cell 5 (840 items)
        "skeleton_bfcl": [("k", 1, "confidence", None), ("k", 2, "confidence", None),
                          ("k", 4, "confidence", None), ("k", 8, "confidence", None),
                          ("k", 16, "confidence", None), ("k", 1, "left_to_right", None),
                          ("tau", 0.9, "confidence", None)],
        "skeleton_probe": [("k", 1, "confidence", None), ("k", 2, "confidence", None),
                           ("k", 4, "confidence", None), ("k", 8, "confidence", None)],
        "probe_sampling_seeds": 0,
        "free_bfcl": [("k", 2, "confidence", None)],
        "attribution_k": [4], "dvs_records": 150,
        # LLaDA2.0 only: one block over the whole canvas, the block-length contrast (§6.4)
        "block_full_bfcl": [("k", k, "confidence", "full") for k in (1, 4, 16)],
        "block_full_probe": [("k", k, "confidence", "full") for k in (1, 4)],
        "llada_light": False,
        "debug_hours": 3,
    },
    "full": {
        "dllms": ["dream", "llada2", "llada"],
        "ar": ["qwen2.5-7b", "ling-mini-2.0", "llama-3.1-8b"],
        "bfcl": ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"],
        "probe_scale": 2,  # --per-cell 10
        "skeleton_bfcl": [("k", k, "confidence", None) for k in (1, 2, 4, 8, 16)]
                         + [("k", 1, "left_to_right", None), ("k", 4, "left_to_right", None)]
                         + [("tau", t, "confidence", None) for t in (0.99, 0.9, 0.7, 0.5)]
                         + [("k", 4, "confidence", 8), ("k", 4, "confidence", 32)],
        "skeleton_probe": [("k", k, "confidence", None) for k in (1, 2, 4, 8)],
        "probe_sampling_seeds": 3,  # k=8, T=1 for Proposition 1
        "free_bfcl": [("k", 2, "confidence", None)],
        "attribution_k": [4, 8], "dvs_records": 300,
        "block_full_bfcl": [("k", k, "confidence", "full") for k in (1, 4, 16)],
        "block_full_probe": [("k", k, "confidence", "full") for k in (1, 4)],
        "llada_light": True,  # LLaDA-8B: only k in {1,4,16} + free, on BFCL
        "debug_hours": 5,
    },
}


def measure(model, specs):
    """Per example: prompt length, skeleton length, and mask positions (gen-relative)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DLLM[model][0], trust_remote_code=True)
    out = {}
    for name, exs in specs.items():
        rows = []
        for e in exs:
            P = len(tok(render_prompt(tok, e), add_special_tokens=False)["input_ids"])
            gen, slots = build_skeleton(tok, e, -1)
            masks = [p for s in slots for p in s.positions]
            rows.append({"P": P, "S": len(gen), "masks": masks, "n_slots": len(slots)})
        out[name] = rows
    return out


def blocks_of(row, B, absolute):
    """Mask counts per block for block length B (None = one block)."""
    if not B:
        return [len(row["masks"])]
    off = row["P"] if absolute else 0
    counts = defaultdict(int)
    for m in row["masks"]:
        counts[(m + off) // B] += 1
    return list(counts.values())


def steps(row, cfg, absolute):
    kind, val, _order, B = cfg
    if B == "full":
        B = None
    elif absolute and not B:
        B = 32  # LLaDA2.0's native block length
    tps = val if kind == "k" else ASSUMPTIONS["tokens_per_step_at_threshold"][val]
    return sum(math.ceil(c / tps) for c in blocks_of(row, B, absolute))


def fwd_ms(model, L, timing):
    ktok, over = timing.get(model, DLLM[model][1:3])
    return over + L * ktok / 1000


def canvas(row, absolute, mode, B=None):
    S = row["S"] if mode == "skeleton" else row["S"] + 32
    if mode == "free" and not absolute:
        S = 256
    # LLaDA2.0 forwards only up to the current block: on average half the generation region
    return row["P"] + (S / 2 if absolute and B != "full" else S)


def dllm_hours(model, rows_by_set, plan, timing):
    absolute = DLLM[model][3]
    ms = defaultdict(float)
    light = model == "llada" and plan["llada_light"]
    bfcl_cfgs = [("k", k, "confidence", None) for k in (1, 4, 16)] if light else plan["skeleton_bfcl"]

    def run(rows, cfgs, mode, key, reps=1):
        for r in rows:
            for cfg in cfgs:
                L = canvas(r, absolute, mode, cfg[3])
                if mode == "free":
                    n = math.ceil((r["S"] + 32) / cfg[1]) if absolute else math.ceil(256 / cfg[1])
                else:
                    n = steps(r, cfg, absolute)
                ms[key] += reps * n * fwd_ms(model, L, timing)

    run(rows_by_set["bfcl"], bfcl_cfgs, "skeleton", "RQ1-2 BFCL skeleton")
    run(rows_by_set["bfcl"], plan["free_bfcl"], "free", "free-mode control")
    if not light:
        run(rows_by_set["probe"], plan["skeleton_probe"], "skeleton", "RQ3 ParaProbe")
        if plan["probe_sampling_seeds"]:
            run(rows_by_set["probe"], [("k", 8, "confidence", None)], "skeleton",
                "Prop.1 sampling", reps=plan["probe_sampling_seeds"])
        attr_cfgs = [("k", k, "confidence", None) for k in plan["attribution_k"]]
        if absolute:
            run(rows_by_set["bfcl"], plan["block_full_bfcl"], "skeleton", "block-length contrast BFCL")
            run(rows_by_set["probe"], plan["block_full_probe"], "skeleton", "block-length contrast ParaProbe")
            attr_cfgs.append(("k", 4, "confidence", "full"))
        A = ASSUMPTIONS
        rows = rows_by_set["bfcl"]
        for cfg in attr_cfgs:
            k = cfg[1]
            n_err = min(A["max_errors"], int(A["error_rate"] * len(rows)))
            n_plc = min(A["max_placebo"], len(rows) - n_err)
            per = [(0.5 * steps(r, cfg, absolute) + k) * fwd_ms(model, canvas(r, absolute, "skeleton", cfg[3]), timing)
                   for r in rows]
            avg = sum(per) / len(per)
            ms["attribution"] += n_err * A["labels_per_error"] * A["replays_per_label"] * avg
            ms["attribution"] += n_plc * A["placebo_replays"] * avg
        dv = rows[: plan["dvs_records"]]
        ms["DVS"] += sum(len(r["masks"]) * fwd_ms(model, canvas(r, absolute, "skeleton"), timing) for r in dv)
    return {k: v / 3.6e6 for k, v in ms.items()}


def ar_hours(name, rows_by_set, plan):
    _, ktok, dec = AR[name]
    ms = 0.0
    for key in ("bfcl", "probe"):
        for r in rows_by_set[key]:
            prefill = 15 + r["P"] * ktok / 1000
            skel = r["n_slots"] * (ASSUMPTIONS["ar_value_tokens_per_slot"] + 1) + r["n_slots"] + 1
            ms += prefill + skel * dec
            if key == "bfcl":
                ms += prefill + r["S"] * dec  # free mode
    return ms / 3.6e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="minimal", choices=list(PLANS))
    ap.add_argument("--timing", action="append", default=[],
                    help="model=ms_per_1k_tokens:ms_overhead, from the smoke test")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--probe", default="data/paraprobe.jsonl")
    ap.add_argument("--gpus", type=int, default=2)
    args = ap.parse_args()
    plan = PLANS[args.plan]
    timing = {}
    for t in args.timing:
        m, v = t.split("=")
        a, b = v.split(":")
        timing[m] = (float(a), float(b))

    bf = [e for c in plan["bfcl"] for e in bfcl.load(c, args.bfcl_dir)]
    probe = load_jsonl(args.probe) * plan["probe_scale"]
    measured = {m: measure(m, {"bfcl": bf, "probe": probe}) for m in set(plan["dllms"]) | {AR[a][0] for a in plan["ar"]}}

    total, rows = 0.0, []
    n_invocations = 0
    for m in plan["dllms"]:
        h = dllm_hours(m, measured[m], plan, timing)
        for k, v in h.items():
            rows.append((m, k, v))
            total += v
        n_invocations += 4 + len(plan["attribution_k"]) * 2 + (3 if DLLM[m][3] else 0)
    for a in plan["ar"]:
        v = ar_hours(a, measured[AR[a][0]], plan)
        rows.append((a, "AR baseline (skeleton + free)", v))
        total += v
        n_invocations += 3
    load = n_invocations * ASSUMPTIONS["model_load_min"] / 60
    rows.append(("-", f"model loading ({n_invocations} runs)", load))
    rows.append(("-", "smoke tests / debugging", plan["debug_hours"]))
    total += load + plan["debug_hours"]

    print(f"plan={args.plan}  BFCL items={len(bf)}  ParaProbe items={len(probe)}")
    print("| model | work | A100 h |\n|---|---|---|")
    for m, k, v in rows:
        print(f"| {m} | {k} | {v:.1f} |")
    buf = ASSUMPTIONS["rerun_buffer"]
    print(f"\nsubtotal {total:.1f} h; with x{buf} rerun buffer: {total * buf:.1f} A100-hours")
    print(f"wall-clock on {args.gpus} GPUs (models in parallel): ~{total * buf / args.gpus:.1f} h")


if __name__ == "__main__":
    main()
