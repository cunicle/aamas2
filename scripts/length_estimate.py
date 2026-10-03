"""Slot lengths from the model itself: one forward pass, no gold lengths.

Every skeleton slot gets a fixed number of masks for its type (SLOT_LENGTHS, above p99 of
BFCL values), and one forward pass over that canvas gives, for every slot position, the
probability under the skeleton constraint that a value of length j ends there (h_j):
for strings / numbers / booleans the slot's closing tokens or padding at position j; for
arrays and dicts, whose values contain ',' themselves, a closing bracket (']', '],', '}'
...) at position j-1 (an inner bracket of a nested value also counts: a known
underestimate for nested values). Read as a hazard, P(length = j) = h_j * prod_{i<j}(1 - h_i)
for j = 1 .. cap-1 (values are never empty), and the rest of the mass is a value that
fills the slot; the estimate is the mode (`length_from_hazard`). This is the first-step
length estimate of CAL (Diffusion LMs can approximate optimal infilling lengths
implicitly, arXiv 2602.00476) done per slot. The output feeds `run_dllm.py --lengths <file>`.

Records per item the estimated and oracle lengths and every slot's hazard, and prints
how often the estimate equals the oracle length, by slot class.

  python scripts/length_estimate.py --model Dream-org/Dream-v0-Instruct-7B \
      --data bfcl:parallel,parallel_multiple --out results/dream/length_estimate.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.adapters import load_adapter  # noqa: E402
from ptcdiag.decoding.constraints import (SLOT_LENGTHS, SkeletonConstraint, length_from_hazard,  # noqa: E402
                                          lengths_to_list, oracle_lengths, slot_class, token_texts)
from ptcdiag.prompting import render_prompt  # noqa: E402


def caps(example):
    """{bfcl_type: masks} for every parameter type in the example (unknown types get the
    cap of their slot class), so no slot falls back to its oracle length."""
    by_class = {"string": SLOT_LENGTHS["string"], "integer": SLOT_LENGTHS["integer"],
                "float": SLOT_LENGTHS["float"], "boolean": SLOT_LENGTHS["boolean"],
                "generic": SLOT_LENGTHS["array"]}
    out = dict(SLOT_LENGTHS)
    for f in example.functions:
        for p in f["parameters"]["properties"].values():
            t = p.get("type", "string")
            out.setdefault(t, by_class[slot_class(t)])
    out.setdefault("string", SLOT_LENGTHS["string"])  # _param_type's default
    return out


_BRACKETS = {}


def bracket_mask(tokenizer, V, device):
    """Tokens that start with a closing bracket (the last token of an array / dict value)."""
    key = (tokenizer.name_or_path, V, str(device))
    if key not in _BRACKETS:
        flags = [t.startswith(("]", "}")) for t in token_texts(tokenizer)[:V]]
        flags += [False] * (V - len(flags))
        _BRACKETS[key] = torch.tensor(flags, dtype=torch.bool, device=device)
    return _BRACKETS[key]


@torch.no_grad()
def estimate(adapter, example, block_length):
    tok = adapter.tokenizer
    prompt_ids = adapter.encode(render_prompt(tok, example))
    P = len(prompt_ids)
    c = SkeletonConstraint(adapter, example, slot_lengths=caps(example))
    G = adapter.canvas_length(P, len(c.gen_ids), block_length) - P
    gen = c.initial_gen(G, P)
    x = torch.cat([prompt_ids.cpu().long(), torch.tensor(gen, dtype=torch.long)]).to(adapter.device)[None]
    pos = torch.tensor([P + g for s in c.slots for g in s.positions], device=x.device)
    logits = adapter.logits(x)[pos].float()
    probs = torch.softmax(c.filter(x, pos, logits), -1)
    V = probs.shape[-1]
    oracle = oracle_lengths(tok, example)
    est, hazards, i = {}, [], 0
    for s in c.slots:
        n, cls = len(s.positions), slot_class(s.type)
        if cls == "generic":
            h = [0.0] + probs[i:i + n - 1][:, bracket_mask(tok, V, probs.device)].sum(-1).clamp(0, 1).tolist()
        else:
            h = probs[i:i + n][:, c._end_mask(cls, V, probs.device)].sum(-1).clamp(0, 1).tolist()
        i += n
        est[s.call, s.param] = length_from_hazard(h)
        hazards.append([s.call, s.param, s.type, [round(v, 4) for v in h]])
    return {"id": example.id, "lengths": lengths_to_list(est),
            "oracle": lengths_to_list({k: oracle[k] for k in est}), "hazard": hazards}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--block-length", default="none", help="'none' or an int (LLaDA2.0: 32)")
    ap.add_argument("--per-data", type=int, default=None,
                    help="N evenly spaced examples from each --data spec (as run_dllm.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()
    B = None if args.block_length.lower() == "none" else int(args.block_length)

    examples = []
    for spec in args.data:
        exs = load_examples(spec, args.bfcl_dir)
        if args.per_data and len(exs) > args.per_data:
            exs = [exs[i * len(exs) // args.per_data] for i in range(args.per_data)]
        examples += exs
    adapter = load_adapter(args.model, device=args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])  # class -> n, exact, within 1, short, long
    items_exact = 0
    with open(args.out, "w") as f:
        for i, ex in enumerate(examples):
            r = estimate(adapter, ex, B)
            r["model_id"] = args.model
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            orc = {(ci, p): n for ci, p, n in r["oracle"]}
            types = {(ci, p): t for ci, p, t, _ in r["hazard"]}
            all_ok = True
            for ci, p, n in r["lengths"]:
                d = n - orc[ci, p]
                for key in (slot_class(types[ci, p]), "all"):
                    st = stats[key]
                    st[0] += 1
                    st[1] += d == 0
                    st[2] += abs(d) <= 1
                    st[3] += d < 0
                    st[4] += d > 0
                all_ok &= d == 0
            items_exact += all_ok
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(examples)}", flush=True)

    print(f"{args.model}: one-forward slot length estimate vs the reference value's length")
    print("| slot class | n | exact | within 1 | too short | too long |")
    print("|---|---|---|---|---|---|")
    for k in sorted(stats, key=lambda k: (k == "all", k)):
        n, ex_, w1, sh, lo = stats[k]
        print(f"| {k} | {n} | {ex_ / n:.3f} | {w1 / n:.3f} | {sh / n:.3f} | {lo / n:.3f} |")
    print(f"items with every slot exact: {items_exact}/{len(examples)} = {items_exact / len(examples):.3f}")


if __name__ == "__main__":
    main()
