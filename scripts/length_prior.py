"""Length prior of a masked diffusion model: what it wants right after a gold value.

Every skeleton slot gets its reference gold value teacher-forced plus `--surplus` extra
masks; all other slots hold their gold values too, so the context is the correct answer.
At the first surplus position of each slot this records the model's (unconstrained)
probability of padding, of the slot's closing tokens and of anything else, and which
kind of token the skeleton constraint would commit there. A model that read the masks
as "nothing more to say" would put padding or a closer there; one that reads the mask
count as the length of the content puts more content.

  python scripts/length_prior.py --model Dream-org/Dream-v0-Instruct-7B \
      --data bfcl:parallel,parallel_multiple --surplus 4 --out results/dream/length_prior.jsonl

With --closer-in-slot (experiment C3) the skeleton writes no closers: every slot holds its gold
value and then 1 + surplus masks, followed by the next fixed text, and the closer is the model's
to write at the first of them (with surplus 0 that position can only be the closer). The
original probe's surplus s is s masks before the skeleton's closer.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.adapters import load_adapter  # noqa: E402
from ptcdiag.decoding.constraints import (SkeletonConstraint, slot_class, slot_closers,  # noqa: E402
                                          token_texts, value_text)
from ptcdiag.prompting import render_prompt  # noqa: E402


_CLOSE = {}


def _closer_mask(texts, closers, V, device):
    key = (closers, V, str(device))
    if key not in _CLOSE:
        flags = [t.startswith(closers) for t in texts[:V]] + [False] * max(0, V - len(texts))
        _CLOSE[key] = torch.tensor(flags, dtype=torch.bool, device=device)
    return _CLOSE[key]


@torch.no_grad()
def measure(adapter, example, surplus, block_length, closer_in_slot=False):
    tok, texts = adapter.tokenizer, token_texts(adapter.tokenizer)
    prompt_ids = adapter.encode(render_prompt(tok, example))
    P = len(prompt_ids)
    c = SkeletonConstraint(adapter, example, surplus=surplus, closer_in_slot=closer_in_slot)
    G = adapter.canvas_length(P, len(c.gen_ids), block_length) - P
    gen = c.initial_gen(G, P)
    gold = {(ci, p): next(a for a in acc if a != "")
            for ci, (_, params) in enumerate(example.gold_calls) for p, acc in params.items()
            if any(a != "" for a in acc)}
    queries = []
    for s in c.slots:
        ids = tok(value_text(gold[s.call, s.param], s.type), add_special_tokens=False)["input_ids"]
        L = len(s.positions) - surplus - (1 if closer_in_slot else 0)  # the variant's closer position
        gen[s.positions[0]:s.positions[0] + L] = ids[:L]
        queries.append((s, L, P + s.positions[L]))
    x = torch.cat([prompt_ids, torch.tensor(gen, dtype=torch.long)]).to(adapter.device)[None]

    by_window = defaultdict(list)  # LLaDA2.0 forwards only up to the end of the query's block
    for q in queries:
        we = math.ceil((q[2] + 1) / block_length) * block_length if adapter.absolute_blocks and block_length \
            else None
        by_window[we].append(q)
    out = []
    for we, qs in by_window.items():
        pos = torch.tensor([q[2] for q in qs], device=x.device)
        logits = adapter.logits(x, we)[pos].float()
        probs = torch.softmax(logits, -1)
        cprobs = torch.softmax(c.filter(x, pos, logits), -1)
        for (s, L, p), pr, cpr in zip(qs, probs, cprobs):
            V = pr.shape[0]
            is_close = _closer_mask(texts, slot_closers(s.type), V, pr.device)
            p_pad, p_close = float(pr[adapter.pad_id]), float(pr[is_close].sum())
            top = int(cpr.argmax())
            kind = "pad" if top == adapter.pad_id else "closer" if bool(is_close[top]) else "content"
            out.append({"id": example.id, "call": s.call, "param": s.param, "type": s.type,
                        "class": slot_class(s.type), "value_len": L, "p_pad": p_pad, "p_close": p_close,
                        "p_content": max(0.0, 1 - p_pad - p_close),
                        "top_unconstrained": texts[int(pr.argmax())] if int(pr.argmax()) < len(texts) else "",
                        "constrained_top": texts[top] if top < len(texts) else "", "constrained_kind": kind,
                        "constrained_top_p": float(cpr[top])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--surplus", type=int, default=4)
    ap.add_argument("--closer-in-slot", action="store_true",
                    help="the skeleton variant without closers; surplus 0 is allowed")
    ap.add_argument("--block-length", default="none", help="'none' or an int (LLaDA2.0: 32)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()
    assert args.surplus >= 1 or args.closer_in_slot, "need at least one surplus position to look at"
    B = None if args.block_length.lower() == "none" else int(args.block_length)

    examples = [ex for spec in args.data for ex in load_examples(spec, args.bfcl_dir)][: args.limit]
    adapter = load_adapter(args.model, device=args.device)
    rows = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for i, ex in enumerate(examples):
            for r in measure(adapter, ex, args.surplus, B, args.closer_in_slot):
                r["model_id"], r["surplus"] = args.model, args.surplus
                if args.closer_in_slot:
                    r["closer_in_slot"] = True
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(examples)}", flush=True)

    print(f"{args.model}, surplus {args.surplus}{', closer in slot' if args.closer_in_slot else ''}: "
          "next token right after the gold value")
    print("| slot class | n | median P(pad) | mean P(closer) | mean P(content) | constrained pick: closer / pad / content |")
    print("|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in rows:
        groups[r["class"]].append(r)
        groups["all"].append(r)
    for cls, rs in sorted(groups.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        n = len(rs)
        kinds = [sum(r["constrained_kind"] == k for r in rs) / n for k in ("closer", "pad", "content")]
        print(f"| {cls} | {n} | {statistics.median(r['p_pad'] for r in rs):.1e} | "
              f"{statistics.mean(r['p_close'] for r in rs):.3f} | {statistics.mean(r['p_content'] for r in rs):.3f} | "
              + " / ".join(f"{k:.3f}" for k in kinds) + " |")


if __name__ == "__main__":
    main()
