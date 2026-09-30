"""End-to-end run of one example: prompt -> decode -> parse -> diagnose -> record."""

import dataclasses
import time

from ptcdiag.decoding.constraints import NoConstraint, SkeletonConstraint
from ptcdiag.decoding.sampler import Sampler
from ptcdiag.eval.taxonomy import diagnose
from ptcdiag.prompting import parse_tool_calls, render_prompt

MODES = ("free", "skeleton")


def make_constraint(adapter, example, mode, slot_lengths=None):
    if mode == "free":
        return NoConstraint()
    if mode == "skeleton":
        return SkeletonConstraint(adapter, example, slot_lengths)
    raise ValueError(mode)


def decode_region(adapter, canvas, trace, mode):
    """Text of the generation region, plus the canvas position of every kept token.

    free:     cut at the first EOS.
    skeleton: drop padding/EOS tokens wherever they occur (slots are padded).
    """
    ids = canvas[trace.gen_start:trace.gen_end].tolist()
    positions = list(range(trace.gen_start, trace.gen_end))
    special = adapter.special_ids
    kept_ids, kept_pos = [], []
    for i, p in zip(ids, positions):
        if i in special:
            if mode == "free" and i in adapter.eos_ids:
                break
            continue
        kept_ids.append(i)
        kept_pos.append(p)
    text = adapter.tokenizer.decode(kept_ids, skip_special_tokens=False)
    return text, kept_ids, kept_pos


def run_example(adapter, example, cfg, mode="free", slot_lengths=None, keep_trace=True):
    prompt = render_prompt(adapter.tokenizer, example)
    prompt_ids = adapter.encode(prompt)
    constraint = make_constraint(adapter, example, mode, slot_lengths)
    if mode == "skeleton":
        # the canvas only needs to hold the skeleton; the recorded cfg keeps the real length
        cfg = dataclasses.replace(cfg, gen_length=len(constraint.gen_ids))
    t0 = time.time()
    canvas, trace = Sampler(adapter, cfg, constraint).generate(prompt_ids)
    elapsed = time.time() - t0
    text, kept_ids, kept_pos = decode_region(adapter, canvas, trace, mode)
    parsed = parse_tool_calls(text)
    diag = diagnose(example, parsed)
    rec = {
        "id": example.id,
        "category": example.category,
        "meta": example.meta,
        "model": adapter.name,
        "mode": mode,
        "cfg": cfg.to_dict(),
        "cfg_tag": cfg.tag(),
        "prompt_len": len(prompt_ids),
        "text": text,
        "syntax_ok": parsed.syntax_ok,
        "calls": parsed.calls,
        "diagnosis": diag.to_dict(),
        "nfe": trace.nfe,
        "n_steps": len(trace.steps),
        "seconds": round(elapsed, 3),
        "gen_ids": canvas[trace.gen_start:trace.gen_end].tolist(),
    }
    if keep_trace:
        rec["trace"] = trace.to_dict()
    return rec
