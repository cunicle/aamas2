"""End-to-end run of one example: prompt -> decode -> parse -> diagnose -> record."""

import dataclasses
import time

from ptcdiag.decoding.constraints import NoConstraint, SkeletonConstraint, lengths_from_list, lengths_to_list
from ptcdiag.decoding.sampler import Sampler
from ptcdiag.eval.taxonomy import diagnose
from ptcdiag.prompting import parse_tool_calls, render_prompt

MODES = ("free", "skeleton")


def make_constraint(adapter, example, mode, slot_lengths=None, surplus=0, end_bias=0.0, lengths=None,
                    closer_in_slot=False, tolerant=False, active_call=None):
    if (closer_in_slot or tolerant or active_call is not None) and mode != "skeleton":
        raise ValueError("closer_in_slot / tolerant / active_call are variants of the skeleton mode")
    if mode == "free":
        return NoConstraint()
    if mode == "skeleton":
        return SkeletonConstraint(adapter, example, slot_lengths, surplus, end_bias, lengths, closer_in_slot,
                                  tolerant, active_call)
    raise ValueError(mode)


def record_constraint(adapter, example, record):
    """The constraint a result record was decoded under (for replay, attribution, DVS)."""
    lengths = lengths_from_list(record["lengths"]) if record.get("lengths") else None
    return make_constraint(adapter, example, record["mode"], record.get("slot_lengths"),
                           record.get("surplus", 0), record.get("end_bias", 0.0), lengths,
                           record.get("closer_in_slot", False), record.get("tolerant", False),
                           record.get("active_call"))


def decode_region(adapter, canvas, trace, mode, constraint=None):
    """Text of the generation region, plus the canvas position of every kept token.

    free:     cut at the first EOS.
    skeleton: cut every value at its closing token (`constraint.cuts`), and drop
              padding/EOS tokens wherever they occur (slots are padded).
    """
    ids = canvas[trace.gen_start:trace.gen_end].tolist()
    positions = list(range(trace.gen_start, trace.gen_end))
    special = adapter.special_ids
    if mode == "skeleton":
        assert constraint is not None, "skeleton decoding needs the constraint (value cuts)"
        cuts = constraint.cuts(ids)
    else:
        cuts = {}
    kept_ids, kept_pos = [], []
    for j, (i, p) in enumerate(zip(ids, positions)):
        if j in cuts:  # a re-tokenized prefix keeps the position of the token it came from
            kept_ids += cuts[j]
            kept_pos += [p] * len(cuts[j])
            continue
        if i in special:
            if mode == "free" and i in adapter.eos_ids:
                break
            continue
        kept_ids.append(i)
        kept_pos.append(p)
    text = adapter.tokenizer.decode(kept_ids, skip_special_tokens=False)
    return text, kept_ids, kept_pos


def run_example(adapter, example, cfg, mode="free", slot_lengths=None, keep_trace=True,
                surplus=0, end_bias=0.0, lengths=None, length_mode="oracle", closer_in_slot=False,
                tolerant=False, active_call=None):
    """lengths / length_mode: per-slot lengths replacing the oracle ones, and the name of
    where they came from ("swap", an estimate file); recorded with the skeleton runs.
    closer_in_slot: the skeleton variant in which the model writes each value's closer
    (ptcdiag/decoding/constraints.py); recorded only when set. tolerant / active_call: the
    format-tolerant slots and the position agents of experiment D; recorded only when set."""
    prompt = render_prompt(adapter.tokenizer, example)
    prompt_ids = adapter.encode(prompt)
    constraint = make_constraint(adapter, example, mode, slot_lengths, surplus, end_bias, lengths,
                                 closer_in_slot, tolerant, active_call)
    if mode == "skeleton":
        # the canvas only needs to hold the skeleton; the recorded cfg keeps the real length
        cfg = dataclasses.replace(cfg, gen_length=len(constraint.gen_ids))
    t0 = time.time()
    canvas, trace = Sampler(adapter, cfg, constraint).generate(prompt_ids)
    elapsed = time.time() - t0
    text, kept_ids, kept_pos = decode_region(adapter, canvas, trace, mode, constraint)
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
    if slot_lengths:  # replay must rebuild the same skeleton
        rec["slot_lengths"] = slot_lengths
    if mode == "skeleton":
        rec["surplus"], rec["end_bias"], rec["length_mode"] = surplus, end_bias, length_mode
        if lengths:
            rec["lengths"] = lengths_to_list(lengths)
        if closer_in_slot:
            rec["closer_in_slot"] = True
        if tolerant:
            rec["tolerant"] = True
        if active_call is not None:
            rec["active_call"] = active_call
    if keep_trace:
        rec["trace"] = trace.to_dict()
    return rec
