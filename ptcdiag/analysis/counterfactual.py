"""Counterfactual sequentialisation (proposal §6.5 B).

For an erroneous output: find the canvas positions that carry the error, the steps
that committed them together with other positions, and re-decode from the state
right before such a step with that single step made sequential. If the error label
disappears, the error is attributed to simultaneous commitment at that step.

Placebo: the same search (same step budget) on a random value of correct outputs;
the rate at which correct outputs break is the noise floor. Attributable fraction =
P(fixed | error) - P(broken | correct).

Requires deterministic decoding (temperature 0, order != random), so that resuming
from the reconstructed state is identical to re-running the prefix. `reproduces`
checks this on real hardware (bf16 kernels can be nondeterministic).
"""

import dataclasses
import random

import torch

from ptcdiag.decoding.sampler import DecodeConfig, Sampler, Trace, state_before_step
from ptcdiag.eval.taxonomy import diagnose
from ptcdiag.pipeline import decode_region, record_constraint
from ptcdiag.prompting import parse_tool_calls, render_prompt, token_char_offsets, tokens_in_span

LOCALIZABLE = {"cross_binding", "chimera_value", "wrong_value", "type_error",
               "duplicate_call", "inconsistent_shared_arg"}


def _canvas(prompt_ids, record):
    return torch.cat([prompt_ids, torch.tensor(record["gen_ids"], dtype=torch.long)])


def error_positions(adapter, record, detail, trace, canvas, constraint=None):
    """Canvas positions responsible for one taxonomy detail, or [] if not localisable.

    constraint: the record's constraint (`record_constraint`); needed in skeleton mode.
    """
    text, kept_ids, kept_pos = decode_region(adapter, canvas, trace, record["mode"], constraint)
    parsed = parse_tool_calls(text)
    if not parsed.syntax_ok:
        return []
    spans = []
    lab = detail["label"]
    if lab in ("cross_binding", "chimera_value", "wrong_value", "type_error"):
        spans.append(parsed.value_span(detail["pred"], detail["param"]))
    elif lab == "duplicate_call":
        # the argument values of the surplus copies (their names/brackets are not the issue)
        for i in detail.get("surplus", []):
            spans += [parsed.value_span(i, p) for p in parsed.calls[i]["arguments"]]
    elif lab == "inconsistent_shared_arg":
        for pi, _ in record["diagnosis"]["matching"]:
            spans.append(parsed.value_span(pi, detail["param"]))
    spans = [s for s in spans if s]
    if not spans:
        return []
    ends = token_char_offsets(adapter.tokenizer, kept_ids)
    idx = sorted({i for s in spans for i in tokens_in_span(ends, s)})
    return [kept_pos[i] for i in idx]


def parallel_steps(trace, positions):
    """Steps that committed any of `positions` together with other positions.

    Ordered by the lowest probability with which one of `positions` was committed at
    that step (ascending): low-confidence commits are where an ambiguity was resolved,
    so they are tried first under a fixed search budget. Ties: earlier step first.
    """
    want = set(positions)
    ranked = []
    for s in trace.steps:
        if len(s.positions) < 2 or s.sequentialized:
            continue
        ps = [pr for p, pr in zip(s.positions, s.probs) if p in want]
        if ps:
            ranked.append((min(ps), s.step))
    return [t for _, t in sorted(ranked)]


def replay(adapter, example, record, step, sequentialize=True):
    """Re-decode from the state before `step`; returns (text, diagnosis dict)."""
    cfg = DecodeConfig(**{k: v for k, v in record["cfg"].items()
                          if k in {f.name for f in dataclasses.fields(DecodeConfig)}})
    assert cfg.temperature == 0 and cfg.order != "random", "counterfactuals need deterministic decoding"
    cfg = dataclasses.replace(cfg, sequentialize_steps=(step,) if sequentialize else ())
    prompt_ids = adapter.encode(render_prompt(adapter.tokenizer, example))
    trace = Trace.from_dict(record["trace"])
    canvas = _canvas(prompt_ids, record)
    init = state_before_step(trace, canvas, step, adapter.mask_id)
    constraint = record_constraint(adapter, example, record)
    new_canvas, new_trace = Sampler(adapter, cfg, constraint).generate(
        prompt_ids, init_gen=init, start_step=step)
    text, _, _ = decode_region(adapter, new_canvas.cpu(), new_trace, record["mode"], constraint)
    diag = diagnose(example, parse_tool_calls(text)).to_dict()
    return text, diag


def reproduces(adapter, example, record, step=0):
    """Resuming at `step` without intervention should give the original text."""
    text, _ = replay(adapter, example, record, step, sequentialize=False)
    return text == record["text"]


def attribute(adapter, example, record, labels=LOCALIZABLE, max_steps=6):
    """Try sequentialising each parallel step behind each localisable error.

    Returns one dict per error detail: steps tried, the first step whose
    sequentialisation removes the label (or None), and the resulting labels.
    """
    prompt_ids = adapter.encode(render_prompt(adapter.tokenizer, example))
    trace = Trace.from_dict(record["trace"])
    canvas = _canvas(prompt_ids, record)
    constraint = record_constraint(adapter, example, record)
    results = []
    for det in record["diagnosis"]["details"]:
        if det["label"] not in labels:
            continue
        pos = error_positions(adapter, record, det, trace, canvas, constraint)
        steps = parallel_steps(trace, pos)[:max_steps]
        res = {"label": det["label"], "detail": det, "positions": pos, "steps_tried": steps,
               "fixed_by": None, "became_correct": False, "after": []}
        for t in steps:
            text, diag = replay(adapter, example, record, t)
            res["after"].append({"step": t, "labels": diag["labels"], "correct": diag["correct"]})
            if det["label"] not in diag["labels"]:
                res["fixed_by"] = t
                res["became_correct"] = diag["correct"]
                break
        results.append(res)
    return results


def placebo(adapter, example, record, rng=None, max_steps=6):
    """Same search as `attribute`, applied to a random value of a correct output.

    Tries up to `max_steps` parallel steps behind that value (the same budget as for
    errors) and reports whether any of them breaks the output.
    """
    rng = rng or random.Random(0)
    prompt_ids = adapter.encode(render_prompt(adapter.tokenizer, example))
    trace = Trace.from_dict(record["trace"])
    canvas = _canvas(prompt_ids, record)
    constraint = record_constraint(adapter, example, record)
    calls = record["calls"]
    choices = [(i, p) for i, c in enumerate(calls) for p in c["arguments"]]
    rng.shuffle(choices)
    for i, p in choices:
        det = {"label": "wrong_value", "pred": i, "param": p}
        steps = parallel_steps(trace, error_positions(adapter, record, det, trace, canvas, constraint))[:max_steps]
        if not steps:
            continue
        for t in steps:
            _, diag = replay(adapter, example, record, t)
            if not diag["correct"]:
                return {"pred": i, "param": p, "steps_tried": steps, "broken": True,
                        "broken_by": t, "labels": diag["labels"]}
        return {"pred": i, "param": p, "steps_tried": steps, "broken": False,
                "broken_by": None, "labels": []}
    return None
