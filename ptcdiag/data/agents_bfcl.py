"""Experiment C: teams of agents on the BFCL parallel requests, one call per agent.

A team of n agents answers a request with n reference calls; agent i (1-based) makes the
i-th one, so its skeleton is that call's single-call skeleton: the function name, the
parameter names and one slot per value (`bfcl_agent_example`). The protocol texts are the
ones of ptcdiag/data/agents.py (experiment B's four, and sim-rule).

  C1 (order cue)   the team-symmetric requests (`team_symmetric`): every call of a function
                   has the same parameters with the same slot lengths, so the n agents of a
                   function get identical skeletons and only the request's order and the
                   protocol text can tell agent i which mentioned entity is its own.
  C2 (length cue)  the requests where the length swap changes a slot (`swap_changes`, the
                   165 items of the one-canvas swap run); agent i's slots have its own
                   reference lengths (oracle) or the rotated ones of the whole request
                   (`agent_lengths(..., "swap")`).
"""

import dataclasses
from collections import Counter, defaultdict

from ptcdiag.data.agents import agent_example
from ptcdiag.decoding.constraints import oracle_lengths, swapped_lengths


def call_signature(lengths, ci, params):
    """Single-call skeleton signature of call ci: its (param, slot length) pairs, sorted. The
    skeleton's parameters are the ones in `lengths` (oracle_lengths leaves out parameters whose
    only acceptable value is "")."""
    return tuple(sorted((p, lengths[ci, p]) for p in params if (ci, p) in lengths))


def team_symmetric(tokenizer, ex):
    """True if every call of each function has the same single-call skeleton signature and some
    function is called at least twice: the agents of a function then get identical skeletons,
    and no slot length tells them apart."""
    L = oracle_lengths(tokenizer, ex)
    sigs, counts = defaultdict(set), Counter()
    for ci, (fname, params) in enumerate(ex.gold_calls):
        sigs[fname].add(call_signature(L, ci, params))
        counts[fname] += 1
    return all(len(s) == 1 for s in sigs.values()) and max(counts.values(), default=0) >= 2


def swap_changes(tokenizer, ex):
    """True if the length swap (`swapped_lengths`) changes some slot of the request."""
    return swapped_lengths(tokenizer, ex) != oracle_lengths(tokenizer, ex)


def bfcl_agent_example(ex, i, n, protocol, previous=()):
    """The one-call example agent i (1-based) of the n-agent team for BFCL request `ex` answers:
    `agent_example` (the request + a blank line + the protocol text; id f"{ex.id}_a{i}"), with
    the request's i-th reference call as its ground truth, so its skeleton is that call's.

    previous: the calls of agents 1..i-1 (dicts {"name", "arguments"}, or the raw output text
    of an agent whose output did not parse); empty for simultaneous protocols."""
    if n != len(ex.ground_truth):
        raise ValueError(f"{ex.id}: a team of {n} for {len(ex.ground_truth)} reference calls")
    a = agent_example(ex, i, n, protocol, previous)
    return dataclasses.replace(a, ground_truth=[ex.ground_truth[i - 1]], meta={**a.meta, "call": i - 1})


def agent_lengths(tokenizer, ex, i, mode):
    """Slot lengths of agent i's skeleton: None for "oracle" (its own reference lengths), or for
    "swap" the lengths `swapped_lengths` gives call i-1 in the whole request, keyed by the
    agent's own call index 0: {(0, param): n}."""
    if mode == "oracle":
        return None
    if mode == "swap":
        return {(0, p): n for (ci, p), n in swapped_lengths(tokenizer, ex).items() if ci == i - 1}
    raise ValueError(f"unknown length mode {mode!r}")
