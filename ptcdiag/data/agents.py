"""Experiment B: teams of LLM agents on the choose-N probes.

Each choose-N item (ptcdiag/data/choose.py) asks for n get_weather calls without saying
which call takes which city. Here a team of n agents answers it, each agent making one
of the n calls. Two factors are crossed (PROTOCOLS, values = (observe, label)):

  observe  the agent sees the calls of agents 1..i-1 (turn-taking) or none of the
           others' calls (simultaneous moves)
  label    the agent is told its index ("assistant i of n") or is anonymous

The protocol text is appended to the original request after a blank line; the system
prompt is the shared one of ptcdiag/prompting.py. Every agent's skeleton holds one call
(the item's first ground-truth call), so its single city slot has the same length as a
slot of the one-canvas runs (one token).

Experiment C adds `sim-rule` (RULE_PROTOCOLS): sim-label plus the convention that agent i
makes the i-th call in the order the request mentions them. The BFCL teams of experiment C
are built in ptcdiag/data/agents_bfcl.py.
"""

import json

from ptcdiag.types import Example

PROTOCOLS = {"sim-anon": (False, False), "sim-label": (False, True),
             "turn-anon": (True, False), "turn-label": (True, True)}

SIM_ANON = ("You are one of {n} assistants answering this request at the same time. Each assistant "
            "makes exactly one of the {n} calls, and the assistants cannot see each other's calls. "
            "Make your one call.")
SIM_LABEL = ("You are assistant {i} of {n} answering this request at the same time. Each assistant "
             "makes exactly one of the {n} calls, and the assistants cannot see each other's calls. "
             "Make your one call.")
TURN_ANON = ("You are one of {n} assistants answering this request one after another. Each assistant "
             "makes exactly one of the {n} calls. Calls made so far: {calls}. Make your one call.")
TURN_LABEL = ("You are assistant {i} of {n} answering this request one after another. Each assistant "
              "makes exactly one of the {n} calls. Calls made so far: {calls}. Make your one call.")
INSTRUCTIONS = {"sim-anon": SIM_ANON, "sim-label": SIM_LABEL,
                "turn-anon": TURN_ANON, "turn-label": TURN_LABEL}

# Experiment C's positive control: simultaneous indexed agents that are also told the convention
# (agent i makes the i-th call in mention order). Kept out of PROTOCOLS, so experiment B's
# protocols, texts and their order stay exactly as they were.
SIM_RULE = ("You are assistant {i} of {n} answering this request at the same time. Each assistant "
            "makes exactly one of the {n} calls, and the assistants cannot see each other's calls. "
            "By convention, assistant {i} makes the {ordinal} of the calls, in the order in which the "
            "request mentions them. Make your one call.")
RULE_PROTOCOLS = {"sim-rule": (False, True)}
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def ordinal(i):
    """'first' .. 'eighth' for i = 1..8 (the largest BFCL team has eight agents)."""
    if not 1 <= i <= len(ORDINALS):
        raise ValueError(f"no ordinal for agent {i}")
    return ORDINALS[i - 1]


def protocol_flags(protocol):
    """(observe, label) of an experiment B protocol or of sim-rule."""
    return PROTOCOLS[protocol] if protocol in PROTOCOLS else RULE_PROTOCOLS[protocol]


def format_calls(previous):
    """{calls} of the turn-taking texts: "none", or a JSON array with one entry per earlier
    agent: its parsed call (a dict) as JSON, or its raw output text (a str, the agent's
    output did not parse) inserted as it is."""
    if not previous:
        return "none"
    return "[" + ", ".join(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
                           for p in previous) + "]"


def instruction(protocol, i, n, previous=()):
    observe, _ = protocol_flags(protocol)
    if not observe and previous:
        raise ValueError(f"{protocol}: simultaneous agents see no earlier calls")
    if protocol in RULE_PROTOCOLS:
        return SIM_RULE.format(i=i, n=n, ordinal=ordinal(i))
    return INSTRUCTIONS[protocol].format(i=i, n=n, calls=format_calls(previous))


def agent_example(ex, i, n, protocol, previous=()):
    """The one-call example agent i (1-based) of an n-agent team answers under `protocol`.

    previous: the calls of agents 1..i-1 (dicts {"name", "arguments"}, or the raw output
    text of an agent whose output did not parse); empty for simultaneous protocols."""
    if not 1 <= i <= n:
        raise ValueError(f"agent {i} of {n}")
    (msg,) = ex.messages
    text = msg["content"] + "\n\n" + instruction(protocol, i, n, previous)
    return Example(
        id=f"{ex.id}_a{i}",
        category=ex.category,
        messages=[{"role": "user", "content": text}],
        functions=ex.functions,
        ground_truth=ex.ground_truth[:1],
        meta={**ex.meta, "agent": i, "protocol": protocol},
    )
