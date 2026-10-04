"""Experiments C1 / C2: teams of agents answer the BFCL parallel requests, one call per agent.

  python scripts/run_agents_bfcl.py --model Qwen/Qwen2.5-7B-Instruct --backend ar --subset sym \
      --protocols sim-anon,sim-label,sim-rule,turn-anon,turn-label --out results/qwen/agents_c1.jsonl
  python scripts/run_agents_bfcl.py --model Dream-org/Dream-v0-Instruct-7B --backend dllm --subset swap \
      --length-mode swap --out results/dream/agents_c2_swap.jsonl
  python scripts/run_agents_bfcl.py --subset sym --protocols sim-anon,sim-rule,turn-label \
      --dry-run --limit 2      # only print the agents' user texts and slot lengths (no model)

Per request of the subset (sym: the team-symmetric requests, C1; swap: the requests where the
length swap changes a slot, C2) and protocol, agents 1..n make the request's n reference
calls, agent i the i-th (ptcdiag/data/agents_bfcl.py), each decoded with its call's
single-call skeleton: Qwen with `run_example_ar`, Dream with `run_example` (k=1, confidence
order, no blocks), greedy. Slot lengths are the oracle ones, or (Dream only) the swapped
lengths of the whole request. Turn-taking agents see the earlier agents' parsed calls (the raw
text when an output did not parse into exactly one call), as in scripts/run_agents.py.

The team's output is the JSON array of the n calls, diagnosed against the whole request like
a one-canvas output; a team in which some agent did not make exactly one call is unparsed
(not correct, labels ["syntax_error"]). One JSONL record per team, with every agent's user
text, slot lengths and output. Re-running with the same --out skips the (id, protocol,
length mode) already there.

Experiment D adds `pos-anon`, position agents (ptcdiag/data/agents.py): agent i gets the request
with no note and the skeleton of the whole turn (oracle lengths), and fills call i only; Dream
leaves the other calls' slots masked, Qwen writes placeholders into them. The agent's call is
the i-th of its decoded array.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.data.agents import (POSITION_PROTOCOLS, PROTOCOLS, RULE_PROTOCOLS,  # noqa: E402
                                 position_agent_example, protocol_flags)
from ptcdiag.data.agents_bfcl import (agent_lengths, bfcl_agent_example, swap_changes,  # noqa: E402
                                      team_symmetric)
from ptcdiag.decoding.constraints import lengths_to_list, oracle_lengths  # noqa: E402
from ptcdiag.eval.taxonomy import diagnose  # noqa: E402
from ptcdiag.prompting import ParsedOutput, parse_tool_calls  # noqa: E402

DREAM = "Dream-org/Dream-v0-Instruct-7B"
SUBSETS = {"sym": team_symmetric, "swap": swap_changes}


def run_team(ex, protocol, decode, lengths_of):
    """Agents 1..n of the team for request `ex` under `protocol`, one after another.

    decode(agent_example, lengths) -> (text, syntax_ok, calls), lengths None for the oracle ones.
    lengths_of(i, agent_example) -> (the lengths passed to decode, the slot lengths used).
    An agent passes on its parsed call, or its raw text when the output does not parse into
    exactly one call."""
    n = len(ex.ground_truth)
    observe, _ = protocol_flags(protocol)
    pos = protocol in POSITION_PROTOCOLS
    previous, agents = [], []
    for i in range(1, n + 1):
        if pos:  # the whole turn's skeleton; the agent's call is the i-th of its output
            aex = position_agent_example(ex, i, n)
            lengths, used = lengths_of(i, aex)
            text, syntax_ok, calls = decode(aex, lengths, active_call=i - 1)
            one = syntax_ok and len(calls) == n
            call = calls[i - 1] if one else None
        else:
            aex = bfcl_agent_example(ex, i, n, protocol, previous if observe else ())
            lengths, used = lengths_of(i, aex)
            text, syntax_ok, calls = decode(aex, lengths)
            one = syntax_ok and len(calls) == 1
            call = calls[0] if one else None
        agents.append({"i": i, "user_text": aex.messages[0]["content"], "text": text,
                       "syntax_ok": syntax_ok, "call": call, "lengths": lengths_to_list(used)})
        previous.append(call if one else text)
    return agents


def assemble(ex, agents):
    """The team's output: the agents' calls as one JSON array, parsed and diagnosed against the
    whole request as a one-canvas output is; unparsed (correct False, labels ["syntax_error"])
    when some agent did not make exactly one call."""
    missing = [a["i"] for a in agents if a["call"] is None]
    if missing:
        text = None
        parsed = ParsedOutput("", False, error=f"agents {missing} did not make exactly one call")
    else:
        text = json.dumps([a["call"] for a in agents])
        parsed = parse_tool_calls(text)
    return {"syntax_ok": parsed.syntax_ok, "text": text, "calls": parsed.calls,
            "diagnosis": diagnose(ex, parsed).to_dict()}


def lengths_fn(tokenizer, ex, mode):
    """lengths_of for run_team: (None, the agent's oracle lengths) or (the swapped lengths, the same)."""
    def lengths_of(i, aex):
        if mode == "oracle":
            return None, oracle_lengths(tokenizer, aex)
        sw = agent_lengths(tokenizer, ex, i, mode)
        return sw, sw
    return lengths_of


def placeholder_decode(aex, lengths, active_call=None):
    """--dry-run: agent i 'calls' its function with placeholder values, so turn-taking texts show
    the format (a position agent: every call of the turn)."""
    calls = [{"name": fname, "arguments": {p: f"<{p} of agent {aex.meta['agent']}>" for p, acc in params.items()
                                           if any(a != "" for a in acc)}}
             for fname, params in aex.gold_calls]
    return json.dumps(calls), True, calls


def make_decoder(model, backend):
    """(decode, decoding tag): greedy single-call skeleton decoding."""
    if backend == "ar":
        from ptcdiag.decoding.ar import run_example_ar

        def decode(aex, lengths, active_call=None):
            if lengths is not None:
                raise ValueError("AR skeleton decoding takes no per-slot lengths")
            r = run_example_ar(model, aex, mode="skeleton", active_call=active_call)
            return r["text"], r["syntax_ok"], r["calls"]

        return decode, "ar_greedy"

    from ptcdiag.decoding.sampler import DecodeConfig
    from ptcdiag.pipeline import run_example

    cfg = DecodeConfig(block_length=None, k=1, order="confidence")

    def decode(aex, lengths, active_call=None):
        r = run_example(model, aex, cfg, mode="skeleton", keep_trace=False, lengths=lengths,
                        length_mode="swap" if lengths else "oracle", active_call=active_call)
        return r["text"], r["syntax_ok"], r["calls"]

    return decode, cfg.tag()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["dllm", "ar"])
    ap.add_argument("--data", default="bfcl:parallel,parallel_multiple")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    ap.add_argument("--subset", choices=list(SUBSETS), required=True,
                    help="sym: team-symmetric requests (C1); swap: requests the length swap changes (C2)")
    ap.add_argument("--length-mode", choices=["oracle", "swap"], default="oracle")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS),
                    help="comma list of experiment B's protocols, sim-rule and/or pos-anon")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the user texts and slot lengths; no model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.backend == "ar" and args.length_mode == "swap":
        raise SystemExit("--backend ar --length-mode swap: the AR skeleton decoding has no per-slot lengths "
                         "and stops at the closer, so it never reads a slot length; run it with oracle lengths")
    protocols = args.protocols.split(",")
    known = [*PROTOCOLS, *RULE_PROTOCOLS, *POSITION_PROTOCOLS]
    for p in protocols:
        if p not in known:
            raise SystemExit(f"unknown protocol {p!r} (one of {', '.join(known)})")
    if args.length_mode != "oracle" and any(p in POSITION_PROTOCOLS for p in protocols):
        raise SystemExit("position agents (pos-anon) run with oracle lengths only")
    if not args.dry_run and not (args.model and args.backend and args.out):
        raise SystemExit("--model, --backend and --out are required (except with --dry-run)")

    model = None
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model or DREAM, trust_remote_code=True)
    else:
        from run_agents import load_model

        model = load_model(args.backend, args.model, args.device)
        tokenizer = model.tokenizer
    examples = [ex for ex in load_examples(args.data, args.bfcl_dir) if SUBSETS[args.subset](tokenizer, ex)]
    n_all = len(examples)
    if args.limit:
        examples = examples[: args.limit]

    if args.dry_run:
        for ex in examples:
            for p in protocols:
                print(f"===== {ex.id} ({ex.category})  n={len(ex.ground_truth)}  {p}  lengths={args.length_mode}")
                for a in run_team(ex, p, placeholder_decode, lengths_fn(tokenizer, ex, args.length_mode)):
                    print(f"--- agent {a['i']}  slot lengths {a['lengths']}\n{a['user_text']}")
        return

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in map(json.loads, f):
                if "error" not in r:
                    done.add((r["id"], r["protocol"], r["length_mode"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = len(examples) * len(protocols)
    print(f"subset {args.subset}: {n_all} requests ({len(examples)} run) x {len(protocols)} protocols = {total} "
          f"teams, lengths {args.length_mode} ({len(done)} already done)", flush=True)

    decode, tag = make_decoder(model, args.backend)
    t0, n_run, n_agents = time.time(), 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for p in protocols:
            for ex in examples:
                if (ex.id, p, args.length_mode) in done:
                    continue
                rec = {"id": ex.id, "category": ex.category, "subset": args.subset,
                       "length_mode": args.length_mode, "n": len(ex.ground_truth), "protocol": p,
                       "model_id": args.model, "backend": args.backend, "decoding": tag}
                t1 = time.time()
                try:
                    rec["agents"] = run_team(ex, p, decode, lengths_fn(tokenizer, ex, args.length_mode))
                    rec["team"] = assemble(ex, rec["agents"])
                    n_agents += len(rec["agents"])
                except Exception as e:  # keep the sweep going; failures are recorded
                    rec["error"] = repr(e)
                rec["seconds"] = round(time.time() - t1, 3)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_run += 1
                if n_run % 20 == 0:
                    dt = time.time() - t0
                    print(f"{n_run}/{total - len(done)} teams  {dt / max(n_agents, 1):.2f}s/agent  {p}", flush=True)
    print(f"done: {n_run} teams, {n_agents} agents in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
