"""Experiment B: teams of n agents answer the choose-N items, one call per agent.

  python scripts/run_agents.py --model Qwen/Qwen2.5-7B-Instruct --backend ar \
      --data probe:data/choose.jsonl --protocols sim-anon,sim-label,turn-anon,turn-label \
      --out results/qwen/agents.jsonl
  python scripts/run_agents.py --data probe:data/choose.jsonl --protocols sim-anon,turn-label \
      --dry-run --limit 3      # only print the agents' user texts (no model)
  python scripts/run_agents.py --model Qwen/Qwen2.5-7B-Instruct --backend ar --variant list \
      --protocols sim-rule --out results/qwen/agents_rule.jsonl     # experiment C's control

Per item x protocol x seed, agents 1..n are built (ptcdiag/data/agents.py) and decoded
one after another with the oracle skeleton of one call: Qwen with `run_example_ar`, Dream
with `run_example` (k=1, confidence order, no blocks). Turn-taking agents see the calls of
the agents before them; simultaneous agents see none. One JSONL record per team, with
every agent's user text, output and city. With --temperature > 0, agent i of the team
run with seed s samples with seed 1000 s + i. Re-running with the same --out skips the teams
that are already there.

Experiment D adds `pos-anon`, position agents: agent i gets the request with no note and the
skeleton of all n calls, and fills call i only (Dream leaves the other city slots masked, Qwen
writes placeholders into them); its city is that of the i-th call of its decoded array.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.data.agents import (POSITION_PROTOCOLS, PROTOCOLS, RULE_PROTOCOLS, agent_example,  # noqa: E402
                                 position_agent_example, protocol_flags)


def as_list(s, typ):
    return [typ(v) for v in s.split(",")]


def agent_seed(seed, i):
    """Agent i's own sampling seed in the team run with `seed`: agents that see the same
    input must not share their random draws."""
    return 1000 * seed + i


def run_team(ex, protocol, decode):
    """Agents 1..n of the team for `ex` under `protocol`.

    decode(agent_example) -> (text, syntax_ok, calls). An agent passes on its parsed call,
    or its raw text when the output does not parse into exactly one call."""
    n = ex.meta["n"]
    observe, _ = protocol_flags(protocol)
    pos = protocol in POSITION_PROTOCOLS
    previous, agents = [], []
    for i in range(1, n + 1):
        if pos:  # the skeleton of all n calls; the agent's call is the i-th of its output
            aex = position_agent_example(ex, i, n)
            text, syntax_ok, calls = decode(aex, active_call=i - 1)
            one = syntax_ok and len(calls) == n
            call = calls[i - 1] if one else None
        else:
            aex = agent_example(ex, i, n, protocol, previous if observe else ())
            text, syntax_ok, calls = decode(aex)
            one = syntax_ok and len(calls) == 1
            call = calls[0] if one else None
        city = call["arguments"].get("city") if one else None
        agents.append({"i": i, "user_text": aex.messages[0]["content"], "text": text,
                       "syntax_ok": syntax_ok, "city": city})
        previous.append(call if one else text)
    return agents


def placeholder_decode(aex, active_call=None):
    """--dry-run: agent i 'calls' a placeholder city, so turn-taking texts show the format (a
    position agent: every call of the turn)."""
    calls = [{"name": "get_weather", "arguments": {"city": f"<city of agent {aex.meta['agent']}>"}}
             for _ in aex.gold_calls]
    return json.dumps(calls), True, calls


def load_model(backend, model_id, device):
    if backend == "ar":
        from ptcdiag.decoding.ar import ARModel

        return ARModel(model_id, device=device)
    from ptcdiag.decoding.adapters import load_adapter

    return load_adapter(model_id, device=device)


def make_decoder(model, backend, temperature, seed):
    """(decode, decoding tag) for one temperature / team seed; agent i samples with
    agent_seed(seed, i) (greedy decoding ignores it)."""
    if backend == "ar":
        from ptcdiag.decoding.ar import run_example_ar

        def decode(aex, active_call=None):
            r = run_example_ar(model, aex, mode="skeleton", temperature=temperature,
                               seed=agent_seed(seed, aex.meta["agent"]), active_call=active_call)
            return r["text"], r["syntax_ok"], r["calls"]

        return decode, f"ar_T{temperature}" if temperature else "ar_greedy"

    import dataclasses

    from ptcdiag.decoding.sampler import DecodeConfig
    from ptcdiag.pipeline import run_example

    cfg = DecodeConfig(block_length=None, k=1, order="confidence", temperature=temperature, seed=seed)

    def decode(aex, active_call=None):
        acfg = dataclasses.replace(cfg, seed=agent_seed(seed, aex.meta["agent"]))
        r = run_example(model, aex, acfg, mode="skeleton", keep_trace=False, active_call=active_call)
        return r["text"], r["syntax_ok"], r["calls"]

    return decode, cfg.tag()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["dllm", "ar"])
    ap.add_argument("--data", default="probe:data/choose.jsonl")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS),
                    help="comma list of experiment B's protocols, sim-rule (experiment C) or pos-anon (D)")
    ap.add_argument("--variant", choices=["list", "open"], default=None,
                    help="only the items of this choose-N variant (default: all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the user texts; no model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    examples = load_examples(args.data)
    if args.variant:
        examples = [ex for ex in examples if ex.meta["variant"] == args.variant]
    if args.limit:
        examples = examples[: args.limit]
    protocols = args.protocols.split(",")
    known = [*PROTOCOLS, *RULE_PROTOCOLS, *POSITION_PROTOCOLS]
    for p in protocols:
        if p not in known:
            raise SystemExit(f"unknown protocol {p!r} (one of {', '.join(known)})")
    seeds = as_list(args.seeds, int)

    if args.dry_run:
        for ex in examples:
            for p in protocols:
                print(f"===== {ex.id}  {ex.meta['variant']}  n={ex.meta['n']}  {p}")
                for a in run_team(ex, p, placeholder_decode):
                    print(f"--- agent {a['i']}\n{a['user_text']}")
        return
    if not (args.model and args.backend and args.out):
        raise SystemExit("--model, --backend and --out are required (except with --dry-run)")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in map(json.loads, f):
                if "error" not in r:
                    done.add((r["id"], r["protocol"], r["temperature"], r["seed"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = len(examples) * len(protocols) * len(seeds)
    print(f"{len(examples)} items x {len(protocols)} protocols x {len(seeds)} seeds = {total} teams "
          f"({len(done)} already done)", flush=True)

    model = load_model(args.backend, args.model, args.device)
    t0, n_run, n_agents = time.time(), 0, 0
    with open(args.out, "a") as f:
        for seed in seeds:
            decode, tag = make_decoder(model, args.backend, args.temperature, seed)
            for p in protocols:
                for ex in examples:
                    if (ex.id, p, args.temperature, seed) in done:
                        continue
                    rec = {"id": ex.id, "variant": ex.meta["variant"], "n": ex.meta["n"],
                           "listed": ex.meta.get("listed", []), "protocol": p,
                           "temperature": args.temperature, "seed": seed, "model_id": args.model,
                           "backend": args.backend, "decoding": tag}
                    t1 = time.time()
                    try:
                        rec["agents"] = run_team(ex, p, decode)
                        for a in rec["agents"]:
                            a["seed"] = agent_seed(seed, a["i"])
                        rec["cities"] = [a["city"] for a in rec["agents"]]
                        n_agents += len(rec["agents"])
                    except Exception as e:  # keep the sweep going; failures are recorded
                        rec["error"] = repr(e)
                    rec["seconds"] = round(time.time() - t1, 3)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_run += 1
                    if n_run % 20 == 0:
                        dt = time.time() - t0
                        print(f"{n_run}/{total - len(done)} teams  {dt / max(n_agents, 1):.2f}s/agent  "
                              f"{p} seed={seed}", flush=True)
    print(f"done: {n_run} teams, {n_agents} agents in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
