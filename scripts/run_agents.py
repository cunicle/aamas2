"""Experiment B: teams of n agents answer the choose-N items, one call per agent.

  python scripts/run_agents.py --model Qwen/Qwen2.5-7B-Instruct --backend ar \
      --data probe:data/choose.jsonl --protocols sim-anon,sim-label,turn-anon,turn-label \
      --out results/qwen/agents.jsonl
  python scripts/run_agents.py --data probe:data/choose.jsonl --protocols sim-anon,turn-label \
      --dry-run --limit 3      # only print the agents' user texts (no model)

Per item x protocol x seed, agents 1..n are built (ptcdiag/data/agents.py) and decoded
one after another with the oracle skeleton of one call: Qwen with `run_example_ar`, Dream
with `run_example` (k=1, confidence order, no blocks). Turn-taking agents see the calls of
the agents before them; simultaneous agents see none. One JSONL record per team, with
every agent's user text, output and city. Re-running with the same --out skips the teams
that are already there.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.data.agents import PROTOCOLS, agent_example  # noqa: E402


def as_list(s, typ):
    return [typ(v) for v in s.split(",")]


def run_team(ex, protocol, decode):
    """Agents 1..n of the team for `ex` under `protocol`.

    decode(agent_example) -> (text, syntax_ok, calls). An agent passes on its parsed call,
    or its raw text when the output does not parse into exactly one call."""
    n = ex.meta["n"]
    observe, _ = PROTOCOLS[protocol]
    previous, agents = [], []
    for i in range(1, n + 1):
        aex = agent_example(ex, i, n, protocol, previous if observe else ())
        text, syntax_ok, calls = decode(aex)
        one = syntax_ok and len(calls) == 1
        city = calls[0]["arguments"].get("city") if one else None
        agents.append({"i": i, "user_text": aex.messages[0]["content"], "text": text,
                       "syntax_ok": syntax_ok, "city": city})
        previous.append(calls[0] if one else text)
    return agents


def placeholder_decode(aex):
    """--dry-run: agent i 'calls' a placeholder city, so turn-taking texts show the format."""
    call = {"name": "get_weather", "arguments": {"city": f"<city of agent {aex.meta['agent']}>"}}
    return json.dumps([call]), True, [call]


def load_model(backend, model_id, device):
    if backend == "ar":
        from ptcdiag.decoding.ar import ARModel

        return ARModel(model_id, device=device)
    from ptcdiag.decoding.adapters import load_adapter

    return load_adapter(model_id, device=device)


def make_decoder(model, backend, temperature, seed):
    """(decode, decoding tag) for one temperature / seed."""
    if backend == "ar":
        from ptcdiag.decoding.ar import run_example_ar

        def decode(aex):
            r = run_example_ar(model, aex, mode="skeleton")
            return r["text"], r["syntax_ok"], r["calls"]

        return decode, "ar_greedy"

    from ptcdiag.decoding.sampler import DecodeConfig
    from ptcdiag.pipeline import run_example

    cfg = DecodeConfig(block_length=None, k=1, order="confidence", temperature=temperature, seed=seed)

    def decode(aex):
        r = run_example(model, aex, cfg, mode="skeleton", keep_trace=False)
        return r["text"], r["syntax_ok"], r["calls"]

    return decode, cfg.tag()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["dllm", "ar"])
    ap.add_argument("--data", default="probe:data/choose.jsonl")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the user texts; no model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]
    protocols = args.protocols.split(",")
    for p in protocols:
        if p not in PROTOCOLS:
            raise SystemExit(f"unknown protocol {p!r} (one of {', '.join(PROTOCOLS)})")
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
    if args.backend == "ar" and args.temperature:
        raise SystemExit("the AR skeleton decoding is greedy only")

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
