"""Dependency-violation scores for multi-token steps (Figure 5).

For each record and each step that committed >= 2 tokens, compute DVS and whether the
step touched an error position. Reports the AUROC of DVS for separating error steps
from clean steps. Observational: confounded by instance difficulty (proposal §6.5 C).

  python scripts/dvs.py --model Dream-org/Dream-v0-Instruct-7B --results results/x.jsonl \
      --data bfcl:parallel --cfg-tag confidence_k4_tnone_bfull_T0.0 --out results/x_dvs.jsonl
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.analysis.counterfactual import LOCALIZABLE, error_positions  # noqa: E402
from ptcdiag.analysis.dependency import step_dvs  # noqa: E402
from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.adapters import load_adapter  # noqa: E402
from ptcdiag.decoding.sampler import Trace  # noqa: E402
from ptcdiag.pipeline import make_constraint  # noqa: E402
from ptcdiag.prompting import render_prompt  # noqa: E402


def auroc(pos, neg):
    """P(score of a random positive > random negative), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--cfg-tag", required=True)
    ap.add_argument("--max-records", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    exs = {ex.id: ex for spec in args.data for ex in load_examples(spec, args.bfcl_dir)}
    with open(args.results) as f:
        recs = [r for r in map(json.loads, f)
                if r.get("cfg_tag") == args.cfg_tag and "trace" in r and r["id"] in exs and r["syntax_ok"]]
    recs = recs[: args.max_records]
    adapter = load_adapter(args.model, device=args.device)

    err_scores, clean_scores = [], []
    with open(args.out, "w") as f:
        for r in recs:
            ex = exs[r["id"]]
            trace = Trace.from_dict(r["trace"])
            prompt_ids = adapter.encode(render_prompt(adapter.tokenizer, ex))
            canvas = torch.cat([prompt_ids, torch.tensor(r["gen_ids"], dtype=torch.long)])
            constraint = make_constraint(adapter, ex, r["mode"])
            bad = set()
            for det in r["diagnosis"]["details"]:
                if det["label"] in LOCALIZABLE:
                    bad |= set(error_positions(adapter, r, det, trace, canvas, constraint))
            for s in trace.steps:
                if len(s.positions) < 2 or s.sequentialized:
                    continue
                res = step_dvs(adapter, ex, r, s.step)
                res["touches_error"] = bool(bad & set(s.positions))
                res["id"] = r["id"]
                res["correct"] = r["diagnosis"]["correct"]
                (err_scores if res["touches_error"] else clean_scores).append(res["dvs"])
                f.write(json.dumps(res) + "\n")
    print(f"error steps {len(err_scores)}, clean steps {len(clean_scores)}, "
          f"AUROC(DVS) = {auroc(err_scores, clean_scores):.3f}")


if __name__ == "__main__":
    main()
