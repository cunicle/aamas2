"""Dependency among co-committed tokens (proposal §6.5 C).

step_dvs: the pseudo-cost of Wen et al. (arXiv 2608.25505) turned into a per-step
"dependency-violation score":

    DVS_t = sum_j [ log p(x_j | ctx) - log p(x_j | ctx, x_<j) ]

over the tokens committed together at step t (in commit order). It compares the
mean-field probability each token was chosen with against its probability once the
earlier tokens of the same step are revealed. Large positive values mean the
committed combination is much less coherent than each token looked on its own.
Cost: m forward passes for a step of size m.

pairwise_tv: DEMASK's D_ij (arXiv 2604.02560) restricted to one step:
TV( p(x_j | ctx), p(x_j | ctx, x_i) ). Cost: m+1 forward passes.
"""

import dataclasses

import torch

from ptcdiag.decoding.sampler import DecodeConfig, Trace, _block_bounds, state_before_step
from ptcdiag.pipeline import make_constraint
from ptcdiag.prompting import render_prompt


def _setup(adapter, example, record, step):
    cfg = DecodeConfig(**{k: v for k, v in record["cfg"].items()
                          if k in {f.name for f in dataclasses.fields(DecodeConfig)}})
    prompt_ids = adapter.encode(render_prompt(adapter.tokenizer, example))
    trace = Trace.from_dict(record["trace"])
    canvas = torch.cat([prompt_ids, torch.tensor(record["gen_ids"], dtype=torch.long)])
    rec = next(s for s in trace.steps if s.step == step)
    constraint = make_constraint(adapter, example, record["mode"])
    G = adapter.canvas_length(len(prompt_ids), cfg.gen_length, cfg.block_length) - len(prompt_ids)
    constraint.initial_gen(G, len(prompt_ids))
    state = state_before_step(trace, canvas, step, adapter.mask_id)
    x = torch.cat([prompt_ids, state]).to(adapter.device)[None]
    _, be = _block_bounds(adapter, cfg, trace.gen_start, trace.gen_end, rec.positions[0])
    window = be if adapter.absolute_blocks else None
    return cfg, rec, constraint, x, window


def _logprobs(adapter, constraint, x, positions, window):
    pos = torch.tensor(positions, device=x.device)
    logits = adapter.logits(x, window)[pos].float()
    logits = constraint.filter(x, pos, logits)
    return torch.log_softmax(logits, -1)


@torch.no_grad()
def step_dvs(adapter, example, record, step):
    cfg, rec, constraint, x, window = _setup(adapter, example, record, step)
    pos, tok = rec.positions, rec.tokens
    lp = _logprobs(adapter, constraint, x, pos, window)
    mf = [float(lp[j, t]) for j, t in enumerate(tok)]
    seq = [mf[0]]
    for j in range(1, len(pos)):
        x[0, pos[j - 1]] = tok[j - 1]
        lpj = _logprobs(adapter, constraint, x, [pos[j]], window)
        seq.append(float(lpj[0, tok[j]]))
    per = [m - s for m, s in zip(mf, seq)]
    return {"step": step, "m": len(pos), "dvs": sum(per), "max_term": max(per) if per else 0.0,
            "per_token": per, "positions": pos}


@torch.no_grad()
def pairwise_tv(adapter, example, record, step):
    cfg, rec, constraint, x, window = _setup(adapter, example, record, step)
    pos, tok = rec.positions, rec.tokens
    base = _logprobs(adapter, constraint, x, pos, window).exp()
    m = len(pos)
    D = torch.zeros(m, m)
    for i in range(m):
        xi = x.clone()
        xi[0, pos[i]] = tok[i]
        cond = _logprobs(adapter, constraint, xi, pos, window).exp()
        D[i] = 0.5 * (cond - base).abs().sum(-1).cpu()
        D[i, i] = 0.0
    return {"step": step, "positions": pos, "tv": D.tolist()}
