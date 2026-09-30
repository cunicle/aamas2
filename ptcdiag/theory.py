"""Proposition 1 (proposal §3.3) and its schedule-level generalisation.

Setting: n interchangeable calls, n entities, joint distribution uniform over the n!
assignments (permutations). Given the entities already committed, each still-masked
slot's exact conditional marginal is uniform over the unused entities.

If a step commits m slots at once from these marginals with r unused entities left,
all m picks are distinct with probability (r)_m / r^m (falling factorial). A whole
schedule of step sizes m_1, m_2, ... is valid with probability the product of these
terms. One step of size n gives n!/n^n.

`toy_symmetric_adapter` realises exactly this model as a ToyAdapter, so the real
Sampler can be checked against the formula (tests/test_theory.py, scripts/sim_prop1.py).
"""

import math

import torch

from ptcdiag.decoding.adapters import ToyAdapter


def p_valid_one_step(n):
    return math.factorial(n) / n ** n


def p_valid_schedule(n, sizes):
    assert sum(sizes) == n
    p, r = 1.0, n
    for m in sizes:
        p *= math.perm(r, m) / r ** m
        r -= m
    return p


def fixed_k_sizes(n, k):
    sizes = [k] * (n // k)
    if n % k:
        sizes.append(n % k)
    return sizes


def toy_symmetric_adapter(n):
    """Vocab: entities 0..n-1, MASK = n, PAD = n+1. Canvas = 1 PAD prompt token + n slots."""
    MASK, PAD = n, n + 1
    V = n + 2

    def fn(x):
        L = len(x)
        used = {int(t) for t in x.tolist() if t < n}
        logits = torch.full((L, V), -1e9)
        free = [e for e in range(n) if e not in used] or list(range(n))
        for i in range(L):
            if int(x[i]) == MASK:
                logits[i, free] = 0.0
            else:
                logits[i, int(x[i])] = 0.0
        return logits

    return ToyAdapter(fn, V, mask_id=MASK, pad_id=PAD)


def simulate(n, k, trials=2000, temperature=1.0, order="random", seed=0):
    """Fraction of valid assignments produced by the real Sampler on the toy model."""
    from ptcdiag.decoding.sampler import DecodeConfig, Sampler

    a = toy_symmetric_adapter(n)
    prompt = torch.tensor([n + 1])
    valid = 0
    for t in range(trials):
        cfg = DecodeConfig(gen_length=n, k=k, order=order, temperature=temperature,
                           seed=seed + t, record_topk=0, eos_early_stop=False)
        canvas, _ = Sampler(a, cfg).generate(prompt)
        vals = canvas[1:].tolist()
        valid += len(set(vals)) == n
    return valid / trials
