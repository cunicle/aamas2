import torch

from ptcdiag.decoding.adapters import ToyAdapter
from ptcdiag.decoding.sampler import FIXED, DecodeConfig, Sampler, state_before_step
from ptcdiag.theory import toy_symmetric_adapter

PROMPT = torch.tensor([7])


def fixed_target_adapter(target, V=10, mask=8, pad=9, conf=None):
    """Every masked position prefers target[i]; `conf[i]` sets how peaked (logit gap)."""
    target = list(target)

    def fn(x):
        L = len(x)
        logits = torch.zeros(L, V)
        for i in range(1, L):
            j = i - 1
            if j < len(target):
                logits[i, target[j]] = conf[j] if conf else 5.0
        return logits

    return ToyAdapter(fn, V, mask_id=mask, pad_id=pad, eos_ids=[pad])


def run(adapter, prompt=PROMPT, **kw):
    kw.setdefault("record_topk", 0)
    kw.setdefault("eos_early_stop", False)
    return Sampler(adapter, DecodeConfig(**kw)).generate(prompt)


def test_every_position_committed_once_and_k_per_step():
    a = fixed_target_adapter([1, 2, 3, 4, 5, 6])
    canvas, tr = run(a, gen_length=6, k=2)
    assert canvas[1:].tolist() == [1, 2, 3, 4, 5, 6]
    assert sorted(p for s in tr.steps for p in s.positions) == list(range(1, 7))
    assert [len(s.positions) for s in tr.steps] == [2, 2, 2]
    assert all(t >= 0 for t in tr.commit_step)
    assert tr.nfe == 3


def test_blocks_are_semi_autoregressive():
    # later positions are more confident, but blocks force left blocks first
    a = fixed_target_adapter([1, 2, 3, 4], conf=[1.0, 2.0, 3.0, 4.0])
    _, tr = run(a, gen_length=4, k=1, block_length=2)
    assert [s.positions[0] for s in tr.steps] == [2, 1, 4, 3]
    _, tr = run(a, gen_length=4, k=1)
    assert [s.positions[0] for s in tr.steps] == [4, 3, 2, 1]


def test_left_to_right_order():
    a = fixed_target_adapter([1, 2, 3, 4], conf=[1.0, 2.0, 3.0, 4.0])
    _, tr = run(a, gen_length=4, k=1, order="left_to_right")
    assert [s.positions[0] for s in tr.steps] == [1, 2, 3, 4]


def test_threshold_commits_all_confident_at_least_k():
    a = fixed_target_adapter([1, 2, 3, 4], conf=[9.0, 9.0, 0.5, 0.5])
    _, tr = run(a, gen_length=4, k=1, threshold=0.9)
    assert sorted(tr.steps[0].positions) == [1, 2]      # both above threshold
    assert all(len(s.positions) == 1 for s in tr.steps[1:])


def test_symmetric_greedy_collides_and_sequential_fixes():
    n = 3
    a = toy_symmetric_adapter(n)
    pad = torch.tensor([a.pad_id])
    canvas, tr = run(a, pad, gen_length=n, k=n)
    assert len(set(canvas[1:].tolist())) == 1           # greedy ties: every slot picks entity 0
    # counterfactual: the same single step, committed one position at a time
    canvas2, tr2 = run(a, pad, gen_length=n, k=n, sequentialize_steps=(0,))
    assert sorted(canvas2[1:].tolist()) == list(range(n))
    assert tr2.steps[0].sequentialized and tr2.nfe == n
    # one position at a time from the start is always valid
    canvas3, _ = run(a, pad, gen_length=n, k=1)
    assert sorted(canvas3[1:].tolist()) == list(range(n))


def test_resume_from_state_reproduces():
    a = fixed_target_adapter([1, 2, 3, 4, 5, 6], conf=[1, 6, 2, 5, 3, 4])
    canvas, tr = run(a, gen_length=6, k=2)
    for step in range(3):
        init = state_before_step(tr, canvas, step, a.mask_id)
        c2, _ = Sampler(a, DecodeConfig(gen_length=6, k=2, record_topk=0, eos_early_stop=False)) \
            .generate(PROMPT, init_gen=init, start_step=step)
        assert torch.equal(c2, canvas)


def test_absolute_blocks_and_padding():
    # absolute blocks of 4 with a 1-token prompt: first gen block is positions 1..3
    a = fixed_target_adapter([1, 2, 3, 4, 5], conf=[1.0, 2.0, 3.0, 4.0, 5.0])
    a.absolute_blocks = True
    a.canvas_length = lambda P, G, B: -(-(P + G) // B) * B
    canvas, tr = run(a, gen_length=5, k=1, block_length=4)
    assert len(canvas) == 8
    first = [s.positions[0] for s in tr.steps[:3]]
    assert sorted(first) == [1, 2, 3]


def test_fixed_positions_untouched():
    a = fixed_target_adapter([1, 2, 3, 4])
    init = [5, a.mask_id, 6, a.mask_id]
    canvas, tr = Sampler(a, DecodeConfig(gen_length=4, k=1, record_topk=0, eos_early_stop=False)) \
        .generate(PROMPT, init_gen=init)
    assert canvas[1:].tolist() == [5, 2, 6, 4]
    assert tr.commit_step[0] == FIXED and tr.commit_step[2] == FIXED
