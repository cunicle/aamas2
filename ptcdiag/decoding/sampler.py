"""One traced masked-diffusion sampler for every dLLM (proposal §6.4-6.5).

A decoding *step* = one forward pass, then committing a set S_t of masked positions
at once, each from its own marginal. The knobs the experiments intervene on:

  k            minimum number of positions committed per step
  threshold    if set, commit every candidate whose chosen-token probability is
               >= threshold, provided at least k qualify (else the top-k by score).
               k=1 + threshold = Fast-dLLM; k=1 + threshold + absolute blocks = LLaDA2.0.
  order        score used to rank candidates: confidence | margin | entropy |
               random | left_to_right
  block_length semi-autoregressive blocks; only masks in the first unfinished block
               are candidates. None = one block over the whole generation region.

With k = gen_length / steps, order=confidence, temperature 0 and the same block
length, this reproduces LLaDA's reference `generate` (low_confidence remasking);
see scripts/smoke_test.py.

Counterfactual sequentialisation: for a step t in `sequentialize_steps`, the
selected set S_t is committed one position at a time (highest score first), with a
fresh forward pass before each commit. Only the positions of S_t are re-decided;
the rest of the schedule is unchanged. Resuming from a mid-trajectory state
(`init_gen` + `start_step`) lets the replay skip the unchanged prefix.
"""

from dataclasses import asdict, dataclass, field

import torch

ORDERS = ("confidence", "margin", "entropy", "random", "left_to_right")

# Trace.commit_step codes for positions not committed by a decoding step.
FIXED = -1      # given before decoding (constraint skeleton / padding)
PENDING = -2    # still masked (only seen mid-generation)
EARLY_PAD = -3  # filled with pad_id by EOS early stopping


@dataclass
class DecodeConfig:
    gen_length: int = 256
    block_length: int = None
    k: int = 1
    threshold: float = None
    order: str = "confidence"
    temperature: float = 0.0
    seed: int = 0
    record_topk: int = 5
    eos_early_stop: bool = True  # after a finished block, stop if an EOS is already committed
    sequentialize_steps: tuple = ()

    def to_dict(self):
        return asdict(self)

    def tag(self):
        b = self.block_length if self.block_length else "full"
        t = self.threshold if self.threshold is not None else "none"
        return f"{self.order}_k{self.k}_t{t}_b{b}_T{self.temperature}"


@dataclass
class StepRecord:
    step: int
    positions: list          # canvas positions committed at this step, in commit order
    tokens: list
    probs: list              # probability of the committed token when it was chosen
    scores: list
    topk_ids: list = field(default_factory=list)
    topk_probs: list = field(default_factory=list)
    n_candidates: int = 0
    sequentialized: bool = False
    # positions the constraint set to padding after this step's commits (the rest of a
    # value slot once its closing token is in); not decided by the model, so they are
    # not part of `positions` and do not count towards k
    forced: list = field(default_factory=list)


@dataclass
class Trace:
    prompt_len: int
    gen_start: int
    gen_end: int
    commit_step: list        # per gen-region position: step index, or FIXED / EARLY_PAD
    steps: list
    nfe: int = 0

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["steps"] = [StepRecord(**s) for s in d["steps"]]
        return cls(**d)

    def step_of(self, pos):
        return self.commit_step[pos - self.gen_start]

    def co_committed(self, pos):
        """Other canvas positions committed in the same step as `pos`."""
        t = self.step_of(pos)
        if t < 0:
            return []
        rec = next(s for s in self.steps if s.step == t)
        return [p for p in rec.positions if p != pos]


def _block_bounds(adapter, cfg, gen_start, gen_end, pos):
    B = cfg.block_length
    if not B:
        return gen_start, gen_end
    offset = 0 if adapter.absolute_blocks else gen_start
    bs = offset + ((pos - offset) // B) * B
    return max(bs, gen_start), min(bs + B, gen_end)


def _scores(order, probs, chosen_p, positions, gen):
    if order == "confidence":
        return chosen_p
    if order == "margin":
        top2 = probs.topk(2, dim=-1).values
        return top2[:, 0] - top2[:, 1]
    if order == "entropy":
        return (probs * torch.log(probs + 1e-10)).sum(-1)
    if order == "random":
        return torch.rand(len(positions), generator=gen, device="cpu").to(probs.device)
    if order == "left_to_right":
        return -positions.to(probs.dtype)
    raise ValueError(order)


class Sampler:
    def __init__(self, adapter, cfg: DecodeConfig, constraint=None):
        assert cfg.order in ORDERS, cfg.order
        self.a = adapter
        self.cfg = cfg
        self.constraint = constraint
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    # -- one forward pass: token proposals for candidate positions
    def _propose(self, x, cand, window_end):
        logits = self.a.logits(x, window_end)[cand].float()
        if self.constraint is not None:
            logits = self.constraint.filter(x, cand, logits)
        T = self.cfg.temperature
        probs = torch.softmax(logits / T if T > 0 else logits, dim=-1)
        if T > 0:
            chosen = torch.multinomial(probs.cpu(), 1, generator=self.gen).squeeze(-1).to(probs.device)
        else:
            chosen = probs.argmax(-1)
        chosen_p = probs.gather(-1, chosen[:, None]).squeeze(-1)
        return probs, chosen, chosen_p

    def _record(self, probs, idx):
        k = self.cfg.record_topk
        if not k:
            return [], []
        top = probs[idx].topk(min(k, probs.shape[-1]), dim=-1)
        return top.indices.tolist(), [[round(v, 5) for v in row] for row in top.values.tolist()]

    @torch.no_grad()
    def generate(self, prompt_ids, init_gen=None, start_step=0, stop_before_step=None):
        """Decode. Returns (canvas LongTensor [L], Trace).

        prompt_ids: 1-D LongTensor.
        init_gen:   initial generation region (list/1-D tensor, mask_id where undecided).
                    Default: constraint.initial_gen() if available, else all masks.
        start_step: step index to use for the first step (for resuming).
        stop_before_step: stop (without committing) when this step index is reached;
                    used to recover the state before a given step.
        """
        a, cfg = self.a, self.cfg
        dev = a.device
        P = len(prompt_ids)
        G = a.canvas_length(P, cfg.gen_length, cfg.block_length) - P
        # always call initial_gen: it also tells the constraint where the prompt ends
        base = self.constraint.initial_gen(G, P) if self.constraint is not None else None
        if init_gen is None:
            init_gen = base
        if init_gen is None:
            init_gen = [a.mask_id] * G
        init_gen = torch.as_tensor(init_gen, dtype=torch.long)
        if len(init_gen) < G:  # pad with EOS so block-aligned models get a full canvas
            init_gen = torch.cat([init_gen, torch.full((G - len(init_gen),), a.pad_id, dtype=torch.long)])
        x = torch.cat([prompt_ids.cpu().long(), init_gen[:G]]).to(dev)[None]
        gs, ge = P, P + G

        commit_step = [FIXED if int(t) != a.mask_id else PENDING for t in x[0, gs:ge].tolist()]
        trace = Trace(prompt_len=P, gen_start=gs, gen_end=ge, commit_step=commit_step, steps=[])
        seq_steps = set(cfg.sequentialize_steps)
        step = start_step

        while True:
            masked = (x[0, gs:ge] == a.mask_id).nonzero().flatten() + gs
            if len(masked) == 0:
                break
            bs, be = _block_bounds(a, cfg, gs, ge, int(masked[0]))
            cand = masked[(masked >= bs) & (masked < be)]
            if stop_before_step is not None and step >= stop_before_step:
                break
            window_end = be if a.absolute_blocks else None

            probs, chosen, chosen_p = self._propose(x, cand, window_end)
            trace.nfe += 1
            scores = _scores(cfg.order, probs, chosen_p, cand, self.gen)
            kk = min(cfg.k, len(cand))
            sel = scores.topk(kk).indices
            if cfg.threshold is not None:
                above = (chosen_p > cfg.threshold).nonzero().flatten()
                if len(above) >= kk:
                    sel = above[scores[above].argsort(descending=True)]

            if step in seq_steps and len(sel) > 1:
                rec = self._commit_sequential(x, cand[sel], window_end, step, len(cand), trace,
                                              first=(probs[sel], chosen[sel], chosen_p[sel]))
            else:
                pos = cand[sel]
                x[0, pos] = chosen[sel]
                ids, ps = self._record(probs, sel)
                rec = StepRecord(step=step, positions=pos.tolist(), tokens=chosen[sel].tolist(),
                                 probs=[round(v, 5) for v in chosen_p[sel].tolist()],
                                 scores=[round(v, 5) for v in scores[sel].tolist()],
                                 topk_ids=ids, topk_probs=ps, n_candidates=len(cand),
                                 forced=self._after_commit(x))
            for p in rec.positions + rec.forced:
                trace.commit_step[p - gs] = step
            trace.steps.append(rec)
            step += 1

            # early stop once a block is complete and an EOS has been committed
            early_ok = self.constraint is None or getattr(self.constraint, "early_stop_ok", True)
            if cfg.eos_early_stop and early_ok and cfg.block_length \
                    and not (x[0, bs:be] == a.mask_id).any():
                region = x[0, gs:be]
                if any(bool((region == e).any()) for e in a.eos_ids):
                    rest = (x[0, gs:ge] == a.mask_id).nonzero().flatten() + gs
                    x[0, rest] = a.pad_id
                    for p in rest.tolist():
                        trace.commit_step[p - gs] = EARLY_PAD
                    break

        return x[0], trace

    def _after_commit(self, x):
        hook = getattr(self.constraint, "after_commit", None)
        return hook(x) if hook is not None else []

    def _commit_sequential(self, x, positions, window_end, step, n_cand, trace, first):
        """Commit `positions` one at a time, re-running the model before each commit.

        `first` holds the proposals already computed for this state, so the first
        commit reuses them and only the following commits cost a forward pass.
        """
        remaining = positions.clone()
        rec = StepRecord(step=step, positions=[], tokens=[], probs=[], scores=[],
                         n_candidates=n_cand, sequentialized=True)
        while len(remaining):
            if first is not None:
                probs, chosen, chosen_p = first
                first = None
            else:
                trace.nfe += 1
                probs, chosen, chosen_p = self._propose(x, remaining, window_end)
            scores = _scores(self.cfg.order, probs, chosen_p, remaining, self.gen)
            j = int(scores.argmax())
            p = int(remaining[j])
            x[0, p] = chosen[j]
            ids, ps = self._record(probs, torch.tensor([j], device=probs.device))
            rec.positions.append(p)
            rec.tokens.append(int(chosen[j]))
            rec.probs.append(round(float(chosen_p[j]), 5))
            rec.scores.append(round(float(scores[j]), 5))
            rec.topk_ids += ids
            rec.topk_probs += ps
            forced = self._after_commit(x)  # may pad positions of this step that are still pending
            rec.forced += forced
            done = set(forced) | {p}
            remaining = remaining[[i for i, q in enumerate(remaining.tolist()) if q not in done]]
        return rec


def state_before_step(trace, canvas, step, mask_id):
    """Generation-region tokens as they were right before `step` was committed.

    Valid because committed tokens are never remasked by this sampler.
    """
    gen = canvas[trace.gen_start:trace.gen_end].clone()
    for i, t in enumerate(trace.commit_step):
        if t >= step or t == EARLY_PAD:
            gen[i] = mask_id
    return gen
