"""Autoregressive baselines under the same prompt and the same two modes.

free:     greedy generation of the whole answer.
skeleton: the oracle skeleton is teacher-forced; each value slot is generated
          greedily with the same per-type token restrictions as for dLLMs, until the
          model emits the slot's closing token (a quote for strings, ',' '}' for the
          rest; for arrays / dicts only outside brackets and strings, the same
          `value_end` rule as for dLLMs) or the slot length is reached.
"""

import time

import torch

from ptcdiag.decoding.constraints import (build_skeleton, slot_class, slot_closers, token_classes,
                                          token_texts, value_end)
from ptcdiag.eval.taxonomy import diagnose
from ptcdiag.prompting import parse_tool_calls, render_prompt


class ARModel:
    def __init__(self, model_id, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.device = device
        eos = self.tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if gen_eos is not None:
            self.eos_ids |= {gen_eos} if isinstance(gen_eos, int) else set(gen_eos)

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

    @torch.no_grad()
    def step(self, ids, past):
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values


@torch.no_grad()
def generate_free(ar, prompt_ids, max_new_tokens=256):
    logits, past = ar.step(prompt_ids, None)
    out = []
    for _ in range(max_new_tokens):
        t = int(logits.argmax())
        if t in ar.eos_ids:
            break
        out.append(t)
        logits, past = ar.step(torch.tensor([[t]], device=ar.device), past)
    return ar.tokenizer.decode(out, skip_special_tokens=True), len(out)


_CLOSER_CACHE = {}


def _closer_mask(tok, V, closers, device):
    """Tokens whose text starts with one of `closers` (they end a value slot)."""
    key = (tok.name_or_path, V, tuple(closers), str(device))
    if key not in _CLOSER_CACHE:
        n = min(V, len(tok))
        texts = tok.batch_decode([[j] for j in range(n)])
        flags = [any(t.startswith(c) for c in closers) for t in texts] + [False] * (V - n)
        _CLOSER_CACHE[key] = torch.tensor(flags, dtype=torch.bool, device=device)
    return _CLOSER_CACHE[key]


def _class_mask(tok, V, cls, device, _cache={}):
    key = (tok.name_or_path, V, cls, str(device))
    if key not in _cache:
        m = torch.zeros(V, dtype=torch.bool, device=device)
        m[torch.tensor([j for j in token_classes(tok)[cls] if j < V], device=device)] = True
        _cache[key] = m
    return _cache[key]


@torch.no_grad()
def generate_skeleton(ar, prompt_ids, example, slot_lengths=None, surplus=0):
    tok = ar.tokenizer
    MASK = -1
    gen_ids, slots = build_skeleton(tok, example, MASK, slot_lengths, surplus)
    slot_at = {s.positions[0]: s for s in slots}
    texts = token_texts(tok)

    logits, past = ar.step(prompt_ids, None)
    V = logits.shape[-1]
    out, i, nfe = [], 0, 1
    while i < len(gen_ids):
        if gen_ids[i] != MASK:
            j = i
            while j < len(gen_ids) and gen_ids[j] != MASK:
                j += 1
            out.extend(gen_ids[i:j])  # fixed run: one forward pass for the whole chunk
            logits, past = ar.step(torch.tensor([gen_ids[i:j]], device=ar.device), past)
            nfe += 1
            i = j
            continue
        s = slot_at[i]
        closers = slot_closers(s.type)
        nested = slot_class(s.type) == "generic"
        closer = _closer_mask(tok, V, list(closers), logits.device)
        allowed = _class_mask(tok, V, slot_class(s.type), logits.device) | closer
        slot_texts = []
        for _ in range(len(s.positions)):
            t = int(logits.masked_fill(~allowed, float("-inf")).argmax())
            slot_texts.append(texts[t] if t < len(texts) else "")
            end = value_end(slot_texts, closers, nested)
            if end is not None:
                # the closer ends the value; inside a token (e.g. '],') keep the text before it
                prefix = slot_texts[-1][:end[1]]
                if prefix:
                    pids = tok(prefix, add_special_tokens=False)["input_ids"]
                    out.extend(pids)
                    logits, past = ar.step(torch.tensor([pids], device=ar.device), past)
                    nfe += 1
                break
            out.append(t)
            logits, past = ar.step(torch.tensor([[t]], device=ar.device), past)
            nfe += 1
        i = s.positions[-1] + 1
    return tok.decode(out, skip_special_tokens=True), nfe


def run_example_ar(ar, example, mode="free", slot_lengths=None, max_new_tokens=256, surplus=0):
    prompt = render_prompt(ar.tokenizer, example)
    prompt_ids = ar.encode(prompt)
    t0 = time.time()
    if mode == "free":
        text, nfe = generate_free(ar, prompt_ids, max_new_tokens)
    else:
        text, nfe = generate_skeleton(ar, prompt_ids, example, slot_lengths, surplus)
    parsed = parse_tool_calls(text)
    rec = {
        "id": example.id, "category": example.category, "meta": example.meta,
        "model": ar.name, "mode": mode, "cfg": {"ar": True}, "cfg_tag": "ar_greedy",
        "prompt_len": int(prompt_ids.shape[1]), "text": text, "syntax_ok": parsed.syntax_ok,
        "calls": parsed.calls, "diagnosis": diagnose(example, parsed).to_dict(),
        "nfe": nfe, "seconds": round(time.time() - t0, 3),
    }
    if mode == "skeleton":
        rec["surplus"] = surplus
    return rec
