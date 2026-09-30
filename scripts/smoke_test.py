"""GPU smoke test: run this first on a new machine, per model.

1. LLaDA-8B:  our Sampler must reproduce the official `generate` (low_confidence,
              temperature 0) token for token, for k = gen_length / steps.
2. LLaDA2.0:  our Sampler (k=1, threshold, absolute blocks) must reproduce
              `model.generate(threshold=..., steps=block_length)` once its sampler is
              made greedy (the official one samples even at temperature 0).
3. Dream:     no exact reference schedule exists for fixed k, so it only runs one
              BFCL example end to end and prints the output.
All:          one BFCL parallel example in free and skeleton mode, and a determinism
              check (resume at step 3 without intervention == original).

  python scripts/smoke_test.py --model GSAI-ML/LLaDA-8B-Instruct
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcdiag.analysis.counterfactual import reproduces  # noqa: E402
from ptcdiag.data import load_examples  # noqa: E402
from ptcdiag.decoding.adapters import load_adapter  # noqa: E402
from ptcdiag.decoding.sampler import DecodeConfig, Sampler  # noqa: E402
from ptcdiag.pipeline import run_example  # noqa: E402
from ptcdiag.prompting import render_prompt  # noqa: E402


@torch.no_grad()
def llada_reference(model, prompt, steps, gen_length, block_length, mask_id=126336):
    """ML-GSAI/LLaDA generate.py (MIT), low_confidence remasking, temperature 0, no CFG."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    for nb in range(num_blocks):
        s, e = prompt.shape[1] + nb * block_length, prompt.shape[1] + (nb + 1) * block_length
        block_mask_index = x[:, s:e] == mask_id
        mask_num = block_mask_index.sum(dim=1, keepdim=True)
        base, rem = mask_num // steps, mask_num % steps
        ntt = torch.zeros(1, steps, device=x.device, dtype=torch.int64) + base
        ntt[0, :rem[0]] += 1
        for i in range(steps):
            mask_index = x == mask_id
            logits = model(x).logits
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, -1, x0[..., None]).squeeze(-1)
            x0_p[:, e:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, -float("inf"))
            _, idx = torch.topk(conf[0], k=int(ntt[0, i]))
            x[0, idx] = x0[0, idx]
    return x[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bfcl-dir", default="data/bfcl")
    args = ap.parse_args()

    a = load_adapter(args.model, device=args.device)
    ex = load_examples("bfcl:parallel", args.bfcl_dir)[0]
    prompt_ids = a.encode(render_prompt(a.tokenizer, ex))
    print(f"adapter={a.name} prompt_len={len(prompt_ids)}")

    if a.name == "llada":
        G, B, k = 64, 32, 2
        ref = llada_reference(a.model, prompt_ids[None].to(a.device), steps=G // k, gen_length=G, block_length=B)
        cfg = DecodeConfig(gen_length=G, block_length=B, k=k, eos_early_stop=False)
        ours, _ = Sampler(a, cfg).generate(prompt_ids)
        print("LLaDA reference match:", bool((ref.cpu() == ours.cpu()).all()))
    if a.name == "llada2":
        # The official sampler draws with torch.multinomial even at temperature 0, so for
        # an exact comparison swap in greedy selection; this checks our block layout,
        # attention mask and threshold rule, not the official sampling noise.
        def greedy(logits, temperature=0.0, top_k=0, top_p=1.0):
            probs = torch.softmax(logits.float().reshape(-1, logits.shape[-1]), -1)
            p, tok = probs.max(-1)
            return tok.view(*logits.shape[:-1]), p.view(*logits.shape[:-1])

        a.model._sample_with_temperature_topk_topp = greedy
        G, B, tau = 64, 32, 0.95
        ref = a.model.generate(inputs=prompt_ids[None], gen_length=G, block_length=B, steps=B,
                               threshold=tau, temperature=0.0)
        cfg = DecodeConfig(gen_length=G, block_length=B, k=1, threshold=tau)
        ours, tr = Sampler(a, cfg).generate(prompt_ids)
        ours_gen = ours[tr.gen_start:].tolist()
        ref_gen = ref[0].tolist()
        print("LLaDA2.0 reference match (up to EOS):", ours_gen[:len(ref_gen)] == ref_gen)

    for mode in ("free", "skeleton"):
        cfg = DecodeConfig(gen_length=256, block_length=32 if a.name == "llada2" else None, k=2)
        rec = run_example(a, ex, cfg, mode)
        print(f"[{mode}] syntax_ok={rec['syntax_ok']} labels={rec['diagnosis']['labels']} "
              f"nfe={rec['nfe']} {rec['seconds']}s\n  {rec['text'][:300]!r}")
        print(f"  determinism (resume at step 3): {reproduces(a, ex, rec, step=3)}")


if __name__ == "__main__":
    main()
