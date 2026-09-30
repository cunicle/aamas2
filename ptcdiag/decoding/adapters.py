"""Model adapters: a uniform `logits(x)` view over different dLLMs.

Every adapter returns logits aligned so that row i scores the token at canvas
position i (Dream's shift is undone here), for a batch of one canvas.

The three HF adapters follow each model's reference sampler:
  LLaDA-8B  ML-GSAI/LLaDA generate.py: full attention, `model(x).logits`.
  Dream-7B  generation_utils.py: `model(x, "full", None)`, logits shifted right by one.
  LLaDA2.0  modeling_llada2_moe.generate: block-causal attention over absolute blocks,
            forward only up to the end of the current block.
They have not been run on real weights in this repository yet; run
scripts/smoke_test.py on a GPU before trusting them.
"""

import math

import torch


class Adapter:
    name = "base"
    absolute_blocks = False  # block boundaries at absolute positions (LLaDA2.0)

    def __init__(self, tokenizer, mask_id, eos_ids, pad_id, device="cpu"):
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.eos_ids = list(eos_ids)
        self.pad_id = pad_id
        self.device = device

    def logits(self, x, window_end=None):
        """x: LongTensor [1, L]. Returns FloatTensor [L', V] (L' = window_end or L)."""
        raise NotImplementedError

    def canvas_length(self, prompt_len, gen_length, block_length):
        return prompt_len + gen_length

    def encode(self, text):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    @property
    def special_ids(self):
        return set(self.eos_ids) | {self.pad_id, self.mask_id}


def _load(model_id, dtype, device):
    """Load the class with the LM head.

    LLaDA2.0 maps AutoModel to its headless base model, so prefer AutoModelForCausalLM
    whenever the config's auto_map provides it (Dream only registers AutoModel).
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cls = AutoModelForCausalLM if "AutoModelForCausalLM" in (getattr(config, "auto_map", None) or {}) \
        else AutoModel
    model = cls.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
    return tok, model.to(device).eval()


class LLaDAAdapter(Adapter):
    name = "llada"

    def __init__(self, model_id="GSAI-ML/LLaDA-8B-Instruct", device="cuda", dtype=torch.bfloat16):
        tok, self.model = _load(model_id, dtype, device)
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        super().__init__(tok, mask_id=126336, eos_ids=[tok.eos_token_id, eot],
                         pad_id=tok.eos_token_id, device=device)

    def logits(self, x, window_end=None):
        return self.model(x).logits[0]


class DreamAdapter(Adapter):
    name = "dream"

    def __init__(self, model_id="Dream-org/Dream-v0-Instruct-7B", device="cuda", dtype=torch.bfloat16):
        tok, self.model = _load(model_id, dtype, device)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        super().__init__(tok, mask_id=tok.mask_token_id, eos_ids=[tok.eos_token_id, im_end],
                         pad_id=tok.eos_token_id, device=device)

    def logits(self, x, window_end=None):
        logits = self.model(x, "full", None).logits
        # Dream predicts position i from hidden state i-1.
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[0]


class LLaDA2Adapter(Adapter):
    name = "llada2"
    absolute_blocks = True

    def __init__(self, model_id="inclusionAI/LLaDA2.0-mini", device="cuda", dtype=torch.bfloat16,
                 block_length=32):
        tok, self.model = _load(model_id, dtype, device)
        role_end = tok.convert_tokens_to_ids("<|role_end|>")
        super().__init__(tok, mask_id=156895, eos_ids=[156892, role_end], pad_id=156892, device=device)
        self.dtype = dtype
        self.block_length = block_length
        self._mask_cache = {}

    def canvas_length(self, prompt_len, gen_length, block_length):
        # Called at the start of every generation: keeps the attention mask's block
        # size in sync with the sampler's DecodeConfig.block_length.
        if not block_length:
            raise ValueError("LLaDA2.0 is a block-diffusion model; set block_length (e.g. 32)")
        self.block_length = block_length
        return math.ceil((prompt_len + gen_length) / block_length) * block_length

    def _block_mask(self, n_blocks, B):
        key = (n_blocks, B)
        if key not in self._mask_cache:
            m = torch.tril(torch.ones(n_blocks, n_blocks, device=self.device))
            m = m.repeat_interleave(B, 0).repeat_interleave(B, 1)[None, None]
            self._mask_cache[key] = m.log().to(self.dtype)  # 0 -> -inf, 1 -> 0
        return self._mask_cache[key]

    def logits(self, x, window_end=None):
        B = self.block_length
        end = window_end or x.shape[1]
        assert end % B == 0, "window must end on a block boundary"
        mask = self._block_mask(x.shape[1] // B, B)[:, :, :end, :end]
        pos = torch.arange(end, device=x.device)[None]
        return self.model(x[:, :end], attention_mask=mask, position_ids=pos).logits[0]


class ToyAdapter(Adapter):
    """Tiny analytic "dLLM" for tests and for simulating Proposition 1.

    `fn(x) -> logits [L, V]` is any callable; no neural network involved.
    """

    name = "toy"

    def __init__(self, fn, vocab_size, mask_id, eos_ids=(), pad_id=None, tokenizer=None,
                 absolute_blocks=False):
        super().__init__(tokenizer, mask_id=mask_id, eos_ids=eos_ids,
                         pad_id=pad_id if pad_id is not None else mask_id, device="cpu")
        self.fn = fn
        self.vocab_size = vocab_size
        self.absolute_blocks = absolute_blocks

    def logits(self, x, window_end=None):
        return self.fn(x[0])


MODELS = {
    "llada": LLaDAAdapter,
    "dream": DreamAdapter,
    "llada2": LLaDA2Adapter,
}


def load_adapter(model_id, device="cuda", dtype=torch.bfloat16, **kw):
    name = model_id.lower()
    if "llada2" in name:
        return LLaDA2Adapter(model_id, device=device, dtype=dtype, **kw)
    if "llada" in name:
        return LLaDAAdapter(model_id, device=device, dtype=dtype)
    if "dream" in name:
        return DreamAdapter(model_id, device=device, dtype=dtype)
    raise ValueError(f"no adapter for {model_id}")
