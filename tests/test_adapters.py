import torch

from ptcdiag.decoding.adapters import LLaDA2Adapter


class _FakeModel(torch.nn.Module):
    """Records the attention mask it is called with."""

    def __init__(self, vocab=11):
        super().__init__()
        self.vocab, self.masks = vocab, []

    def forward(self, x, attention_mask=None, position_ids=None):
        self.masks.append(attention_mask)
        return type("Out", (), {"logits": torch.zeros(1, x.shape[1], self.vocab)})()


def _llada2():
    a = LLaDA2Adapter.__new__(LLaDA2Adapter)  # skip weight loading
    a.model, a.device, a.dtype, a._mask_cache = _FakeModel(), "cpu", torch.float32, {}
    return a


def test_llada2_block_causal_mask():
    a = _llada2()
    assert a.canvas_length(40, 30, 32) == 96  # rounded up to whole blocks
    a.logits(torch.zeros(1, 96, dtype=torch.long), window_end=64)
    m = a.model.masks[-1][0, 0]
    assert m.shape == (64, 64)
    assert m[0, 31] == 0 and m[0, 32] == float("-inf")  # sees its own block, not the next
    assert m[40, 10] == 0 and m[40, 63] == 0            # sees earlier blocks and its own


def test_llada2_single_block_is_full_attention():
    a = _llada2()
    assert a.canvas_length(40, 30, None) == 70  # no rounding
    a.logits(torch.zeros(1, 70, dtype=torch.long), window_end=70)
    m = a.model.masks[-1][0, 0]
    assert m.shape == (70, 70) and bool((m == 0).all())
