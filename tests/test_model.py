import torch

from noema.model import GPT, GPTConfig


def test_forward_shapes():
    cfg = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss.item() > 0


def test_backward_grads_finite():
    cfg = GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(x, targets=x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_generate_extends_sequence():
    cfg = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg).eval()
    x = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(x, max_new_tokens=5)
    assert out.shape == (1, 6)
