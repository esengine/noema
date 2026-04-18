import torch

from noema.arith import ArithBatcher, make_problem
from noema.latent import LatentGPT
from noema.model import GPTConfig
from noema.tokenizer import ArithTokenizer


def test_tokenizer_roundtrip():
    tok = ArithTokenizer()
    ids = tok.encode("3+5=8")
    assert tok.decode(ids) == "3+5=8"


def test_arith_problem_correctness():
    import random
    p = make_problem(4, random.Random(0))
    assert p.answer == sum(p.operands) % 10


def test_batcher_layout():
    tok = ArithTokenizer()
    b = ArithBatcher(tok, n_terms=3, n_thoughts=2, seed=0)
    x, y, m = b.sample(batch_size=2, device="cpu")
    assert x.shape == y.shape == m.shape
    assert m.sum().item() == 2 * 2
    assert (x[m] == tok.pad_id).all()


def test_latent_forward_zero_thoughts_matches_plain():
    torch.manual_seed(0)
    tok = ArithTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = LatentGPT(cfg).eval()
    x = torch.randint(4, tok.vocab_size, (2, 16))
    mask = torch.zeros_like(x, dtype=torch.bool)
    latent_logits, _ = model.forward_latent(x, mask)
    plain_logits, _ = model(x)
    assert torch.allclose(latent_logits, plain_logits, atol=1e-5)


def test_latent_forward_with_thoughts_runs():
    torch.manual_seed(0)
    tok = ArithTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = LatentGPT(cfg)
    b = ArithBatcher(tok, n_terms=3, n_thoughts=2, seed=0)
    x, y, m = b.sample(batch_size=2, device="cpu")
    logits, loss = model.forward_latent(x, m, y)
    assert logits.shape == (2, x.shape[1], tok.vocab_size)
    assert loss.item() > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
