# Noema

Small-scale experiments in **latent-space reasoning** for language models.

> *Noema (νόημα)* — in phenomenology, the object of thought; the content of a thinking act.

## What this is

A research playground for one question:

> Can a small language model (≤300M parameters) learn to **reason in continuous latent space** instead of through discrete chain-of-thought tokens — and does it improve sample efficiency, reasoning depth, or speed?

Inspired by Meta's *Chain of Continuous Thought* (Coconut, 2024) and related latent-reasoning work, but reframed for hardware most researchers actually have: a single consumer GPU.

## Why small scale

Frontier mechanisms are usually invented at toy scale. Mamba, nanoGPT, TinyStories, the original Transformer — all started as sub-billion-parameter experiments. Big labs scale ideas; small labs find them.

Noema is sized to fit an 8GB GPU. Everything in this repo must be reproducible on a single RTX 3060.

## Roadmap

- **Phase 0** — nanoGPT-style baseline (124M params on a small corpus). Confirm the training loop works end-to-end.
- **Phase 1** — Add a continuous-thought head: emit latent vectors between tokens, feed them back as inputs.
- **Phase 2** — Curriculum: train on math / logic puzzles where reasoning depth matters.
- **Phase 3** — Compare latent-CoT vs. discrete-CoT vs. no-CoT on small reasoning benchmarks (GSM8K-tiny, ProsQA, custom synthetic).
- **Phase 4** — If results are interesting: write up, open-source weights, invite collaborators.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[train]"
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The second torch install replaces the CPU-only default wheel with a CUDA 12.1 build. Verify with `python -c "import torch; print(torch.cuda.is_available())"` — must print `True`.

## Hardware target

| Component | Minimum | Comfortable |
|-----------|---------|-------------|
| GPU       | RTX 3060 8GB (or any 8GB+ NVIDIA card with bf16) | 12GB+ |
| RAM       | 16GB    | 32GB |
| Disk      | 50GB free | 200GB |

CPU-only training is technically possible for the smallest configs (≤10M params) but not recommended.

## Phase 1 — latent-thinking arithmetic

Train the no-CoT baseline and the latent-CoT model, then compare accuracy on held-out problems.

```bash
python -m noema.latent_train --config configs/phase1_baseline.yaml
python -m noema.latent_train --config configs/phase1_latent.yaml

python scripts/eval_arith.py --ckpt runs/phase1_baseline/best.pt --n-thoughts 0
python scripts/eval_arith.py --ckpt runs/phase1_latent/best.pt   --n-thoughts 2
```

To probe how accuracy degrades with depth, sweep `--n-terms 3 4 5 6 7 8` against both checkpoints.

## Sampling from a checkpoint

```bash
python scripts/generate.py --ckpt runs/phase0_tiny/best.pt --prompt "Once upon a time"
```

## Status

Phase 0 — scaffolding.

## License

MIT (see `LICENSE`).
