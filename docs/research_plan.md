# Research plan

## The bet

Discrete chain-of-thought (CoT) forces every reasoning step through the bottleneck of natural-language tokens. That bottleneck is lossy: a single hidden state holds far more information than the one token it gets projected into. **Continuous-space reasoning** keeps that richer state across steps.

Meta's *Coconut* (Hao et al., 2024) showed this can work at billion-parameter scale. The open question we're chasing:

> Does latent-space reasoning still pay off at sub-300M scale, where the model's per-token state is already small? If yes, by how much, and on what kinds of problems?

A negative result is also publishable.

## Phases

### Phase 0 — Baseline (current)
- nanoGPT-style decoder, 10M / 124M configs
- Train on TinyStories (sanity), then a small math/logic corpus
- Confirm loss curves match published numbers

### Phase 1 — Latent thinking head
- Add a "thought mode" token. When emitted, the next K hidden states are fed back as input embeddings instead of being decoded to tokens.
- Train end-to-end: the model decides when to think vs. emit.
- Loss: language modeling on visible tokens; latent steps get no direct supervision.

### Phase 2 — Curriculum
- Synthetic arithmetic of increasing depth
- ProsQA-style logical inference
- (Stretch) GSM8K-tiny

### Phase 3 — Comparison
Three identical-budget models:
1. No-CoT baseline
2. Discrete-CoT trained with rationales
3. Noema (latent-CoT)

Compare: accuracy, tokens-per-answer, sample efficiency, depth ceiling.

### Phase 4 — Scale & share (if worth it)
If Phase 3 shows a real signal:
- Open weights and full training logs
- Write up findings (blog post + arXiv)
- Open the door to distributed contribution (rollouts, synthetic data, eval)

## Non-goals

- Beating GPT-4 at anything
- Multimodality
- Production deployment
- Training a model anyone would actually want to use as a chatbot

This is a research probe, not a product.

## Risks

- **Latent steps collapse** — model learns to ignore them. Mitigation: curriculum starting from 1 thought step, KL or auxiliary losses on the latent states.
- **No signal at small scale** — possible. That's still a result worth publishing.
- **Reproducing Coconut is harder than it looks** — they used Llama as the backbone; we're using a tiny GPT. Inductive biases differ.
