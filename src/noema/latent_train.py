from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from noema.arith import ArithBatcher
from noema.latent import LatentGPT
from noema.model import GPTConfig
from noema.tokenizer import ArithTokenizer


@dataclass
class LatentTrainConfig:
    n_terms: int = 5
    thought_schedule: list[list[int]] = field(default_factory=lambda: [[0, 0], [2000, 1], [5000, 2]])
    eval_n_thoughts: int = 2

    block_size: int = 64
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.0

    batch_size: int = 64
    max_steps: int = 10000
    eval_every: int = 500
    eval_iters: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    device: str = "cuda"
    dtype: str = "bfloat16"
    out_dir: str = "runs/phase1_latent"
    seed: int = 1337

    stop_grad_thoughts: bool = False

    early_stop_patience: int = 0
    early_stop_min_step: int = 20000
    early_stop_min_acc: float = 0.15


def load_config(path: str) -> LatentTrainConfig:
    with open(path, "rb") as f:
        raw = yaml.safe_load(f.read().decode("utf-8")) or {}
    return LatentTrainConfig(**raw)


def current_k(step: int, schedule: list[list[int]]) -> int:
    k = schedule[0][1]
    for threshold, value in schedule:
        if step >= threshold:
            k = value
    return k


def get_lr(step: int, cfg: LatentTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def eval_accuracy(model: LatentGPT, tok: ArithTokenizer, cfg: LatentTrainConfig, k: int, n_batches: int) -> float:
    model.eval()
    batcher = ArithBatcher(tok, cfg.n_terms, k, seed=9999)
    ans_pos = batcher.answer_position()
    correct = 0
    total = 0
    for _ in range(n_batches):
        x, y, m = batcher.sample(cfg.batch_size, device=cfg.device)
        logits, _ = model.forward_latent(x, m)
        preds = logits[:, ans_pos - 1].argmax(dim=-1)
        true = x[:, ans_pos]
        correct += (preds == true).sum().item()
        total += preds.numel()
    model.train()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    tok = ArithTokenizer()
    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = LatentGPT(model_cfg).to(cfg.device)
    print(f"model params: {model.num_params() / 1e6:.3f}M  vocab: {tok.vocab_size}")
    init_l2 = sum(p.detach().float().pow(2).sum().item() for p in model.parameters())
    print(f"init param L2^2: {init_l2:.4f}  stop_grad_thoughts: {cfg.stop_grad_thoughts}")

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=cfg.device.startswith("cuda"),
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if cfg.device.startswith("cuda")
        else torch.amp.autocast(device_type="cpu", dtype=dtype)
    )

    best_acc = 0.0
    grokking_step = None
    grokking_threshold = 0.5
    eval_history: list[tuple[int, float]] = []

    t0 = time.time()
    for step in range(cfg.max_steps):
        k = current_k(step, cfg.thought_schedule)
        batcher = ArithBatcher(tok, cfg.n_terms, k, seed=cfg.seed + step)

        lr = get_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        x, y, m = batcher.sample(cfg.batch_size, device=cfg.device)
        with autocast_ctx:
            _, loss = model.forward_latent(x, m, y, stop_grad_thoughts=cfg.stop_grad_thoughts)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if step % 50 == 0:
            print(f"step {step:5d} | k={k} | loss {loss.item():.4f} | lr {lr:.2e} | {time.time() - t0:.1f}s")

        if step > 0 and step % cfg.eval_every == 0:
            acc = eval_accuracy(model, tok, cfg, cfg.eval_n_thoughts, cfg.eval_iters)
            print(f"  >> eval@K={cfg.eval_n_thoughts}: acc {acc:.3f}")
            eval_history.append((step, acc))
            if grokking_step is None and acc >= grokking_threshold:
                grokking_step = step
                print(f"  >> grokking crossed {grokking_threshold:.0%} at step {step}")
            if cfg.early_stop_patience > 0 and step >= cfg.early_stop_min_step and grokking_step is None:
                best_recent = max(a for _, a in eval_history[-cfg.early_stop_patience:])
                if best_recent < cfg.early_stop_min_acc:
                    print(f"  >> early stop: last {cfg.early_stop_patience} evals all below {cfg.early_stop_min_acc:.2f}")
                    break
            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_cfg": model_cfg.__dict__,
                        "step": step,
                        "acc": acc,
                        "n_terms": cfg.n_terms,
                        "eval_n_thoughts": cfg.eval_n_thoughts,
                    },
                    Path(cfg.out_dir) / "best.pt",
                )
                print(f"  >> saved best.pt @ acc {best_acc:.3f}")

    final_l2 = sum(p.detach().float().pow(2).sum().item() for p in model.parameters())
    print(f"final param L2^2: {final_l2:.4f}  delta: {final_l2 - init_l2:+.4f}")

    tail = eval_history[-10:]
    summary = {
        "config": args.config,
        "n_terms": cfg.n_terms,
        "eval_n_thoughts": cfg.eval_n_thoughts,
        "max_steps": cfg.max_steps,
        "grokking_step": grokking_step,
        "final_acc": tail[-1][1] if tail else None,
        "best_acc": best_acc,
        "acc_tail": tail,
    }
    with open(Path(cfg.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"config          {args.config}")
    print(f"n_terms         {cfg.n_terms}")
    print(f"max_steps       {cfg.max_steps}")
    print(f"grokking step   {grokking_step}" if grokking_step is not None else "grokking step   (never crossed 50%)")
    print(f"final acc       {tail[-1][1]:.3f}" if tail else "final acc       n/a")
    print(f"best acc        {best_acc:.3f}")
    print(f"acc tail        {[(s, round(a, 3)) for s, a in tail]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
