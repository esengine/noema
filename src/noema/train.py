from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from noema.data import TokenDataset
from noema.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"

    vocab_size: int = 50304
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

    batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 5000
    eval_every: int = 250
    eval_iters: int = 50
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    grad_checkpointing: bool = False

    out_dir: str = "runs/phase0"
    seed: int = 1337


def load_config(path: str) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainConfig(**raw)


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, datasets: dict[str, TokenDataset], cfg: TrainConfig) -> dict[str, float]:
    out = {}
    model.eval()
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = ds.sample_batch(cfg.batch_size, device=cfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": TokenDataset(cfg.train_bin, cfg.block_size),
        "val": TokenDataset(cfg.val_bin, cfg.block_size),
    }

    model_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(cfg.device)
    print(f"model params: {model.num_params() / 1e6:.2f}M")

    if cfg.compile:
        model = torch.compile(model)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=(cfg.device.startswith("cuda")),
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if cfg.device.startswith("cuda")
        else torch.amp.autocast(device_type="cpu", dtype=dtype)
    )

    best_val = float("inf")
    t0 = time.time()
    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            x, y = datasets["train"].sample_batch(cfg.batch_size, device=cfg.device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optim)
        scaler.update()

        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item() * cfg.grad_accum_steps:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if step > 0 and step % cfg.eval_every == 0:
            losses = estimate_loss(model, datasets, cfg)
            print(f"  >> val: train {losses['train']:.4f}  val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "step": step,
                    "val_loss": best_val,
                }
                torch.save(ckpt, Path(cfg.out_dir) / "best.pt")
                print(f"  >> saved best.pt @ val {best_val:.4f}")

    print(f"done. best val {best_val:.4f}")


if __name__ == "__main__":
    main()
