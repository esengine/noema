from __future__ import annotations

import argparse

import torch

from noema.arith import ArithBatcher
from noema.latent import LatentGPT
from noema.model import GPTConfig
from noema.tokenizer import ArithTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-terms", type=int, default=None)
    p.add_argument("--n-thoughts", type=int, default=None)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    tok = ArithTokenizer()
    cfg = GPTConfig(**ckpt["model_cfg"])
    model = LatentGPT(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_terms = args.n_terms if args.n_terms is not None else ckpt["n_terms"]
    n_thoughts = args.n_thoughts if args.n_thoughts is not None else ckpt["eval_n_thoughts"]
    batcher = ArithBatcher(tok, n_terms=n_terms, n_thoughts=n_thoughts, seed=42)

    ans_pos = batcher.answer_position()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(args.batches):
            x, _, m = batcher.sample(args.batch_size, device=args.device)
            logits, _ = model.forward_latent(x, m)
            preds = logits[:, ans_pos - 1].argmax(dim=-1)
            true = x[:, ans_pos]
            correct += (preds == true).sum().item()
            total += preds.numel()

    acc = correct / max(total, 1)
    print(f"n_terms={n_terms}  n_thoughts={n_thoughts}  acc={acc:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    main()
