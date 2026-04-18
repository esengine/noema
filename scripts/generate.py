from __future__ import annotations

import argparse

import tiktoken
import torch

from noema.model import GPT, GPTConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/phase0_tiny/best.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode(args.prompt)], device=args.device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
