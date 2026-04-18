from __future__ import annotations

from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def tokenize_split(split_name: str, out_path: Path) -> None:
    ds = load_dataset("roneneldan/TinyStories", split=split_name)
    print(f"{split_name}: {len(ds)} stories")

    total = 0
    for ex in tqdm(ds, desc=f"counting {split_name}"):
        total += len(enc.encode_ordinary(ex["text"])) + 1

    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total,))
    idx = 0
    for ex in tqdm(ds, desc=f"writing {split_name}"):
        ids = enc.encode_ordinary(ex["text"])
        ids.append(EOT)
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
    arr.flush()
    print(f"  -> {out_path}  ({idx:,} tokens)")


if __name__ == "__main__":
    tokenize_split("train", OUT / "train.bin")
    tokenize_split("validation", OUT / "val.bin")
