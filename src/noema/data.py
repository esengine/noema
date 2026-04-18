from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: str | Path, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run a prepare_*.py script in scripts/ first."
            )
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def sample_batch(self, batch_size: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, len(self), size=batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if device.startswith("cuda"):
            x_t = x_t.pin_memory().to(device, non_blocking=True)
            y_t = y_t.pin_memory().to(device, non_blocking=True)
        else:
            x_t = x_t.to(device)
            y_t = y_t.to(device)
        return x_t, y_t
