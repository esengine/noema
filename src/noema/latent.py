from __future__ import annotations

import torch
import torch.nn.functional as F

from noema.model import GPT


class LatentGPT(GPT):
    """GPT where selected positions use the previous step's final hidden state
    as their input embedding instead of a token embedding."""

    def _trunk(self, emb: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x = self.drop(emb + pos_emb.unsqueeze(0))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward_latent(
        self,
        input_ids: torch.Tensor,
        thought_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        assert T <= self.cfg.block_size
        assert thought_mask.shape == input_ids.shape
        assert thought_mask.dtype == torch.bool

        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)

        emb = self.tok_emb(input_ids)

        thought_idx = thought_mask[0].nonzero(as_tuple=True)[0]
        # Requires the same thought layout across the batch — enforced by the trainer.
        if (thought_mask != thought_mask[0]).any():
            raise ValueError("thought_mask must be identical across the batch")

        for t_pos in thought_idx.tolist():
            assert t_pos > 0, "thought cannot be the first position"
            h = self._trunk(emb, pos_emb)
            new_emb = emb.clone()
            new_emb[:, t_pos, :] = h[:, t_pos - 1, :]
            emb = new_emb

        h = self._trunk(emb, pos_emb)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

