from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from noema.tokenizer import ArithTokenizer


@dataclass
class ArithProblem:
    operands: list[int]
    answer: int

    def prompt(self) -> str:
        return "+".join(str(o) for o in self.operands) + "="

    def answer_str(self) -> str:
        return str(self.answer)


def make_problem(n_terms: int, rng: random.Random) -> ArithProblem:
    operands = [rng.randint(0, 9) for _ in range(n_terms)]
    return ArithProblem(operands=operands, answer=sum(operands) % 10)


class ArithBatcher:
    """Builds training batches with a shared thought layout per batch."""

    def __init__(self, tokenizer: ArithTokenizer, n_terms: int, n_thoughts: int, seed: int = 0):
        self.tok = tokenizer
        self.n_terms = n_terms
        self.n_thoughts = n_thoughts
        self.rng = random.Random(seed)

        sample_prompt = "+".join(["9"] * n_terms) + "="
        self.prompt_len = len(sample_prompt)
        self.seq_len = self.prompt_len + 1 + n_thoughts + 1 + 1

    def sample(self, batch_size: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.full((batch_size, self.seq_len), self.tok.pad_id, dtype=torch.long)
        thought_mask = torch.zeros((batch_size, self.seq_len), dtype=torch.bool)

        for b in range(batch_size):
            p = make_problem(self.n_terms, self.rng)
            prompt = self.tok.encode(p.prompt())
            assert len(prompt) == self.prompt_len
            ans = self.tok.encode(p.answer_str())[0]

            t = 0
            input_ids[b, t : t + self.prompt_len] = torch.tensor(prompt)
            t += self.prompt_len
            input_ids[b, t] = self.tok.bot_id
            t += 1
            bot_end = t
            t += self.n_thoughts
            input_ids[b, t] = self.tok.eot_id
            t += 1
            input_ids[b, t] = ans

            thought_mask[b, bot_end : bot_end + self.n_thoughts] = True

        # Standard next-token LM targets, shifted by one.
        targets = torch.full_like(input_ids, -100)
        targets[:, :-1] = input_ids[:, 1:]

        # Thought positions emit hidden states, not predictions — no loss there.
        targets[thought_mask] = -100

        # Positions whose next token is a thought placeholder also carry no signal.
        target_is_thought = torch.zeros_like(thought_mask)
        target_is_thought[:, :-1] = thought_mask[:, 1:]
        targets[target_is_thought] = -100

        if device.startswith("cuda"):
            input_ids = input_ids.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
            thought_mask = thought_mask.pin_memory().to(device, non_blocking=True)
        else:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            thought_mask = thought_mask.to(device)
        return input_ids, targets, thought_mask

    def answer_position(self) -> int:
        return self.prompt_len + self.n_thoughts + 2
