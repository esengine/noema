from __future__ import annotations


class ArithTokenizer:
    SPECIALS = ("<pad>", "<bot>", "<eot>")
    CHARS = tuple("0123456789+-*=\n ")

    def __init__(self) -> None:
        vocab = list(self.SPECIALS) + list(self.CHARS)
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for i, s in enumerate(vocab)}
        self.pad_id = self.stoi["<pad>"]
        self.bot_id = self.stoi["<bot>"]
        self.eot_id = self.stoi["<eot>"]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.itos[i] for i in ids if self.itos[i] not in self.SPECIALS
        )
