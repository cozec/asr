"""SentencePiece word-piece tokenizer (paper §3.2 uses a 1k WPM vocabulary).

Token id 0 is reserved as the blank/pad symbol for both CTC and the transducer, so
SentencePiece pieces are shifted up by one.
"""

import os

import sentencepiece as spm

BLANK_ID = 0


class Tokenizer:
    """Wraps a SentencePiece model, reserving id 0 for blank."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.blank_id = BLANK_ID

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() + 1

    def encode(self, text: str) -> list[int]:
        return [piece + 1 for piece in self.sp.encode(text.upper(), out_type=int)]

    def decode(self, ids: list[int]) -> str:
        pieces = [i - 1 for i in ids if i != self.blank_id]
        return self.sp.decode(pieces)


def train_sentencepiece(texts: list[str], model_prefix: str, vocab_size: int = 1024,
                        model_type: str = "unigram") -> str:
    """Train a SentencePiece model on transcripts; returns the .model path."""
    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
    text_file = f"{model_prefix}.txt"
    with open(text_file, "w") as fh:
        for line in texts:
            fh.write(line.upper() + "\n")

    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
    )
    return f"{model_prefix}.model"


class CharTokenizer:
    """Character vocabulary -- a dependency-free fallback for quick smoke runs."""

    def __init__(self, vocab: str = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"):
        self.itos = ["<blank>"] + list(vocab)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.blank_id = BLANK_ID

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text.upper() if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != self.blank_id)
