"""CTC decoding for wav2vec2 emissions.

`GreedyCTCDecoder` is the tutorial's decoder, kept verbatim in behaviour: take the
argmax label at each frame, collapse consecutive repeats, then drop blanks.
"""

import torch


class GreedyCTCDecoder(torch.nn.Module):
    """Best-path CTC decoding, with no language model (tutorial step 6)."""

    def __init__(self, labels, blank: int = 0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """emission: (num_frames, num_labels) logits -> transcript string.

        Word boundaries come back as the '|' label; `to_words` converts them to spaces.
        """
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


def to_words(transcript: str) -> str:
    """'I|HAD|THAT' -> 'I HAD THAT'. wav2vec2's vocabulary uses '|' for space."""
    return " ".join(t for t in transcript.split("|") if t)
