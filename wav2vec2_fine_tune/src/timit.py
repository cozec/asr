"""TIMIT loading for both tasks: orthographic ASR and phoneme recognition.

The HF blog calls `load_dataset("timit_asr")`, which needs a local LDC copy anyway --
TIMIT is licensed, so there is no download path. This reads the corpus directly off disk,
which also gives access to the `.PHN` phonetic annotations the blog never touches and
step 2 needs.

Conventions this follows, all standard for TIMIT:

* **SA1/SA2 are excluded.** Every one of the 630 speakers reads the same two "dialect"
  sentences, so keeping them lets the model memorise text it will see again at test time.
  Train drops 4620 -> 3696, full test 1680 -> 1344.
* **61 phones are folded to 39 for scoring** (Lee & Hon, 1989), the standard TIMIT
  phone-error-rate protocol.
* The **core test set** (24 speakers, 192 utterances) is available via `core_test`.
"""

import os
import re
from dataclasses import dataclass

TIMIT_ROOT_HINT = "data/TIMIT/lisa/TIMIT"

# The blog's preprocessing: strip punctuation, lowercase.
CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"]'

# Lee & Hon (1989) 61 -> 39 phone folding. Anything absent maps to itself.
PHONE_FOLD = {
    "ao": "aa", "ax": "ah", "ax-h": "ah", "axr": "er", "hv": "hh", "ix": "ih",
    "el": "l", "em": "m", "en": "n", "nx": "n", "eng": "ng", "zh": "sh",
    "ux": "uw",
    # every closure / silence variant collapses to one symbol
    "pcl": "sil", "tcl": "sil", "kcl": "sil", "bcl": "sil", "dcl": "sil",
    "gcl": "sil", "h#": "sil", "pau": "sil", "epi": "sil",
}
# The glottal stop is deleted outright rather than folded.
PHONE_DELETE = {"q"}

# The 24 core-test speakers defined by the TIMIT documentation.
CORE_TEST_SPEAKERS = {
    "MDAB0", "MWBT0", "FELC0", "MTAS1", "MWEW0", "FPAS0", "MJMP0", "MLNT0",
    "FPKT0", "MLLL0", "MTLS0", "FJLM0", "MBPM0", "MKLT0", "FNLP0", "MCMJ0",
    "MJDH0", "FMGD0", "MGRT0", "MNJM0", "FDHC0", "MJLN0", "MPAM0", "FMLD0",
}


@dataclass
class Utterance:
    utt_id: str
    path: str
    speaker: str
    text: str                  # orthographic, blog-preprocessed
    phones: list[str]          # folded to the 39-phone set


def fold_phones(phones: list[str]) -> list[str]:
    """Apply the 61 -> 39 folding, dropping glottal stops."""
    out = []
    for p in phones:
        if p in PHONE_DELETE:
            continue
        out.append(PHONE_FOLD.get(p, p))
    return out


def _read_text(path: str) -> str:
    """`.TXT` lines are '<start> <end> <the sentence>'."""
    with open(path) as fh:
        line = fh.readline().strip()
    _, _, text = line.split(" ", 2)
    return re.sub(CHARS_TO_IGNORE, "", text).lower().strip()


def _read_phones(path: str) -> list[str]:
    """`.PHN` lines are '<start> <end> <phone>'."""
    phones = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 3:
                phones.append(parts[2])
    return phones


def find_root(start: str = ".") -> str:
    """Locate the TIMIT directory holding TRAIN/ and TEST/."""
    candidate = os.path.join(start, TIMIT_ROOT_HINT)
    if os.path.isdir(os.path.join(candidate, "TRAIN")):
        return candidate
    for dirpath, dirnames, _ in os.walk(os.path.join(start, "data")):
        if "TRAIN" in dirnames and "TEST" in dirnames:
            return dirpath
    raise FileNotFoundError(
        "TIMIT not found. Expected data/TIMIT/... containing TRAIN/ and TEST/.")


def load_split(root: str, split: str, drop_sa: bool = True) -> list[Utterance]:
    """split: 'train' | 'test' | 'core_test'."""
    subdir = "TRAIN" if split == "train" else "TEST"
    utterances = []

    for dirpath, _, filenames in sorted(os.walk(os.path.join(root, subdir))):
        for name in sorted(f for f in filenames if f.endswith(".WAV")):
            stem = name[:-4]
            # Every speaker reads SA1 and SA2; keeping them leaks text across splits.
            if drop_sa and stem.startswith("SA"):
                continue
            speaker = os.path.basename(dirpath)
            if split == "core_test" and speaker not in CORE_TEST_SPEAKERS:
                continue
            text_path = os.path.join(dirpath, f"{stem}.TXT")
            phn_path = os.path.join(dirpath, f"{stem}.PHN")
            if not (os.path.exists(text_path) and os.path.exists(phn_path)):
                continue
            utterances.append(Utterance(
                utt_id=f"{speaker}-{stem}",
                path=os.path.join(dirpath, name),
                speaker=speaker,
                text=_read_text(text_path),
                phones=fold_phones(_read_phones(phn_path)),
            ))
    return utterances


def phone_inventory(splits: list[list[Utterance]]) -> list[str]:
    """Sorted set of folded phones appearing anywhere in the given splits."""
    phones = set()
    for split in splits:
        for utt in split:
            phones.update(utt.phones)
    return sorted(phones)


class PhoneCoder:
    """Maps phone symbols to single characters and back.

    `Wav2Vec2CTCTokenizer` tokenises by splitting text into *characters*, so a
    multi-character symbol like 'aa' can never be produced. Assigning each phone its own
    private-use codepoint lets the standard CTC tokenizer and decoder work unmodified,
    and the mapping is inverted for scoring.
    """

    def __init__(self, phones: list[str]):
        self.phones = list(phones)
        # Private Use Area: guaranteed not to collide with anything in the transcripts.
        self.to_char = {p: chr(0xE000 + i) for i, p in enumerate(self.phones)}
        self.from_char = {c: p for p, c in self.to_char.items()}

    def encode(self, phones: list[str]) -> str:
        return "".join(self.to_char[p] for p in phones if p in self.to_char)

    def decode(self, text: str) -> list[str]:
        return [self.from_char[c] for c in text if c in self.from_char]
