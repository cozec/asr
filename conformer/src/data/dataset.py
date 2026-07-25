"""LibriSpeech dataset + collate functions for CTC and RNN-T training."""

import json
import os
from dataclasses import dataclass

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .features import compute_fbank, spec_augment


@dataclass
class Utterance:
    utt_id: str
    path: str
    duration: float
    text: str


def scan_librispeech(root: str, subsets: list[str]) -> list[Utterance]:
    """Walk LibriSpeech `subsets` under `root`, pairing .flac files with transcripts.

    Each speaker/chapter dir holds one `*.trans.txt` whose lines are `<utt-id> TEXT`.
    """
    utterances = []
    for subset in subsets:
        subset_dir = os.path.join(root, subset)
        if not os.path.isdir(subset_dir):
            raise FileNotFoundError(f"missing LibriSpeech subset: {subset_dir}")
        for dirpath, _, filenames in os.walk(subset_dir):
            trans = [f for f in filenames if f.endswith(".trans.txt")]
            if not trans:
                continue
            with open(os.path.join(dirpath, trans[0])) as fh:
                for line in fh:
                    utt_id, _, text = line.strip().partition(" ")
                    path = os.path.join(dirpath, f"{utt_id}.flac")
                    if not os.path.exists(path):
                        continue
                    info = sf.info(path)
                    utterances.append(
                        Utterance(utt_id, path, info.frames / info.samplerate, text)
                    )
    utterances.sort(key=lambda u: u.utt_id)
    return utterances


def write_manifest(utterances: list[Utterance], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for utt in utterances:
            fh.write(json.dumps(utt.__dict__) + "\n")


def read_manifest(path: str) -> list[Utterance]:
    with open(path) as fh:
        return [Utterance(**json.loads(line)) for line in fh]


def load_manifest_cached(root: str, subsets: list[str], cache_dir: str) -> list[Utterance]:
    """Scan `subsets`, caching one manifest per subset.

    Probing durations means an `sf.info` call per file; over the 960h training set
    that is ~280k stat+header reads, so the result is worth keeping on disk.
    """
    utterances = []
    for subset in subsets:
        cache_path = os.path.join(cache_dir, f"{subset}.jsonl")
        if os.path.exists(cache_path):
            utterances.extend(read_manifest(cache_path))
            continue
        print(f"  scanning {subset} (first run; caching to {cache_path})", flush=True)
        scanned = scan_librispeech(root, [subset])
        write_manifest(scanned, cache_path)
        utterances.extend(scanned)
    return utterances


class LibriSpeechDataset(Dataset):
    """Yields (features, token_ids) per utterance.

    Feature extraction runs here so it parallelises across DataLoader workers.
    """

    def __init__(self, utterances, tokenizer, feature_cfg: dict,
                 specaug_cfg: dict | None = None, max_duration: float | None = None):
        if max_duration is not None:
            utterances = [u for u in utterances if u.duration <= max_duration]
        self.utterances = utterances
        self.tokenizer = tokenizer
        self.feature_cfg = feature_cfg
        self.specaug_cfg = specaug_cfg

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int):
        utt = self.utterances[index]
        waveform, sample_rate = sf.read(utt.path, dtype="float32")
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        features = compute_fbank(
            waveform,
            sample_rate=sample_rate,
            num_mel_bins=self.feature_cfg["num_mel_bins"],
            frame_length=self.feature_cfg["frame_length"],
            frame_shift=self.feature_cfg["frame_shift"],
        )
        # Per-utterance mean/variance normalisation.
        features = (features - features.mean(0)) / (features.std(0) + 1e-5)

        if self.specaug_cfg and self.specaug_cfg.get("enabled"):
            features = spec_augment(
                features,
                num_freq_masks=self.specaug_cfg["num_freq_masks"],
                freq_mask_param=self.specaug_cfg["freq_mask_param"],
                num_time_masks=self.specaug_cfg["num_time_masks"],
                time_mask_ratio=self.specaug_cfg["time_mask_ratio"],
            )

        tokens = torch.tensor(self.tokenizer.encode(utt.text), dtype=torch.long)
        return features, tokens, utt.utt_id


def ctc_collate(batch):
    """-> features (B, T, F), feature_lengths, targets (B, U), target_lengths, ids."""
    features, tokens, utt_ids = zip(*batch)
    feature_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
    return (
        pad_sequence(features, batch_first=True),
        feature_lengths,
        pad_sequence(tokens, batch_first=True, padding_value=0),
        target_lengths,
        list(utt_ids),
    )


def rnnt_collate(blank_id: int = 0):
    """Collate for the transducer: targets get a leading blank as the SOS symbol."""

    def collate(batch):
        features, tokens, utt_ids = zip(*batch)
        feature_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
        target_lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        padded = pad_sequence(tokens, batch_first=True, padding_value=blank_id)
        sos = torch.full((padded.size(0), 1), blank_id, dtype=torch.long)
        return (
            pad_sequence(features, batch_first=True),
            feature_lengths,
            padded,                                  # (B, U)   for the loss
            torch.cat([sos, padded], dim=1),         # (B, U+1) for the decoder
            target_lengths,
            list(utt_ids),
        )

    return collate
