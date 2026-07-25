"""Filterbank features and SpecAugment (paper §3.1)."""

import random

import torch
import torchaudio.compliance.kaldi as kaldi


def compute_fbank(waveform: torch.Tensor, sample_rate: int = 16000,
                  num_mel_bins: int = 80, frame_length: float = 25.0,
                  frame_shift: float = 10.0) -> torch.Tensor:
    """80-channel log-mel filterbanks, 25ms window / 10ms stride (paper §3.1).

    waveform: (1, N) float in [-1, 1]. Returns (T, num_mel_bins).
    """
    # Kaldi's frontend expects int16-scaled samples.
    return kaldi.fbank(
        waveform * (1 << 15),
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate,
        dither=0.0,
        energy_floor=1.0,
    )


def spec_augment(features: torch.Tensor, num_freq_masks: int = 2,
                 freq_mask_param: int = 27, num_time_masks: int = 10,
                 time_mask_ratio: float = 0.05) -> torch.Tensor:
    """SpecAugment, in-place on a copy (paper §3.1).

    Uses the paper's settings: mask parameter F=27 and ten time masks whose maximum
    size is `time_mask_ratio` times the utterance length (the adaptive pS policy),
    rather than a fixed number of frames.

    features: (T, F). Returns a masked copy.
    """
    features = features.clone()
    num_frames, num_bins = features.size()
    max_time_mask = max(1, int(time_mask_ratio * num_frames))

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        if f == 0 or f >= num_bins:
            continue
        f0 = random.randint(0, num_bins - f)
        features[:, f0:f0 + f] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, max_time_mask)
        if t == 0 or t >= num_frames:
            continue
        t0 = random.randint(0, num_frames - t)
        features[t0:t0 + t, :] = 0.0

    return features
