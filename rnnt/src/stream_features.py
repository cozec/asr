"""Streaming-safe log-mel frontend (paper §6.2).

80-dim log-mel at 25 ms / 10 ms, stacked `frame_stack` frames and downsampled by
`frame_stride` to a 30 ms frame rate.

The whole point of this module is that features computed chunk-by-chunk must equal
features computed over the whole utterance. Two pieces of carry-over make that true:

1. a waveform tail, because a 25 ms window overlaps the previous chunk's samples; and
2. a frame tail, because stacking/striding spans chunk boundaries.

`tests/test_rnnt.py::test_streaming_features_match_offline` asserts the equality.
"""

import torch

from data.features import compute_fbank


def stack_frames(features: torch.Tensor, stack: int, stride: int) -> torch.Tensor:
    """(T, D) -> (T', D * stack), taking every `stride`-th stacked window (paper §6.2)."""
    if stack == 1 and stride == 1:
        return features
    num = (features.size(0) - stack) // stride + 1
    if num <= 0:
        return features.new_zeros(0, features.size(1) * stack)
    windows = [features[i * stride: i * stride + stack].reshape(-1) for i in range(num)]
    return torch.stack(windows)


class StreamingFeatureExtractor:
    """Stateful log-mel + frame stacking for chunked audio.

    Feed successive waveform chunks to `__call__`; it returns whatever complete stacked
    frames became available. `reset()` clears state between utterances.
    """

    def __init__(self, sample_rate: int = 16000, num_mel_bins: int = 80,
                 frame_length: float = 25.0, frame_shift: float = 10.0,
                 frame_stack: int = 3, frame_stride: int = 3):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride

        self.win_samples = int(sample_rate * frame_length / 1000)
        self.hop_samples = int(sample_rate * frame_shift / 1000)
        self.reset()

    def reset(self) -> None:
        self._wave_tail = torch.zeros(0)
        self._frame_tail = torch.zeros(0, self.num_mel_bins)

    def _fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() < self.win_samples:
            return torch.zeros(0, self.num_mel_bins)
        return compute_fbank(waveform.unsqueeze(0), sample_rate=self.sample_rate,
                             num_mel_bins=self.num_mel_bins,
                             frame_length=self.frame_length,
                             frame_shift=self.frame_shift)

    def __call__(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: 1-D waveform. Returns (N, num_mel_bins * frame_stack), possibly N=0."""
        waveform = torch.cat([self._wave_tail, chunk])
        frames = self._fbank(waveform)

        # Keep the samples that a further frame would still need: everything after the
        # last frame we were able to start.
        consumed = frames.size(0) * self.hop_samples
        self._wave_tail = waveform[consumed:] if consumed else waveform

        frames = torch.cat([self._frame_tail, frames]) if self._frame_tail.numel() else frames
        stacked = stack_frames(frames, self.frame_stack, self.frame_stride)
        # Retain the frames that the next stacked window will still need.
        used = stacked.size(0) * self.frame_stride
        self._frame_tail = frames[used:]
        return stacked


def offline_features(waveform: torch.Tensor, sample_rate: int = 16000,
                     num_mel_bins: int = 80, frame_length: float = 25.0,
                     frame_shift: float = 10.0, frame_stack: int = 3,
                     frame_stride: int = 3) -> torch.Tensor:
    """Whole-utterance equivalent of the streaming extractor, for training and tests."""
    frames = compute_fbank(waveform.unsqueeze(0) if waveform.dim() == 1 else waveform,
                           sample_rate=sample_rate, num_mel_bins=num_mel_bins,
                           frame_length=frame_length, frame_shift=frame_shift)
    return stack_frames(frames, frame_stack, frame_stride)
