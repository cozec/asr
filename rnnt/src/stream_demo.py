"""Streaming ASR demo: microphone or file in, live partial transcript out.

    # pretrained Emformer RNN-T, on a LibriSpeech file (deterministic, no mic needed)
    python src/stream_demo.py --source file --audio ../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac

    # live microphone
    python src/stream_demo.py --source mic

    # our own model once trained
    python src/stream_demo.py --source mic --model ours --checkpoint exp/rnnt_small/epoch9.pt

Both backends expose the same interface, so the streaming loop, the timing harness and
the display are shared: the only thing that changes is which model produces tokens.
"""

import argparse
import os
import queue
import sys
import time

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ContextCacher:
    """Prepends the previous chunk's tail so the model sees its right context.

    The Emformer treats the newest frames of its input as a lookahead window, so each
    step must be handed `context_length` samples of overlap with the previous step.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros(context_length)

    def reset(self) -> None:
        self.context = torch.zeros(self.context_length)

    def __call__(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        with_context = torch.cat([self.context, chunk])
        self.context = chunk[-self.context_length:]
        return with_context


class PretrainedBackend:
    """torchaudio's Emformer RNN-T trained on LibriSpeech (960 h).

    Used so the demo produces real transcripts today. Downloads ~300 MB on first run.
    """

    name = "Emformer RNN-T (torchaudio, LibriSpeech 960h)"

    def __init__(self, beam_width: int = 10):
        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()
        self.beam_width = beam_width
        self.sample_rate = bundle.sample_rate

        self.segment_samples = bundle.segment_length * bundle.hop_length
        self.context_samples = bundle.right_context_length * bundle.hop_length
        self.cacher = ContextCacher(self.segment_samples, self.context_samples)
        self.reset()

    @property
    def chunk_samples(self) -> int:
        return self.segment_samples

    def reset(self) -> None:
        self.state, self.hypothesis = None, None
        self.cacher.reset()

    def process(self, chunk: torch.Tensor) -> str:
        features, length = self.feature_extractor(self.cacher(chunk))
        hypos, self.state = self.decoder.infer(
            features, length, self.beam_width, state=self.state, hypothesis=self.hypothesis
        )
        self.hypothesis = hypos
        return self.token_processor(hypos[0][0], lstrip=False)


class OursBackend:
    """Our RNN-T (He et al. §3.1 structure + stateless predictor), driven by the same loop."""

    name = "Ours: LSTM RNN-T + stateless predictor"

    def __init__(self, config: str, checkpoint: str, beam_width: int = 4):
        import yaml

        from data.tokenizer import Tokenizer
        from rnnt import build_rnnt
        from stream_features import StreamingFeatureExtractor
        from torchaudio.models import RNNTBeamSearch

        with open(config) as fh:
            cfg = yaml.safe_load(fh)
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg["data"]["tokenizer"])
        self.sample_rate = cfg["data"]["sample_rate"]

        self.model = build_rnnt(cfg, self.tokenizer.vocab_size)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.searcher = RNNTBeamSearch(self.model, blank=self.tokenizer.blank_id)
        self.beam_width = beam_width

        self.extractor = StreamingFeatureExtractor(
            sample_rate=self.sample_rate,
            num_mel_bins=cfg["data"]["num_mel_bins"],
            frame_length=cfg["data"]["frame_length"],
            frame_shift=cfg["data"]["frame_shift"],
            frame_stack=cfg["model"]["frame_stack"],
            frame_stride=cfg["model"]["frame_stride"],
        )
        # chunk_frames stacked frames, each covering frame_stride * frame_shift ms.
        ms = cfg["model"]["frame_stride"] * cfg["data"]["frame_shift"]
        self._chunk_samples = int(cfg["stream"]["chunk_frames"] * ms / 1000 * self.sample_rate)
        self.reset()

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    def reset(self) -> None:
        self.state, self.hypothesis = None, None
        self.extractor.reset()
        self.tokens = []

    @torch.no_grad()
    def process(self, chunk: torch.Tensor) -> str:
        features = self.extractor(chunk)
        if features.size(0) == 0:
            return self.tokenizer.decode(self.tokens)
        hypos, self.state = self.searcher.infer(
            features, torch.tensor([features.size(0)]), self.beam_width,
            state=self.state, hypothesis=self.hypothesis,
        )
        self.hypothesis = hypos
        self.tokens = [t for t in hypos[0][0] if t != self.tokenizer.blank_id]
        return self.tokenizer.decode(self.tokens)


def iter_file(path: str, chunk_samples: int, sample_rate: int, realtime: bool):
    """Yield fixed-size chunks from an audio file, resampling if needed."""
    import soundfile as sf

    # soundfile rather than torchaudio.load: torchaudio 2.11 routes load() through
    # torchcodec, which this project deliberately does not depend on.
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).mean(1)          # to mono
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    for i in range(0, waveform.size(0), chunk_samples):
        chunk = waveform[i:i + chunk_samples]
        if chunk.size(0) < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.size(0)))
        if realtime:
            time.sleep(chunk_samples / sample_rate)
        yield chunk
    return


def iter_mic(chunk_samples: int, sample_rate: int):
    """Yield chunks from the default input device."""
    import sounddevice as sd

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                        blocksize=chunk_samples, callback=callback):
        print("listening -- speak now, Ctrl-C to stop\n")
        while True:
            yield torch.from_numpy(q.get()[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mic", "file"], default="mic")
    parser.add_argument("--audio", help="audio file when --source file")
    parser.add_argument("--model", choices=["pretrained", "ours"], default="pretrained")
    parser.add_argument("--config", default="configs/rnnt_small.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--realtime", action="store_true",
                        help="with --source file, pace playback at 1x to mimic a mic")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after N seconds of audio and print the summary "
                             "(otherwise run until Ctrl-C)")
    args = parser.parse_args()

    if args.model == "ours":
        if not args.checkpoint:
            parser.error("--model ours requires --checkpoint")
        backend = OursBackend(args.config, args.checkpoint, args.beam_width or 4)
    else:
        backend = PretrainedBackend(args.beam_width or 10)

    print(f"model : {backend.name}")
    print(f"chunk : {backend.chunk_samples} samples "
          f"({backend.chunk_samples / backend.sample_rate * 1000:.0f} ms @ "
          f"{backend.sample_rate} Hz)")

    if args.source == "file":
        if not args.audio:
            parser.error("--source file requires --audio")
        print(f"audio : {args.audio}\n")
        chunks = iter_file(args.audio, backend.chunk_samples, backend.sample_rate,
                           args.realtime)
    else:
        chunks = iter_mic(backend.chunk_samples, backend.sample_rate)

    latencies, transcript, num_chunks = [], "", 0
    started = time.time()
    try:
        for chunk in chunks:
            t0 = time.perf_counter()
            transcript = backend.process(chunk)
            latencies.append((time.perf_counter() - t0) * 1000)
            num_chunks += 1
            # Rewrite one line so the partial transcript updates in place.
            print(f"\r\033[K> {transcript[-110:]}", end="", flush=True)
            if args.seconds is not None:
                elapsed = num_chunks * backend.chunk_samples / backend.sample_rate
                if elapsed >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass

    audio_seconds = num_chunks * backend.chunk_samples / backend.sample_rate
    compute_seconds = sum(latencies) / 1000
    print("\n")
    print(f"transcript: {transcript.strip()}")
    if latencies:
        ordered = sorted(latencies)
        p50 = ordered[len(ordered) // 2]
        p90 = ordered[int(len(ordered) * 0.9)]
        print(f"\naudio {audio_seconds:.1f}s | compute {compute_seconds:.1f}s | "
              f"RTF {compute_seconds / max(audio_seconds, 1e-9):.3f}")
        print(f"per-chunk latency: p50 {p50:.0f} ms, p90 {p90:.0f} ms "
              f"(chunk is {backend.chunk_samples / backend.sample_rate * 1000:.0f} ms of audio)")
        print("RTF < 1.0 means faster than real time (paper Table 4 reports RT90 0.51 "
              "quantized / 1.43 float on a Pixel).")


if __name__ == "__main__":
    main()
