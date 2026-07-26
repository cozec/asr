"""Tests for the wav2vec2 pipeline reproduction.

    python tests/test_wav2vec2.py

The decoder tests are self-contained; the pipeline test needs the cached model weights
(run src/pipeline_demo.py once first) and is skipped if they are absent.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from decoder import GreedyCTCDecoder, to_words

PASSED, FAILED, SKIPPED = [], [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def skip(name, why):
    SKIPPED.append(name)
    print(f"  SKIP  {name}  ({why})")


LABELS = ("-", "|", "E", "T", "A", "O", "N", "I", "H", "S")


def test_greedy_collapses_repeats_and_blanks():
    """CTC decoding: collapse consecutive duplicates, then drop blanks."""
    decoder = GreedyCTCDecoder(labels=LABELS, blank=0)
    # H H <blank> E E | T <blank> <blank> A  ->  "HE|TA"
    # (indices into LABELS: 8='H', 0=blank, 2='E', 1='|', 3='T', 4='A')
    ids = [8, 8, 0, 2, 2, 1, 3, 0, 0, 4]
    emission = torch.full((len(ids), len(LABELS)), -10.0)
    for i, t in enumerate(ids):
        emission[i, t] = 0.0
    check("collapses repeats and strips blanks", decoder(emission) == "HE|TA", decoder(emission))


def test_blank_between_repeats_is_preserved():
    """A blank between two identical labels must keep both (the classic CTC case)."""
    decoder = GreedyCTCDecoder(labels=LABELS, blank=0)
    for ids, expect in ([[2, 2], "E"], [[2, 0, 2], "EE"]):
        emission = torch.full((len(ids), len(LABELS)), -10.0)
        for i, t in enumerate(ids):
            emission[i, t] = 0.0
        got = decoder(emission)
        check(f"{ids} -> {expect!r}", got == expect, f"got {got!r}")


def test_word_separator_conversion():
    check("'|' becomes a space", to_words("I|HAD|THAT") == "I HAD THAT")
    check("trailing separators are dropped", to_words("HELLO|") == "HELLO")
    check("repeated separators collapse", to_words("A||B") == "A B")


def test_pipeline_end_to_end():
    """Full pipeline on the tutorial's own audio; asserts the documented transcript."""
    import torchaudio

    from pipeline_demo import TUTORIAL_ASSET, load_audio

    audio = os.path.join(ROOT, "data", TUTORIAL_ASSET)
    if not os.path.exists(audio):
        skip("pipeline reproduces the tutorial transcript", "run src/pipeline_demo.py first")
        return

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    try:
        model = bundle.get_model().eval()
    except Exception as exc:                                # no cached weights, no network
        skip("pipeline reproduces the tutorial transcript", f"{type(exc).__name__}")
        return

    waveform = load_audio(audio, bundle.sample_rate)
    with torch.inference_mode():
        emission, _ = model(waveform)
    transcript = GreedyCTCDecoder(labels=bundle.get_labels())(emission[0])

    expected = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"
    check("pipeline reproduces the tutorial transcript", transcript == expected,
          f"got {transcript!r}")
    check("emission is (batch, frames, labels)",
          emission.dim() == 3 and emission.size(2) == len(bundle.get_labels()),
          str(tuple(emission.shape)))


def test_extract_features_layer_count():
    """extract_features returns one tensor per transformer layer (12 for BASE)."""
    import torchaudio

    try:
        model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().eval()
    except Exception as exc:
        skip("extract_features returns one tensor per layer", f"{type(exc).__name__}")
        return

    with torch.inference_mode():
        features, _ = model.extract_features(torch.randn(1, 16000))
    check("extract_features returns one tensor per layer", len(features) == 12,
          f"{len(features)} layers")
    check("each layer output is (batch, frames, 768)", features[0].size(2) == 768,
          str(tuple(features[0].shape)))


if __name__ == "__main__":
    print("wav2vec2 pipeline tests\n")
    for fn in [test_greedy_collapses_repeats_and_blanks,
               test_blank_between_repeats_is_preserved,
               test_word_separator_conversion,
               test_extract_features_layer_count,
               test_pipeline_end_to_end]:
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    sys.exit(1 if FAILED else 0)
