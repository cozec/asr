"""RNN-T for streaming on-device ASR.

Implements the architecture of "Streaming End-to-end Speech Recognition For Mobile
Devices" (He et al., ICASSP 2019, arXiv:1811.06621), with the stateless prediction
network of Ghodsi et al. 2020 in place of the paper's recurrent one.
"""

from .model import Joiner, StatelessPredictor, build_rnnt
from .transcriber import LSTMTranscriber, TimeReduction

__all__ = [
    "LSTMTranscriber",
    "TimeReduction",
    "StatelessPredictor",
    "Joiner",
    "build_rnnt",
]
