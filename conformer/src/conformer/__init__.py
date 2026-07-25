"""Conformer: Convolution-augmented Transformer for Speech Recognition.

PyTorch implementation of Gulati et al., INTERSPEECH 2020 (arXiv:2005.08100).
"""

from .encoder import ConformerBlock, ConformerEncoder
from .model import ConformerCTC, ConformerTransducer, JointNetwork, TransducerDecoder
from .modules import (
    ConvolutionModule,
    Conv2dSubsampling,
    FeedForwardModule,
    MultiHeadedSelfAttentionModule,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    Swish,
)

__all__ = [
    "ConformerBlock",
    "ConformerEncoder",
    "ConformerCTC",
    "ConformerTransducer",
    "TransducerDecoder",
    "JointNetwork",
    "Conv2dSubsampling",
    "FeedForwardModule",
    "ConvolutionModule",
    "RelPositionalEncoding",
    "RelPositionMultiHeadAttention",
    "MultiHeadedSelfAttentionModule",
    "Swish",
]
