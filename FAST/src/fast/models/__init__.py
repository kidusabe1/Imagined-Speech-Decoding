"""
FAST Model Architecture

Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
"""

from .fast import (
    FAST,
    AttentionBlock,
    CVBlock,
    Conv4Layers,
    EEGNet_Encoder,
    HeadConv_Paper_Version,
    Head,
)

__all__ = [
    "FAST",
    "AttentionBlock",
    "CVBlock",
    "Conv4Layers",
    "EEGNet_Encoder",
    "HeadConv_Paper_Version",
    "Head",
]
