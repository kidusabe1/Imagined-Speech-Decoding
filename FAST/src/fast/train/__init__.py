"""
Training utilities for FAST
"""

from .trainer import EEG_Encoder_Module, cosine_scheduler, inference_on_loader
from .callbacks import HistoryCallback

__all__ = [
    "EEG_Encoder_Module",
    "cosine_scheduler",
    "inference_on_loader",
    "HistoryCallback",
]
