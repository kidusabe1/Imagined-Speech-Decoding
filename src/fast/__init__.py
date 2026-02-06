"""
FAST: Functional Areas Spatio-Temporal Transformer for EEG Classification
"""

__version__ = "1.0.0"

from .models.fast import FAST
from .utils import seed_all, green, yellow, Tick, Tock

__all__ = ["FAST", "seed_all", "green", "yellow", "Tick", "Tock"]
