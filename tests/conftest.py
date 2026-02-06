import os
import tempfile

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from fast.data.preprocess import Electrodes, Zones


@pytest.fixture
def model_config():
    """Standard model config matching the BCIC2020Track3 dataset."""
    sfreq = 250
    return PretrainedConfig(
        electrodes=Electrodes,
        zone_dict=Zones,
        dim_cnn=32,
        dim_token=32,
        seq_len=800,
        window_len=sfreq,
        slide_step=sfreq // 2,
        head="Conv4Layers",
        n_classes=5,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    )


@pytest.fixture
def small_config():
    """Minimal config for fast tests (fewer layers/heads, small zones)."""
    electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    zones = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4'],
        'Central': ['C3', 'C4'],
        'Occipital': ['O1', 'O2'],
    }
    return PretrainedConfig(
        electrodes=electrodes,
        zone_dict=zones,
        dim_cnn=16,
        dim_token=16,
        seq_len=500,
        window_len=250,
        slide_step=125,
        head="Conv4Layers",
        n_classes=3,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def dummy_eeg_batch():
    """Batch of random EEG data matching dataset dimensions (64ch, 800t)."""
    batch_size = 4
    n_channels = len(Electrodes)  # 64
    seq_len = 800
    return torch.randn(batch_size, n_channels, seq_len)


@pytest.fixture
def dummy_eeg_small(small_config):
    """Batch of random EEG data matching small_config dimensions."""
    return torch.randn(2, 8, small_config.seq_len)


@pytest.fixture
def tmp_dir():
    """Temporary directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as d:
        yield d
