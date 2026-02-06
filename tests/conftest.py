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
def dummy_eeg_batch():
    """Batch of random EEG data matching dataset dimensions."""
    batch_size = 4
    n_channels = len(Electrodes)  # 64
    seq_len = 800
    return torch.randn(batch_size, n_channels, seq_len)
