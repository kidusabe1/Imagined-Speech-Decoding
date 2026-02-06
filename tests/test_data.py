import numpy as np
import torch
from fast.data import BasicDataset
from fast.data.preprocess import Electrodes, Zones, SUBJECTS, CLASSES


def test_basic_dataset_wraps_arrays():
    """BasicDataset correctly wraps numpy arrays."""
    n_samples, n_channels, seq_len = 10, 64, 800
    X = np.random.randn(n_samples, n_channels, seq_len).astype(np.float32)
    Y = np.random.randint(0, 5, size=n_samples).astype(np.uint8)

    dataset = BasicDataset(X, Y)
    assert len(dataset) == n_samples

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (n_channels, seq_len)


def test_electrode_zone_coverage():
    """All zone electrodes exist in the electrode list."""
    for zone_name, channels in Zones.items():
        for ch in channels:
            assert ch in Electrodes, f"{ch} from zone '{zone_name}' not in Electrodes"


def test_dataset_constants():
    """Dataset constants have expected values."""
    assert len(SUBJECTS) == 15
    assert len(CLASSES) == 5
    assert len(Electrodes) == 64
    assert "hello" in CLASSES
    assert "yes" in CLASSES
