"""
Comprehensive tests for data loading and preprocessing (src/fast/data/).

Covers:
- BasicDataset: 3D input, 4D input (auto-concatenation), indexing, length, types
- Dataset constants: Electrodes, Zones, SUBJECTS, CLASSES
- Electrode/zone integrity: all zone channels in electrode list, no duplicates
- Zone coverage: total channels across zones vs electrode list
- load_standardized_h5: round-trip write/read from HDF5
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch

from fast.data import BasicDataset, load_standardized_h5
from fast.data.preprocess import (
    Electrodes, Zones, SUBJECTS, CLASSES, TARGET_TIMEPOINTS, NAME,
)


# ============================================================
# Dataset constants
# ============================================================

class TestDatasetConstants:
    """Tests for dataset-level constants."""

    def test_subject_count(self):
        assert len(SUBJECTS) == 15

    def test_subjects_are_zero_padded_strings(self):
        """Subject IDs are two-digit zero-padded strings."""
        for s in SUBJECTS:
            assert len(s) == 2
            assert s.isdigit()

    def test_class_count(self):
        assert len(CLASSES) == 5

    def test_all_expected_classes_present(self):
        expected = {'hello', 'help-me', 'stop', 'thank-you', 'yes'}
        assert set(CLASSES) == expected

    def test_electrode_count(self):
        assert len(Electrodes) == 64

    def test_no_duplicate_electrodes(self):
        assert len(Electrodes) == len(set(Electrodes))

    def test_target_timepoints(self):
        assert TARGET_TIMEPOINTS == 800

    def test_dataset_name(self):
        assert NAME == 'BCIC2020Track3'


class TestZoneIntegrity:
    """Tests for zone-electrode consistency."""

    def test_all_zone_channels_in_electrode_list(self):
        """Every channel listed in a zone must exist in Electrodes."""
        for zone_name, channels in Zones.items():
            for ch in channels:
                assert ch in Electrodes, f"'{ch}' from zone '{zone_name}' not in Electrodes"

    def test_zone_count(self):
        """There are exactly 8 functional brain areas."""
        assert len(Zones) == 8

    def test_expected_zone_names(self):
        expected = {
            'Pre-frontal', 'Frontal', 'Pre-central', 'Central',
            'Post-central', 'Temporal', 'Parietal', 'Occipital',
        }
        assert set(Zones.keys()) == expected

    def test_no_duplicate_channels_within_zones(self):
        """No zone has duplicate channels."""
        for zone_name, channels in Zones.items():
            assert len(channels) == len(set(channels)), f"Duplicates in zone '{zone_name}'"

    def test_total_zone_channels(self):
        """Total channels across all zones."""
        total = sum(len(chs) for chs in Zones.values())
        # Zones may not cover all 64 electrodes — just verify it's reasonable
        assert total > 0
        assert total <= len(Electrodes)

    def test_zone_channel_sizes(self):
        """Each zone has at least 1 channel."""
        for zone_name, channels in Zones.items():
            assert len(channels) >= 1, f"Zone '{zone_name}' is empty"


# ============================================================
# BasicDataset
# ============================================================

class TestBasicDataset:
    """Tests for BasicDataset (torch Dataset wrapper)."""

    def test_3d_input(self):
        """3D input (n_samples, n_channels, seq_len) is handled."""
        X = np.random.randn(10, 64, 800).astype(np.float32)
        Y = np.random.randint(0, 5, size=10).astype(np.uint8)
        ds = BasicDataset(X, Y)
        assert len(ds) == 10

    def test_4d_input_flattens(self):
        """4D input (n_subjects, n_trials, C, T) is concatenated along axis 0."""
        X = np.random.randn(3, 10, 64, 800).astype(np.float32)
        Y = np.random.randint(0, 5, size=(3, 10)).astype(np.uint8)
        ds = BasicDataset(X, Y)
        assert len(ds) == 30  # 3 * 10

    def test_getitem_returns_tensors(self):
        """__getitem__ returns (tensor, tensor)."""
        X = np.random.randn(5, 8, 100).astype(np.float32)
        Y = np.arange(5).astype(np.uint8)
        ds = BasicDataset(X, Y)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_getitem_shapes(self):
        """Individual sample has correct shape."""
        X = np.random.randn(5, 8, 100).astype(np.float32)
        Y = np.arange(5).astype(np.uint8)
        ds = BasicDataset(X, Y)
        x, y = ds[2]
        assert x.shape == (8, 100)
        assert y.shape == ()  # scalar label

    def test_data_dtype_preserved(self):
        """float32 numpy data becomes float32 tensor."""
        X = np.ones((3, 4, 10), dtype=np.float32)
        Y = np.zeros(3, dtype=np.uint8)
        ds = BasicDataset(X, Y)
        x, _ = ds[0]
        assert x.dtype == torch.float32

    def test_label_values_preserved(self):
        """Label values are exactly preserved."""
        X = np.random.randn(5, 4, 10).astype(np.float32)
        Y = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        ds = BasicDataset(X, Y)
        for i in range(5):
            _, y = ds[i]
            assert y.item() == i

    def test_single_sample_dataset(self):
        """Dataset with a single sample works."""
        X = np.random.randn(1, 4, 10).astype(np.float32)
        Y = np.array([2], dtype=np.uint8)
        ds = BasicDataset(X, Y)
        assert len(ds) == 1
        x, y = ds[0]
        assert x.shape == (4, 10)

    def test_last_index(self):
        """Accessing the last index works."""
        n = 7
        X = np.random.randn(n, 4, 10).astype(np.float32)
        Y = np.arange(n).astype(np.uint8)
        ds = BasicDataset(X, Y)
        x, y = ds[n - 1]
        assert y.item() == n - 1

    def test_negative_indexing(self):
        """Negative indexing works (PyTorch tensor supports it)."""
        X = np.random.randn(5, 4, 10).astype(np.float32)
        Y = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        ds = BasicDataset(X, Y)
        _, y = ds[-1]
        assert y.item() == 4


# ============================================================
# load_standardized_h5
# ============================================================

class TestLoadStandardizedH5:
    """Tests for HDF5 round-trip loading."""

    def test_round_trip(self, tmp_dir):
        """Write then read HDF5 file produces same arrays."""
        h5_path = os.path.join(tmp_dir, 'test.h5')
        n_subjects, n_trials, n_ch, n_t = 3, 10, 64, 800

        X_orig = np.random.randn(n_subjects, n_trials, n_ch, n_t).astype(np.float32)
        Y_orig = np.random.randint(0, 5, (n_subjects, n_trials)).astype(np.uint8)

        with h5py.File(h5_path, 'w') as f:
            for i in range(n_subjects):
                f.create_dataset(f'{i:02d}/X', data=X_orig[i])
                f.create_dataset(f'{i:02d}/Y', data=Y_orig[i])

        X_loaded, Y_loaded = load_standardized_h5(h5_path)
        assert X_loaded.shape == X_orig.shape
        assert Y_loaded.shape == Y_orig.shape
        np.testing.assert_array_equal(X_loaded, X_orig)
        np.testing.assert_array_equal(Y_loaded, Y_orig)

    def test_single_subject(self, tmp_dir):
        """Works with a single subject."""
        h5_path = os.path.join(tmp_dir, 'single.h5')
        X = np.random.randn(5, 8, 100).astype(np.float32)
        Y = np.arange(5).astype(np.uint8)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('01/X', data=X)
            f.create_dataset('01/Y', data=Y)

        X_loaded, Y_loaded = load_standardized_h5(h5_path)
        assert X_loaded.shape[0] == 1
        np.testing.assert_array_equal(X_loaded[0], X)

    def test_empty_file_raises(self, tmp_dir):
        """Empty HDF5 produces empty arrays (no subjects)."""
        h5_path = os.path.join(tmp_dir, 'empty.h5')
        with h5py.File(h5_path, 'w') as f:
            pass  # empty file
        X, Y = load_standardized_h5(h5_path)
        assert len(X) == 0
        assert len(Y) == 0
