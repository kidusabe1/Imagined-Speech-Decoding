"""
Dataset classes and H5 loaders for EEG data
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    """Basic EEG Dataset that handles both 3D and 4D input arrays."""
    
    def __init__(self, data, label):
        if len(data.shape) == 4:
            data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_standardized_h5(cache_fn):
    """Load preprocessed EEG data from HDF5 file.
    
    Args:
        cache_fn: Path to the HDF5 file
        
    Returns:
        X: numpy array of shape (n_subjects, n_trials, n_channels, n_timepoints)
        Y: numpy array of shape (n_subjects, n_trials)
    """
    X, Y = [], []
    with h5py.File(cache_fn, 'r') as f:
        subjects = list(f.keys())
        for subject in subjects:
            X.append(f[subject]['X'][()])
            Y.append(f[subject]['Y'][()])
    X, Y = np.array(X), np.array(Y)
    print('Loaded from', cache_fn, X.shape, Y.shape)
    return X, Y
