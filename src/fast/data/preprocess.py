"""
BCI Competition 2020 Track #3 Imagined Speech Dataset Preprocessing

Provides electrode/zone definitions and data loading functions for the official splits.
"""

import os
import numpy as np
import scipy.io
import h5py
import pandas as pd
import multiprocessing as mp


# ==========================================
# DATASET CONSTANTS
# ==========================================
NAME = 'BCIC2020Track3'
SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
CLASSES = ['hello', 'help-me', 'stop', 'thank-you', 'yes']
TARGET_TIMEPOINTS = 800  # Pad to 800 for model input

# 64-channel electrode montage
Electrodes = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 
    'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
    'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
    'PO3', 'POz', 'PO4', 'PO8'
]

# Functional brain area zones
Zones = {
    'Pre-frontal': ['AF7', 'Fp1', 'Fp2', 'AF8', 'AF3', 'AF4'],
    'Frontal': ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    'Pre-central': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
    'Central': ['C1', 'C2', 'C3', 'Cz', 'C4', 'C5', 'C6'],
    'Post-central': ['CP1', 'CP2', 'CP3', 'CPz', 'CP4', 'CP5', 'CP6'],
    'Temporal': ['T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8', 'TP9', 'TP10', 'FT9', 'FT10'],
    'Parietal': ['P1', 'P2', 'P3', 'P4', 'Pz', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'PO9', 'PO10'],
    'Occipital': ['O1', 'O2', 'Oz', 'POz'],
}


# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================

def load_training_set(base_folder):
    """Load all subjects from the Training set folder."""
    X_all, Y_all = [], []
    folder = os.path.join(base_folder, 'Training set')
    
    for SID in SUBJECTS:
        filepath = os.path.join(folder, f'Data_Sample{SID}.mat')
        if os.path.exists(filepath):
            data = scipy.io.loadmat(filepath)
            x = np.asarray(data['epo_train']['x'])[0][0]
            y = np.asarray(data['epo_train']['y'])[0][0].argmax(0)
            x = np.transpose(x, (2, 1, 0)).astype(np.float32)
            # Pad to 800 timepoints
            x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
            X_all.append(x)
            Y_all.append(y.astype(np.uint8))
            print(f"  Train S{SID}: {x.shape}, labels: {np.unique(y, return_counts=True)}")
    
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all


def load_validation_set(base_folder):
    """Load all subjects from the Validation set folder."""
    X_all, Y_all = [], []
    folder = os.path.join(base_folder, 'Validation set')
    
    for SID in SUBJECTS:
        filepath = os.path.join(folder, f'Data_Sample{SID}.mat')
        if os.path.exists(filepath):
            data = scipy.io.loadmat(filepath)
            x = np.asarray(data['epo_validation']['x'])[0][0]
            y = np.asarray(data['epo_validation']['y'])[0][0].argmax(0)
            x = np.transpose(x, (2, 1, 0)).astype(np.float32)
            # Pad to 800 timepoints
            x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
            X_all.append(x)
            Y_all.append(y.astype(np.uint8))
            print(f"  Valid S{SID}: {x.shape}, labels: {np.unique(y, return_counts=True)}")
    
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all


def load_test_set(base_folder, excel_path):
    """Load all subjects from the Test set folder.
    
    Labels come from the Excel answer sheet.
    """
    X_all, Y_all = [], []
    folder = os.path.join(base_folder, 'Test set')
    
    # Load labels from Excel
    df_labels = pd.read_excel(excel_path, header=None)
    
    for i, SID in enumerate(SUBJECTS):
        filepath = os.path.join(folder, f'Data_Sample{SID}.mat')
        if os.path.exists(filepath):
            # Load X using h5py (MATLAB v7.3 format)
            with h5py.File(filepath, 'r') as f:
                if 'epo_test' in f:
                    x = np.array(f['epo_test']['x'])
                    if x.ndim == 3 and x.shape[2] < TARGET_TIMEPOINTS:
                        pad_width = TARGET_TIMEPOINTS - x.shape[2]
                        x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), 'edge')
                    x = x.astype(np.float32)
                    
                    # Load labels from Excel (columns 2, 4, 6, ... for subjects 1-15)
                    col_idx = 2 * (i + 1)
                    raw_labels = pd.to_numeric(df_labels.iloc[3:53, col_idx], errors='coerce').values
                    y = (raw_labels - 1).astype(np.uint8)  # Convert 1-5 to 0-4
                    
                    X_all.append(x)
                    Y_all.append(y)
                    print(f"  Test  S{SID}: {x.shape}, labels: {np.unique(y, return_counts=True)}")
    
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all


def load_test_set_per_subject(base_folder, excel_path):
    """Load test set data separately per subject for per-subject evaluation.
    
    Returns:
        dict: {subject_id: (X, Y)}
    """
    test_data = {}
    folder = os.path.join(base_folder, 'Test set')
    
    # Load labels from Excel
    df_labels = pd.read_excel(excel_path, header=None)
    
    for i, SID in enumerate(SUBJECTS):
        filepath = os.path.join(folder, f'Data_Sample{SID}.mat')
        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                if 'epo_test' in f:
                    x = np.array(f['epo_test']['x'])
                    if x.ndim == 3 and x.shape[2] < TARGET_TIMEPOINTS:
                        pad_width = TARGET_TIMEPOINTS - x.shape[2]
                        x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), 'edge')
                    x = x.astype(np.float32)
                    
                    col_idx = 2 * (i + 1)
                    raw_labels = pd.to_numeric(df_labels.iloc[3:53, col_idx], errors='coerce').values
                    y = (raw_labels - 1).astype(np.uint8)
                    
                    test_data[SID] = (x, y)
    
    return test_data


def load_subject_train_val(base_folder, SID):
    """Load combined train+validation data for one subject."""
    train_path = os.path.join(base_folder, 'Training set', f'Data_Sample{SID}.mat')
    valid_path = os.path.join(base_folder, 'Validation set', f'Data_Sample{SID}.mat')
    X_parts, Y_parts = [], []

    if os.path.exists(train_path):
        data = scipy.io.loadmat(train_path)
        x = np.asarray(data['epo_train']['x'])[0][0]
        y = np.asarray(data['epo_train']['y'])[0][0].argmax(0)
        x = np.transpose(x, (2, 1, 0)).astype(np.float32)
        x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
        X_parts.append(x)
        Y_parts.append(y.astype(np.uint8))

    if os.path.exists(valid_path):
        data = scipy.io.loadmat(valid_path)
        x = np.asarray(data['epo_validation']['x'])[0][0]
        y = np.asarray(data['epo_validation']['y'])[0][0].argmax(0)
        x = np.transpose(x, (2, 1, 0)).astype(np.float32)
        x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
        X_parts.append(x)
        Y_parts.append(y.astype(np.uint8))

    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    return X_all, Y_all


# ==========================================
# PREPROCESSING (for creating H5 cache)
# ==========================================

def proc_one(SID, src_folder='./', name='BCIC2020Track3'):
    """Process one subject's data (train + validation)."""
    data_train = scipy.io.loadmat(f'{src_folder}/{name}/Training set/Data_Sample{SID}.mat')
    data_valid = scipy.io.loadmat(f'{src_folder}/{name}/Validation set/Data_Sample{SID}.mat')
    x_t = np.asarray(data_train['epo_train']['x'])[0][0]
    y_t = np.asarray(data_train['epo_train']['y'])[0][0].argmax(0)
    x_v = np.asarray(data_valid['epo_validation']['x'])[0][0]
    y_v = np.asarray(data_valid['epo_validation']['y'])[0][0].argmax(0)
    x_t = np.transpose(x_t, (2, 1, 0)).astype(np.float32)
    x_v = np.transpose(x_v, (2, 1, 0)).astype(np.float32)
    x = np.concatenate((x_t, x_v), axis=0)
    y = np.concatenate((y_t, y_v), axis=0).astype(np.uint8)
    x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
    print(SID, x.shape, y.shape)
    return SID, x, y


def proc_all(src_folder='./', data_folder='./Processed'):
    """Process all subjects and save to HDF5."""
    os.makedirs(data_folder, exist_ok=True)
    with mp.Pool(len(SUBJECTS)) as pool:
        from functools import partial
        fn = partial(proc_one, src_folder=src_folder, name=NAME)
        with h5py.File(f'{data_folder}/{NAME}.h5', 'w') as f:
            for SID, X, Y in pool.map(fn, SUBJECTS):
                f.create_dataset(f'{SID}/X', data=X)
                f.create_dataset(f'{SID}/Y', data=Y)
                print(SID, X.shape, Y.shape, np.unique(Y, return_counts=True))


if __name__ == '__main__':
    proc_all()
