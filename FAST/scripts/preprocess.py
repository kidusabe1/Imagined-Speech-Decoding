#!/usr/bin/env python
"""
Preprocessing script for BCI Competition 2020 Track #3 dataset.

Converts raw .mat files to processed HDF5 format.
"""

import os
import sys
import argparse
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast.data import (
    load_training_set, load_validation_set, load_test_set,
    Electrodes, Zones, SUBJECTS, CLASSES
)


def resolve_data_folder(data_folder: str) -> str:
    """Resolve data folder, falling back to sibling BCIC2020Track3 if needed."""
    candidates = [
        os.path.abspath(data_folder),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BCIC2020Track3')),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"BCIC2020Track3 folder not found. Tried: {candidates}")


def preprocess_dataset(data_folder, output_folder, excel_path=None):
    """Preprocess BCIC2020Track3 dataset and save to HDF5."""

    data_folder = resolve_data_folder(data_folder)
    os.makedirs(output_folder, exist_ok=True)

    print("="*60)
    print("PREPROCESSING BCIC2020Track3 DATASET")
    print("="*60)
    print(f"Source: {data_folder}")
    print(f"Output: {output_folder}")
    print()

    # --- Training Set ---
    print("Loading Training Set...")
    X_train_all, Y_train_all = [], []
    try:
        X_train, Y_train = load_training_set(data_folder)
        print(f"Total Training: {X_train.shape}")
    except Exception as e:
        X_train, Y_train = None, None
        print(f"Training load FAILED: {e}")

    # --- Validation Set ---
    print("\nLoading Validation Set...")
    try:
        X_valid, Y_valid = load_validation_set(data_folder)
        print(f"Total Validation: {X_valid.shape}")
    except Exception as e:
        X_valid, Y_valid = None, None
        print(f"Validation load FAILED: {e}")

    # --- Test Set ---
    print("\nLoading Test Set...")
    if excel_path is None:
        excel_path = os.path.join(data_folder, 'Test set', 'Track3_Answer Sheet_Test.xlsx')
    
    try:
        X_test, Y_test = load_test_set(data_folder, excel_path)
        print(f"Total Test: {X_test.shape}")
    except Exception as e:
        X_test, Y_test = None, None
        print(f"Test load FAILED: {e}")

    # If everything failed, abort before writing an empty file
    if X_train is None and X_valid is None and X_test is None:
        raise SystemExit("No data loaded. Check --data_folder and Excel path.")

    # --- Save to HDF5 ---
    output_path = os.path.join(output_folder, 'BCIC2020Track3.h5')
    print(f"\nSaving to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        if X_train is not None:
            f.create_dataset('X_train', data=X_train, compression='gzip')
            f.create_dataset('Y_train', data=Y_train, compression='gzip')
        if X_valid is not None:
            f.create_dataset('X_valid', data=X_valid, compression='gzip')
            f.create_dataset('Y_valid', data=Y_valid, compression='gzip')
        if X_test is not None:
            f.create_dataset('X_test', data=X_test, compression='gzip')
            f.create_dataset('Y_test', data=Y_test, compression='gzip')
        
        # Store metadata
        f.attrs['n_subjects'] = len(SUBJECTS)
        f.attrs['n_classes'] = len(CLASSES)
        f.attrs['classes'] = str(CLASSES)
        f.attrs['electrodes'] = str(Electrodes)
        f.attrs['sfreq'] = 250

    print(f"Saved successfully!")
    print("="*60)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Preprocess BCIC2020Track3 dataset')
    parser.add_argument('--data_folder', type=str, default='data/BCIC2020Track3',
                        help='Path to raw BCIC2020Track3 folder')
    parser.add_argument('--output_folder', type=str, default='data/processed',
                        help='Output folder for HDF5 file')
    parser.add_argument('--excel_path', type=str, default=None,
                        help='Path to test labels Excel file')
    args = parser.parse_args()

    preprocess_dataset(args.data_folder, args.output_folder, args.excel_path)


if __name__ == '__main__':
    main()
