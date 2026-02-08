#!/usr/bin/env python
"""
FAST Baseline Training Script — Matches GitHub FAST (Jiang-Muyun) Approach

Replicates the original repo's evaluation strategy:
- Per-subject 5-fold CV (no shuffle) on combined train+validation data
- No separate validation set (train for fixed epochs, no early stopping)
- No held-out test set (KFold test fold = evaluation)
- Full-batch training (batch_size = len(x_train))
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(8)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from transformers import PretrainedConfig
import lightning as pl

logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

from fast.models import FAST
from fast.data import (
    BasicDataset, Electrodes, Zones, SUBJECTS, CLASSES,
    load_subject_train_val
)
from fast.train import EEG_Encoder_Module, inference_on_loader
from fast.utils import seed_all


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def finetune_baseline(config, args, save_dir, max_epochs=200):
    """
    Baseline: per-subject 5-fold CV matching the GitHub FAST repo.
    - KFold(n_splits=5, shuffle=False)
    - Full-batch training
    - Fixed epochs, no validation, no early stopping
    - KFold test fold = evaluation
    """
    seed_all(args.seed)

    base_folder = resolve_data_folder(args.data_folder)

    subjects = SUBJECTS
    subject_results = []
    global_pred = []
    global_true = []

    for SID in subjects:
        print("\n" + "=" * 60)
        print(f"SUBJECT {SID}: 5-FOLD BASELINE")
        print("=" * 60)

        X_sub, Y_sub = load_subject_train_val(base_folder, SID)
        print(f"Subject {SID} data: {X_sub.shape}, labels: {np.unique(Y_sub, return_counts=True)}")

        kf = KFold(n_splits=5, shuffle=False)
        fold_accs = []
        subj_pred = []
        subj_true = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_sub)):
            x_train, y_train = X_sub[train_idx], Y_sub[train_idx]
            x_test, y_test = X_sub[test_idx], Y_sub[test_idx]

            # Full-batch training (batch_size = entire training set)
            train_loader = DataLoader(
                BasicDataset(x_train, y_train), batch_size=len(x_train), shuffle=True,
                num_workers=args.workers, pin_memory=True
            )

            # niter_per_ep=1 since full-batch means 1 iteration per epoch
            model = EEG_Encoder_Module(config, max_epochs, niter_per_ep=1)

            trainer = pl.Trainer(
                strategy='auto',
                accelerator='gpu',
                devices=[args.gpu],
                max_epochs=max_epochs,
                callbacks=[],
                enable_progress_bar=True,
                enable_checkpointing=False,
                precision=args.precision,
                logger=False,
                num_sanity_val_steps=0
            )

            # Train only — no validation loader
            trainer.fit(model, train_dataloaders=train_loader)

            # Evaluate on KFold test fold
            test_loader = DataLoader(
                BasicDataset(x_test, y_test), batch_size=len(x_test), shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
            y_pred, y_true = inference_on_loader(model.model, test_loader)
            fold_acc = accuracy_score(y_true, y_pred)
            fold_accs.append(fold_acc)
            subj_pred.append(y_pred)
            subj_true.append(y_true)

            print(f"  Fold {fold_idx + 1}: test_acc={fold_acc:.4f}")

        # Aggregate across folds for this subject
        all_pred = np.concatenate(subj_pred)
        all_true = np.concatenate(subj_true)
        subj_acc = accuracy_score(all_true, all_pred)
        subj_f1 = f1_score(all_true, all_pred, average='macro')
        mean_fold_acc = np.mean(fold_accs)
        std_fold_acc = np.std(fold_accs)

        print(f"Subject {SID}: overall_acc={subj_acc:.4f}, mean_fold_acc={mean_fold_acc:.4f} +/- {std_fold_acc:.4f}")

        subject_results.append([SID, subj_acc, subj_f1, mean_fold_acc, std_fold_acc])
        global_pred.append(all_pred)
        global_true.append(all_true)

        # Save per-subject predictions
        subj_dir = os.path.join(save_dir, f"sub-{SID}")
        os.makedirs(subj_dir, exist_ok=True)
        np.savetxt(
            os.path.join(subj_dir, "cv_predictions.csv"),
            np.array([all_pred, all_true]).T, delimiter=',', fmt='%d',
            header='Predicted,True'
        )

    # Save overall summary
    df = pd.DataFrame(subject_results, columns=[
        'Subject', 'Overall_Acc', 'Overall_F1', 'Mean_Fold_Acc', 'Std_Fold_Acc'
    ])
    df.to_csv(os.path.join(save_dir, "summary_per_subject.csv"), index=False)

    # Global metrics
    gp = np.concatenate(global_pred)
    gt = np.concatenate(global_true)
    global_acc = accuracy_score(gt, gp)
    mean_subj_acc = df['Overall_Acc'].mean()
    std_subj_acc = df['Overall_Acc'].std()

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Mean Subject Accuracy: {mean_subj_acc:.4f} +/- {std_subj_acc:.4f}")
    print(f"Summary saved to {save_dir}/summary_per_subject.csv")
    print("=" * 60)
    print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description='FAST Baseline (matches GitHub repo)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='Training precision')
    parser.add_argument('--data_folder', type=str, default='BCIC2020Track3', help='Data folder path')
    parser.add_argument('--output_dir', type=str, default='results/baseline/FAST', help='Output directory')
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config)
        args.gpu = args.gpu if args.gpu != 0 else cfg.get('hardware', {}).get('gpu', 0)
        args.epochs = args.epochs if args.epochs != 200 else cfg.get('training', {}).get('max_epochs', 200)

    os.makedirs(args.output_dir, exist_ok=True)

    sfreq = 250
    config = PretrainedConfig(
        electrodes=Electrodes,
        zone_dict=Zones,
        dim_cnn=32,
        dim_token=32,
        seq_len=800,
        window_len=sfreq,
        slide_step=sfreq // 2,
        head='Conv4Layers',
        n_classes=5,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    )

    finetune_baseline(config, args, save_dir=args.output_dir, max_epochs=args.epochs)


if __name__ == '__main__':
    main()
