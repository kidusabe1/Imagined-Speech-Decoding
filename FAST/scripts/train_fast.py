#!/usr/bin/env python
"""
FAST Training Script with Official Train/Validation/Test Splits

Per-subject 5-fold CV finetuning with evaluation on official test set.
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

torch.set_num_threads(8)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from transformers import PretrainedConfig
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast.models import FAST
from fast.data import (
    BasicDataset, Electrodes, Zones, SUBJECTS, CLASSES,
    load_test_set_per_subject, load_subject_train_val
)
from fast.train import EEG_Encoder_Module, inference_on_loader, HistoryCallback
from fast.utils import seed_all, green, yellow


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


def resolve_excel_path(base_folder: str, excel_path: str | None) -> str:
    """Resolve Excel label sheet path with fallbacks."""
    candidates = []
    if excel_path:
        candidates.append(os.path.abspath(excel_path))
    candidates.append(os.path.join(base_folder, 'Test set', 'Track3_Answer Sheet_Test.xlsx'))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Test Excel not found. Tried: {candidates}")


def finetune_per_subject_cv(config, args, save_dir, max_epochs=200, batch_size=64):
    """
    Finetune per subject with 5-fold CV on that subject's train+validation data.
    Saves the best checkpoint (highest val_acc) per subject and reports test metrics.
    """
    seed_all(args.seed)

    base_folder = resolve_data_folder(args.data_folder)
    excel_path = resolve_excel_path(base_folder, args.excel_path)

    # Preload test data per subject
    test_per_subject = load_test_set_per_subject(base_folder, excel_path)

    subjects = SUBJECTS
    subject_results = []
    global_pred = []
    global_true = []

    for SID in subjects:
        print("\n" + "="*60)
        print(f"SUBJECT {SID}: 5-FOLD FINETUNE")
        print("="*60)

        X_sub, Y_sub = load_subject_train_val(base_folder, SID)
        print(f"Subject {SID} data: {X_sub.shape}, labels: {np.unique(Y_sub, return_counts=True)}")

        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_metrics = []
        best_fold_acc = -np.inf
        best_fold_ckpt = None
        best_train_len = 1

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_sub)):
            x_train, y_train = X_sub[train_idx], Y_sub[train_idx]
            x_val, y_val = X_sub[val_idx], Y_sub[val_idx]

            train_loader = DataLoader(
                BasicDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True
            )
            val_loader = DataLoader(
                BasicDataset(x_val, y_val), batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )

            model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
            history_cb = HistoryCallback()

            fold_dir = os.path.join(save_dir, f"sub-{SID}")
            os.makedirs(fold_dir, exist_ok=True)
            ckpt_cb = ModelCheckpoint(
                dirpath=fold_dir,
                filename=f"fold-{fold_idx}-best",
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                verbose=False
            )

            trainer = pl.Trainer(
                strategy='auto',
                accelerator='gpu',
                devices=[args.gpu],
                max_epochs=max_epochs,
                callbacks=[history_cb, ckpt_cb],
                enable_progress_bar=True,
                enable_checkpointing=True,
                precision=args.precision,
                logger=False,
                num_sanity_val_steps=0
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            fold_best = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float('nan')
            fold_metrics.append([fold_idx, fold_best])

            # Save fold learning curves
            if all(len(v) > 0 for v in history_cb.history.values()):
                min_len = min(len(v) for v in history_cb.history.values())
                clean_history = {k: v[:min_len] for k, v in history_cb.history.items()}
                hist_path = os.path.join(fold_dir, f"fold-{fold_idx}_history.csv")
                pd.DataFrame(clean_history).to_csv(hist_path, index_label='Epoch')

                # Plot learning curves
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                if 'loss' in clean_history:
                    plt.plot(clean_history['loss'], label='Train Loss', color='blue')
                if 'val_loss' in clean_history:
                    plt.plot(clean_history['val_loss'], label='Val Loss', color='orange', linestyle='--')
                plt.title(f'Subject {SID} Fold {fold_idx+1} Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                if 'acc' in clean_history:
                    plt.plot(clean_history['acc'], label='Train Acc', color='green')
                if 'val_acc' in clean_history:
                    plt.plot(clean_history['val_acc'], label='Val Acc', color='red', linestyle='--')
                plt.title(f'Subject {SID} Fold {fold_idx+1} Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(fold_dir, f"fold-{fold_idx}_curves.png"))
                plt.close()

            print(f"  Fold {fold_idx+1}: best val_acc={fold_best:.4f}")

            if ckpt_cb.best_model_path and fold_best > best_fold_acc:
                best_fold_acc = fold_best
                best_fold_ckpt = ckpt_cb.best_model_path
                best_train_len = len(train_loader)

        # Load best fold model for this subject
        if best_fold_ckpt:
            best_model = EEG_Encoder_Module.load_from_checkpoint(
                best_fold_ckpt,
                config=config,
                max_epochs=max_epochs,
                niter_per_ep=best_train_len,
                weights_only=False
            )
            subj_best_path = os.path.join(save_dir, f"sub-{SID}", "best_subject.pth")
            torch.save(best_model.model.state_dict(), subj_best_path)
            print(f"Saved best model for subject {SID} -> {subj_best_path}")
        else:
            print(f"No checkpoint found for subject {SID}; skipping save.")
            continue

        # Evaluate on official test set
        test_acc = np.nan
        test_f1 = np.nan
        if SID in test_per_subject:
            X_test, Y_test = test_per_subject[SID]
            test_loader = DataLoader(
                BasicDataset(X_test, Y_test), batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )
            y_pred, y_true = inference_on_loader(best_model.model, test_loader)
            test_acc = accuracy_score(y_true, y_pred)
            test_f1 = f1_score(y_true, y_pred, average='macro')
            np.savetxt(
                os.path.join(save_dir, f"sub-{SID}", "test_predictions.csv"),
                np.array([y_pred, y_true]).T, delimiter=',', fmt='%d', header='Predicted,True'
            )
            print(f"Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
            global_pred.append(y_pred)
            global_true.append(y_true)

        subject_results.append([SID, best_fold_acc, test_acc, test_f1])

        # Save fold metrics
        df_folds = pd.DataFrame(fold_metrics, columns=['Fold', 'Best_Val_Acc'])
        df_folds.to_csv(os.path.join(save_dir, f"sub-{SID}", "fold_metrics.csv"), index=False)

    # Save overall summary
    df_subjects = pd.DataFrame(subject_results, columns=['Subject', 'Best_Val_Acc', 'Test_Acc', 'Test_F1'])
    df_subjects.to_csv(os.path.join(save_dir, "summary_per_subject.csv"), index=False)

    # Global predictions
    if global_pred and global_true:
        gp = np.concatenate(global_pred)
        gt = np.concatenate(global_true)
        np.savetxt(
            os.path.join(save_dir, "global_test_predictions.csv"),
            np.array([gp, gt]).T, delimiter=',', fmt='%d', header='Predicted,True'
        )

    # Global subjects accuracy plot
    if not df_subjects.empty:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_subjects['Subject'], df_subjects['Test_Acc'], color='skyblue', edgecolor='black')
        mean_acc = df_subjects['Test_Acc'].mean()
        plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.4f}')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=9)
        plt.title('Test Accuracy per Subject (Finetune CV)', fontsize=14)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, max(df_subjects['Test_Acc'].max(), mean_acc) * 1.15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "global_subject_accuracy.png"))
        plt.close()

    print("\n" + "="*60)
    print("FINETUNE COMPLETE")
    print(f"Summary saved to {save_dir}/summary_per_subject.csv")
    print("="*60)

    return df_subjects


def main():
    parser = argparse.ArgumentParser(description='Train FAST model on BCI Competition 2020 Track #3')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='Training precision')
    parser.add_argument('--data_folder', type=str, default='data/BCIC2020Track3', help='Data folder path')
    parser.add_argument('--excel_path', type=str, default='data/BCIC2020Track3/Test set/Track3_Answer Sheet_Test.xlsx', help='Test labels Excel path')
    parser.add_argument('--output_dir', type=str, default='results/finetune_official/FAST', help='Output directory')
    args = parser.parse_args()

    # Load config if provided
    if os.path.exists(args.config):
        cfg = load_config(args.config)
        # Override with command line args
        args.gpu = args.gpu if args.gpu != 0 else cfg.get('hardware', {}).get('gpu', 0)
        args.epochs = args.epochs if args.epochs != 200 else cfg.get('training', {}).get('max_epochs', 200)
        args.batch_size = args.batch_size if args.batch_size != 64 else cfg.get('training', {}).get('batch_size', 64)

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
    
    finetune_per_subject_cv(
        config, args, save_dir=args.output_dir,
        max_epochs=args.epochs, batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
