"""
Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.

This version uses the OFFICIAL train/validation/test splits from the dataset:
- Training set: All subjects from Training set folder (for training)
- Validation set: All subjects from Validation set folder (for validation during training)
- Test set: All subjects from Test set folder (for final evaluation)

Labels for test set are loaded from the Excel answer sheet.

Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
torch.set_num_threads(8)
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchmetrics
import logging
import h5py
import einops
import scipy.io
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from transformers import PretrainedConfig
import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

from FAST import FAST as Tower
from utils import green, yellow
from BCIC2020Track3_preprocess import Electrodes, Zones

# --- Constants ---
SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
CLASSES = ['hello', 'help-me', 'stop', 'thank-you', 'yes']
TARGET_TIMEPOINTS = 800  # Pad to 800 for model input

# --- 1. Custom Callback for Learning Curve ---
class HistoryCallback(Callback):
    """Captures train/val loss and metrics at the end of every epoch."""
    def __init__(self):
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        acc = trainer.callback_metrics.get('train_acc')
        if loss is not None: self.history['loss'].append(loss.item())
        if acc is not None: self.history['acc'].append(acc.item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_loss is not None: self.history['val_loss'].append(val_loss.item())
        if val_acc is not None: self.history['val_acc'].append(val_acc.item())

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class BasicDataset(Dataset):
    def __init__(self, data, label):
        if len(data.shape) == 4:
            data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
        self.data, self.labels = torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        return sample, label


class EEG_Encoder_Module(pl.LightningModule):
    def __init__(self, config, max_epochs, niter_per_ep):
        super().__init__()
        self.config = config
        self.model = Tower(config)
        self.loss = nn.CrossEntropyLoss()
        self.cosine_lr_list = cosine_scheduler(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=10)
        
        # Training Metrics
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
        self.train_f1  = torchmetrics.F1Score('multiclass', num_classes=config.n_classes, average='macro')
        
        # Validation Metrics
        self.val_acc = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
        self.val_f1 = torchmetrics.F1Score('multiclass', num_classes=config.n_classes, average='macro')

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: self.cosine_lr_list[self.global_step-1])
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        
        acc = self.train_acc(logits, y)
        f1 = self.train_f1(logits, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        
        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def inference_on_loader(model, loader):
    model.eval()
    model.cuda()
    with torch.no_grad():
        Pred, Real = [], []
        for x, y in loader:
            preds = torch.argmax(model(x.cuda()), dim=1).cpu()
            Pred.append(preds)
            Real.append(y)
        Pred, Real = torch.cat(Pred), torch.cat(Real)
    return Pred.numpy(), Real.numpy()


# =============================================================================
# DATA LOADING FUNCTIONS FOR OFFICIAL SPLITS
# =============================================================================

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
    """
    Load all subjects from the Test set folder.
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
                    # Shape is (Trials, Channels, Time)
                    if x.ndim == 3 and x.shape[2] < TARGET_TIMEPOINTS:
                        pad_width = TARGET_TIMEPOINTS - x.shape[2]
                        x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), 'edge')
                    x = x.astype(np.float32)
                    
                    # Load labels from Excel (columns 2, 4, 6, ... for subjects 1-15)
                    col_idx = 2 * (i + 1)  # Data located in columns 2, 4, 6...
                    raw_labels = pd.to_numeric(df_labels.iloc[3:53, col_idx], errors='coerce').values
                    y = (raw_labels - 1).astype(np.uint8)  # Convert 1-5 to 0-4
                    
                    X_all.append(x)
                    Y_all.append(y)
                    print(f"  Test  S{SID}: {x.shape}, labels: {np.unique(y, return_counts=True)}")
    
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all


def load_test_set_per_subject(base_folder, excel_path):
    """
    Load test set data separately per subject for per-subject evaluation.
    Returns dict: {subject_id: (X, Y)}
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


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def finetune_per_subject_cv(config, args, save_dir, max_epochs=200, batch_size=64):
    """
    Finetune per subject with 5-fold CV on that subject's train+validation data.
    Saves the best checkpoint (highest val_acc) per subject and reports test metrics.
    """
    seed_all(42)

    base_folder = 'BCIC2020Track3'
    excel_path = os.path.join(base_folder, 'Test set', 'Track3_Answer Sheet_Test.xlsx')
    if not os.path.exists(excel_path):
        excel_path = 'Track3_Answer Sheet_Test.xlsx'

    # Preload test data per subject (official labels from Excel)
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

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        best_fold_acc = -np.inf
        best_fold_ckpt = None
        best_train_len = 1

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_sub)):
            x_train, y_train = X_sub[train_idx], Y_sub[train_idx]
            x_val, y_val = X_sub[val_idx], Y_sub[val_idx]

            train_loader = DataLoader(BasicDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)
            val_loader = DataLoader(BasicDataset(x_val, y_val), batch_size=batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

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
                precision='bf16-mixed',
                logger=False,
                num_sanity_val_steps=0
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            fold_best = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float('nan')
            fold_metrics.append([fold_idx, fold_best])

            # Save fold learning curves (train/val loss & acc)
            if all(len(v) > 0 for v in history_cb.history.values()):
                min_len = min(len(v) for v in history_cb.history.values())
                clean_history = {k: v[:min_len] for k, v in history_cb.history.items()}
                hist_path = os.path.join(fold_dir, f"fold-{fold_idx}_history.csv")
                pd.DataFrame(clean_history).to_csv(hist_path, index_label='Epoch')

                # Plot learning curves for this fold
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                if 'loss' in clean_history: plt.plot(clean_history['loss'], label='Train Loss', color='blue')
                if 'val_loss' in clean_history: plt.plot(clean_history['val_loss'], label='Val Loss', color='orange', linestyle='--')
                plt.title(f'Subject {SID} Fold {fold_idx+1} Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                if 'acc' in clean_history: plt.plot(clean_history['acc'], label='Train Acc', color='green')
                if 'val_acc' in clean_history: plt.plot(clean_history['val_acc'], label='Val Acc', color='red', linestyle='--')
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
                weights_only=False  # allow loading full checkpoint (new torch default changed)
            )
            subj_best_path = os.path.join(save_dir, f"sub-{SID}", "best_subject.pth")
            torch.save(best_model.model.state_dict(), subj_best_path)
            print(f"Saved best model for subject {SID} -> {subj_best_path}")
        else:
            print(f"No checkpoint found for subject {SID}; skipping save.")
            continue

        # Evaluate on official test set for this subject (if available)
        test_acc = np.nan
        test_f1 = np.nan
        if SID in test_per_subject:
            X_test, Y_test = test_per_subject[SID]
            test_loader = DataLoader(BasicDataset(X_test, Y_test), batch_size=batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)
            y_pred, y_true = inference_on_loader(best_model.model, test_loader)
            test_acc = accuracy_score(y_true, y_pred)
            test_f1 = f1_score(y_true, y_pred, average='macro')
            np.savetxt(os.path.join(save_dir, f"sub-{SID}", "test_predictions.csv"),
                       np.array([y_pred, y_true]).T, delimiter=',', fmt='%d', header='Predicted,True')
            print(f"Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
            global_pred.append(y_pred)
            global_true.append(y_true)

        subject_results.append([SID, best_fold_acc, test_acc, test_f1])

        # Save fold metrics per subject
        df_folds = pd.DataFrame(fold_metrics, columns=['Fold', 'Best_Val_Acc'])
        df_folds.to_csv(os.path.join(save_dir, f"sub-{SID}", "fold_metrics.csv"), index=False)

    # Save overall summary
    df_subjects = pd.DataFrame(subject_results, columns=['Subject', 'Best_Val_Acc', 'Test_Acc', 'Test_F1'])
    df_subjects.to_csv(os.path.join(save_dir, "summary_per_subject.csv"), index=False)

    # Global predictions concatenated
    if global_pred and global_true:
        gp = np.concatenate(global_pred)
        gt = np.concatenate(global_true)
        np.savetxt(os.path.join(save_dir, "global_test_predictions.csv"),
                   np.array([gp, gt]).T, delimiter=',', fmt='%d', header='Predicted,True')

    # Global subjects accuracy plot
    if not df_subjects.empty:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_subjects['Subject'], df_subjects['Test_Acc'], color='skyblue', edgecolor='black')
        mean_acc = df_subjects[df_subjects['Subject'] != 'MEAN']['Test_Acc'].mean()
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


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--workers', type=int, default=4, help='Dataloader workers.')
    args.add_argument('--epochs', type=int, default=200, help='Max training epochs.')
    args.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    args = args.parse_args()

    Run = "Results_finetune_official/FAST/"
    os.makedirs(Run, exist_ok=True)

    sfreq = 250
    config = PretrainedConfig(
        electrodes=Electrodes,
        zone_dict=Zones,
        dim_cnn=32,
        dim_token=32,
        seq_len=800,
        window_len=sfreq,
        slide_step=sfreq//2,
        head='Conv4Layers',
        n_classes=5,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    )
    
    finetune_per_subject_cv(
        config, args, save_dir=Run,
        max_epochs=args.epochs, batch_size=args.batch_size
    )
