"""
Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.
Currently under review for publication.
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import PretrainedConfig
import lightning as pl
from lightning.pytorch.callbacks import Callback
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

from FAST import FAST as Tower
from utils import green, yellow
from BCIC2020Track3_preprocess import Electrodes, Zones

# --- 1. Custom Callback for Learning Curve ---
class HistoryCallback(Callback):
    """Captures train/val loss and metrics at the end of every epoch."""
    def __init__(self):
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        # Training Metrics
        loss = trainer.callback_metrics.get('train_loss')
        acc = trainer.callback_metrics.get('train_acc')
        if loss is not None: self.history['loss'].append(loss.item())
        if acc is not None: self.history['acc'].append(acc.item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Validation Metrics
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

def load_standardized_h5(cache_fn):
    X, Y = [], []
    with h5py.File(cache_fn, 'r') as f:
        subjects = list(f.keys())
        for subject in subjects:
            X.append(f[subject]['X'][()])
            Y.append(f[subject]['Y'][()])
    X, Y = np.array(X), np.array(Y)
    print('Loaded from', cache_fn, X.shape, Y.shape)
    return X, Y

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

# --- FIXED: LOSO Pretraining (Added num_sanity_val_steps=0) ---
def Pretrain_LOSO(config, All_X, All_Y, target_subject_idx, save_dir, max_epochs=100, num_workers=4):
    ckpt_path = f"{save_dir}/Pretrain_excludes_sub{target_subject_idx}.pth"
    csv_log_path = f"{save_dir}/Pretrain_excludes_sub{target_subject_idx}_metrics.csv"
    plot_path = f"{save_dir}/Pretrain_excludes_sub{target_subject_idx}_curve.png"
    
    if os.path.exists(ckpt_path):
        print(f"Found existing pretraining for Subject {target_subject_idx}: {yellow(ckpt_path)}")
        return ckpt_path
    
    print(f"\n>>> Starting LOSO Pretraining (Excluding Subject {target_subject_idx}) <<<")
    
    # 1. Gather Data (All subjects except target)
    X_train_pool = []
    Y_train_pool = []
    
    for s in range(len(All_X)):
        if s != target_subject_idx:
            X_train_pool.append(All_X[s])
            Y_train_pool.append(All_Y[s])
            
    X_train_pool = np.concatenate(X_train_pool, axis=0)
    Y_train_pool = np.concatenate(Y_train_pool, axis=0)
    
    # 2. CREATE VALIDATION SPLIT (90% Train, 10% Val)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_pool, Y_train_pool, test_size=0.10, random_state=42, stratify=Y_train_pool
    )
    
    print(f"    Pooled Train: {X_train.shape} | Val: {X_val.shape}")
    
    train_data = BasicDataset(X_train, Y_train)
    val_data = BasicDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
    history_cb = HistoryCallback()
    
    trainer = pl.Trainer(
        strategy='auto', 
        accelerator='gpu', 
        devices=[args.gpu], 
        max_epochs=max_epochs,
        callbacks=[history_cb],
        enable_progress_bar=True,
        enable_checkpointing=False, 
        precision='bf16-mixed', 
        logger=False,
        num_sanity_val_steps=0  # <--- FIX: Prevents metric mismatch
    )
    
    # Pass both train and val loaders
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # 3. Save Metrics (With Safety Trimming)
    # This block ensures all lists are the same length before making the DataFrame
    min_len = min(len(v) for v in history_cb.history.values())
    clean_history = {k: v[:min_len] for k, v in history_cb.history.items()}
    
    df_metrics = pd.DataFrame(clean_history)
    df_metrics.to_csv(csv_log_path, index_label='Epoch')
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    if 'loss' in df_metrics: plt.plot(df_metrics['loss'], label='Train Loss', color='blue')
    if 'val_loss' in df_metrics: plt.plot(df_metrics['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title(f'LOSO Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    if 'acc' in df_metrics: plt.plot(df_metrics['acc'], label='Train Acc', color='green')
    if 'val_acc' in df_metrics: plt.plot(df_metrics['val_acc'], label='Val Acc', color='red', linestyle='--')
    plt.title(f'LOSO Pretraining Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    torch.save(model.model.state_dict(), ckpt_path)
    print(f"    LOSO Pretraining Saved: {green(ckpt_path)}")
    return ckpt_path

# --- FIXED: Finetune (Added num_sanity_val_steps=0) ---
def Finetune(config, Data_X, Data_Y, logf, subject_id, max_epochs=200, ckpt_pretrain=None, num_workers=0
):
    seed_all(42)
    Pred, Real = [], []
    kf = KFold(n_splits=5, shuffle=False)
    
    fold_histories = []
    fold_metrics = [] 
    best_fold_acc = -1.0

    fold_idx = 0
    for _train_idx, _test_idx in kf.split(Data_X):
        # 1. Standard CV Split (Train+Val vs Test)
        x_train_full, y_train_full = Data_X[_train_idx], Data_Y[_train_idx]
        x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

        # 2. INTERNAL VALIDATION SPLIT (90% Train, 10% Val)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
        )

        train_data = BasicDataset(x_train, y_train)
        val_data = BasicDataset(x_val, y_val)
        test_data = BasicDataset(x_test, y_test)

        # Optimize workers: 0 for small finetuning sets
        train_loader = DataLoader(train_data, batch_size=len(x_train), shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=len(x_val), shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=False, num_workers=0, pin_memory=True)

        model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
        
        if ckpt_pretrain is not None:
            print(f"Fold {fold_idx}: Loading LOSO weights...")
            model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

        print(f"Fold {fold_idx+1}/5 | Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")
        
        history_cb = HistoryCallback()
        
        trainer = pl.Trainer(
            strategy='auto', 
            accelerator='gpu', 
            devices=[args.gpu], 
            max_epochs=max_epochs, 
            callbacks=[history_cb], 
            enable_progress_bar=False, 
            enable_checkpointing=False, 
            precision='bf16-mixed', 
            logger=False,
            num_sanity_val_steps=0 # <--- FIX: Prevents metric mismatch
        )
        
        # Train with validation
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Safe Append with Trimming
        min_len = min(len(v) for v in history_cb.history.values())
        clean_history = {k: v[:min_len] for k, v in history_cb.history.items()}
        fold_histories.append(clean_history)
        
        # --- Inference on Test Set ---
        pred, real = inference_on_loader(model.model, test_loader)
        
        fold_acc = accuracy_score(real, pred)
        fold_f1 = f1_score(real, pred, average='macro')
        fold_metrics.append([fold_idx, fold_acc, fold_f1])
        
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            save_path = f"{Run}/{subject_id}_best.pth"
            torch.save(model.model.state_dict(), save_path)
            print(f"    -> New Best Fold (Acc: {green(f'{fold_acc:.4f}')}). Model saved.")

        Pred.append(pred)
        Real.append(real)
        fold_idx += 1
        
    Pred, Real = np.concatenate(Pred), np.concatenate(Real)
    
    # Global Metrics
    final_acc = accuracy_score(Real, Pred)
    final_f1 = f1_score(Real, Pred, average='macro')
    
    print(f"\n>>> Subject {subject_id} Completed <<<")
    print(f"    Aggregated Accuracy: {green(f'{final_acc:.4f}')}")
    
    np.savetxt(logf, np.array([Pred, Real]).T, delimiter=',', fmt='%d')
    
    # Save Metrics CSV
    metrics_arr = np.array(fold_metrics)
    metrics_file = f"{Run}/sub-{subject_id}_metrics.csv"
    with open(metrics_file, 'w') as f:
        f.write("Fold,Accuracy,F1_Score\n")
        for row in metrics_arr:
            f.write(f"{int(row[0])},{row[1]:.5f},{row[2]:.5f}\n")
        f.write(f"\nMEAN,{np.mean(metrics_arr[:, 1]):.5f},{np.mean(metrics_arr[:, 2]):.5f}\n")
    
    # --- VISUALIZATION: Average Learning Curves with Validation ---
    if fold_histories:
        def get_avg_metric(key):
            # Only average if key exists in all folds
            vals = [h[key] for h in fold_histories if key in h]
            if not vals: return []
            min_len = min(len(v) for v in vals)
            return np.mean([v[:min_len] for v in vals], axis=0)

        avg_train_loss = get_avg_metric('loss')
        avg_val_loss = get_avg_metric('val_loss')
        avg_train_acc = get_avg_metric('acc')
        avg_val_acc = get_avg_metric('val_acc')
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        if len(avg_train_loss) > 0: plt.plot(avg_train_loss, label='Avg Train Loss', color='blue')
        if len(avg_val_loss) > 0: plt.plot(avg_val_loss, label='Avg Val Loss', color='orange', linestyle='--')
        plt.title(f'Subject {subject_id} Fine-tuning Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(avg_train_acc) > 0: plt.plot(avg_train_acc, label='Avg Train Acc', color='green')
        if len(avg_val_acc) > 0: plt.plot(avg_val_acc, label='Avg Val Acc', color='red', linestyle='--')
        plt.title(f'Subject {subject_id} Fine-tuning Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        curve_path = f"{Run}/sub-{subject_id}_curve.png"
        plt.savefig(curve_path)
        plt.close()
        
    plt.figure(figsize=(6, 4))
    folds_idx = [int(x[0])+1 for x in fold_metrics]
    accs = [x[1] for x in fold_metrics]
    plt.bar(folds_idx, accs, color='teal', alpha=0.7, edgecolor='black')
    plt.axhline(y=np.mean(accs), color='red', linestyle='--', label=f'Mean: {np.mean(accs):.2f}')
    plt.title(f'Subject {subject_id}: 5-Fold Stability')
    plt.ylabel('Test Accuracy')
    plt.savefig(f"{Run}/sub-{subject_id}_folds.png")
    plt.close()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--folds', type=str, default='0-15')
    args.add_argument('--workers', type=int, default=4, help='Dataloader workers. -1 for auto.')
    args = args.parse_args()

    # Auto-detect workers
    if args.workers == -1:
        args.workers = min(8, os.cpu_count() or 1)
        print(f"Auto-detected workers: {args.workers}")

    if '-' in args.folds:
        start, end = [int(x) for x in args.folds.split('-')]
        args.folds = list(range(start, end))
    else:
        args.folds = [int(x) for x in args.folds.split(',')]

    Run = "Results_finetune/FAST/"
    os.makedirs(f"{Run}", exist_ok=True)

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
    
    X, Y = load_standardized_h5('Processed/BCIC2020Track3.h5')
    
    global_acc = []
    subject_ids = []

    for fold in range(15):
        if fold not in args.folds:
            continue
        
        flog = f"{Run}/{fold}-Tune.csv"
        
        # # 1. Pretrain (LOSO)
        # pretrained_path = Pretrain_LOSO(
        #     config, X, Y, target_subject_idx=fold, save_dir=Run, 
        #     max_epochs=50, num_workers=3
        # )
        
        # 2. Finetune (LOFO)
        Finetune(
            config, X[fold], Y[fold], flog, subject_id=fold, 
            max_epochs=200, num_workers=0
        )

        # Global stats logic...
        data = np.loadtxt(flog, delimiter=',', dtype=int)
        acc = accuracy_score(data[:, 1], data[:, 0])
        global_acc.append(acc)
        subject_ids.append(fold)
        print(f"    Rolling Avg Acc: {green(f'{np.mean(global_acc):.4f}')}")
        print("    --------------------------------------------------")

# Final Summary Plot
    if global_acc:
        avg = np.mean(global_acc)
        print(f"Final Average Accuracy: {avg:.4f}")
        
        plt.figure(figsize=(12, 6))
        
        # 1. Create Bar Plot and capture the bar objects
        bars = plt.bar(subject_ids, global_acc, color='skyblue', edgecolor='black')
        
        # 2. Add Mean Line with a Label
        plt.axhline(y=avg, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg:.4f}')
        
        # 3. Add Values on Top of Each Bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,  # X: Center of bar
                height,                               # Y: Top of bar
                f'{height:.2f}',                      # Text: Value to 2 decimal places
                ha='center', va='bottom',             # Alignment
                fontsize=10, fontweight='bold'
            )
        
        # 4. Final Formatting
        plt.title('Accuracy Contribution Breakdown per Subject', fontsize=14)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, max(global_acc) * 1.15) # Add headroom for labels
        plt.legend() # Displays the mean label
        
        plt.savefig(f"{Run}/Global_Accuracy_Breakdown.png", bbox_inches='tight')
        plt.close()