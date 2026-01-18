"""
TSception Training Pipeline (LOFO / 5-Fold CV)
Benchmarks TSception on BCI Competition 2020 Track #3 without pretraining.
- Updated: Fixed 200 Epochs (No Early Stopping).
- Updated: Plots and saves learning curves (Loss/Acc) for every fold.
- Updated: Auto-saves the best model (best fold) for each subject.
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
import h5py
import einops
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as pl
import logging

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Reduce logging clutter
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

# ==============================================================================
# MODEL DEFINITION: TSception
# ==============================================================================
class TSception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        """
        Args:
            input_size: (1, channel, time)
            sampling_rate: Hz (250 for this dataset)
            num_T: Number of temporal kernels per scale
            num_S: Number of spatial kernels
        """
        super(TSception, self).__init__()
        
        self.num_classes = num_classes
        self.channel = input_size[1]
        self.time = input_size[2]
        self.sampling_rate = sampling_rate
        self.num_T = num_T
        self.num_S = num_S
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.pool = 4 

        # --- Temporal Learner (Multi-scale 1D Convolutions) ---
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.sampling_rate * 0.5)), 1, self.pool, padding='same')
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.sampling_rate * 0.25)), 1, self.pool, padding='same')
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.sampling_rate * 0.125)), 1, self.pool, padding='same')

        # --- Spatial Learner (Hemisphere-Aware Convolutions) ---
        self.Sception1 = self.conv_block(num_T * 3, num_S, (self.channel, 1), 1, self.pool, padding=0)
        self.Sception2 = self.conv_block(num_T * 3, num_S, (int(self.channel * 0.5), 1), (int(self.channel * 0.5), 1), self.pool, padding=0)

        # --- Fusion & Classification ---
        self.BN_t = nn.BatchNorm2d(num_T * 3)
        self.BN_s = nn.BatchNorm2d(num_S)
        
        self.global_pool = nn.AdaptiveAvgPool2d((None, 8)) 
        
        size = num_S * 3 * 8 
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def conv_block(self, in_chan, out_chan, kernel, step, pool, padding=0):
        if padding == 'same':
             return nn.Sequential(
                nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel, stride=step, padding='same'),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel, stride=step, padding=padding),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))
            )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Temporal Processing
        y_t1 = self.Tception1(x)
        y_t2 = self.Tception2(x)
        y_t3 = self.Tception3(x)
        y_t = torch.cat((y_t1, y_t2, y_t3), dim=1) 
        y_t = self.BN_t(y_t)

        # Spatial Processing
        y_s1 = self.Sception1(y_t)
        y_s2 = self.Sception2(y_t)
        y_s = torch.cat((y_s1, y_s2), dim=2)
        y_s = self.BN_s(y_s)

        # Classification
        y = self.global_pool(y_s)
        y = self.fc(y)
        return y

# ==============================================================================
# UTILITIES
# ==============================================================================
def green(text): return f"\033[92m{text}\033[0m"

class HistoryCallback(Callback):
    """Captures train/val loss and metrics reliably at the end of every epoch."""
    def __init__(self):
        # We use explicit keys to ensure we track exactly what we want
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        # Try fetching from callback_metrics first, then logged_metrics
        # logged_metrics is often more reliable for 'on_epoch=True' aggregations
        metrics = trainer.callback_metrics
        
        loss = metrics.get('train_loss')
        acc = metrics.get('train_acc')
        
        # Fallback: Sometimes it's stored in logged_metrics
        if loss is None: loss = trainer.logged_metrics.get('train_loss')
        if acc is None: acc = trainer.logged_metrics.get('train_acc')

        if loss is not None: 
            self.history['loss'].append(loss.item())
        if acc is not None: 
            self.history['acc'].append(acc.item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        val_loss = metrics.get('val_loss')
        val_acc = metrics.get('val_acc')
        
        if val_loss is None: val_loss = trainer.logged_metrics.get('val_loss')
        if val_acc is None: val_acc = trainer.logged_metrics.get('val_acc')

        if val_loss is not None: 
            self.history['val_loss'].append(val_loss.item())
        if val_acc is not None: 
            self.history['val_acc'].append(val_acc.item())

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

def load_standardized_h5(cache_fn):
    X, Y = [], []
    with h5py.File(cache_fn, 'r') as f:
        subjects = sorted(list(f.keys()))
        for subject in subjects:
            X.append(f[subject]['X'][()])
            Y.append(f[subject]['Y'][()])
    X, Y = np.array(X), np.array(Y)
    print(f'Loaded from {cache_fn} | Total Shape: {X.shape}, {Y.shape}')
    return X, Y

class BasicDataset(Dataset):
    def __init__(self, data, label):
        self.data, self.labels = torch.from_numpy(data).float(), torch.from_numpy(label).long()
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def inference_on_loader(model, loader, device='cuda'):
    model.eval()
    model.to(device)
    with torch.no_grad():
        Pred, Real = [], []
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu()
            Pred.append(preds)
            Real.append(y)
        Pred, Real = torch.cat(Pred), torch.cat(Real)
    return Pred.numpy(), Real.numpy()

# ==============================================================================
# LIGHTNING MODULE
# ==============================================================================
class TSception_Module(pl.LightningModule):
    def __init__(self, num_classes, chn_num, sampling_rate, input_time_len, learning_rate=1e-3, max_epochs=200):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
            )

        
        self.model = TSception(
            num_classes=num_classes,
            input_size=(1, chn_num, input_time_len), 
            sampling_rate=sampling_rate,       
            num_T=15,                
            num_S=15,                
            hidden=128,
            dropout_rate=0.5
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = self.train_acc(logits, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# ==============================================================================
# MAIN TRAINING ROUTINE (LOFO)
# ==============================================================================
def Train_Subject_LOFO(Data_X, Data_Y, logf, subject_id, max_epochs, gpu_id):
    seed_all(42)
    
    kf = KFold(n_splits=5, shuffle=False)
    
    fold_histories = []
    fold_metrics = [] 
    Pred_All, Real_All = [], []
    
    n_samples, n_channels, n_time = Data_X.shape
    sampling_rate = 250 

    # --- TRACK BEST FOLD ---
    best_fold_acc = -1.0
    # -----------------------

    fold_idx = 0
    for _train_idx, _test_idx in kf.split(Data_X):
        # 1. Split Data
        x_train_full, y_train_full = Data_X[_train_idx], Data_Y[_train_idx]
        x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

        # 2. Internal Validation Split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
        )

        train_loader = DataLoader(BasicDataset(x_train, y_train), batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(BasicDataset(x_val, y_val), batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(BasicDataset(x_test, y_test), batch_size=32, shuffle=False, num_workers=0)

        print(f"Sub-{subject_id} | Fold {fold_idx+1}/5 | Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

        # 3. Initialize Model
        model = TSception_Module(
            num_classes=5, 
            chn_num=n_channels,        
            sampling_rate=sampling_rate, 
            input_time_len=n_time,     
            max_epochs=max_epochs
        )
        
        history_cb = HistoryCallback()
        
        # --- FIXED: REMOVED EARLY STOPPING ---
        trainer = pl.Trainer(
            accelerator='gpu', devices=[gpu_id],
            max_epochs=max_epochs,
            callbacks=[history_cb], # No EarlyStopping here
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
            precision='bf16-mixed'
        )
        
        # 4. Train
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # --- NEW: SAVE HISTORY & PLOT CURVES ---
        # Ensure we don't crash if history is empty or keys are missing
        if history_cb.history['loss']:
            min_len = min(len(v) for v in history_cb.history.values() if len(v) > 0)
            clean_history = {k: v[:min_len] for k, v in history_cb.history.items() if len(v) > 0}
            
            # Save CSV
            fold_save_dir = os.path.dirname(logf)
            os.makedirs(fold_save_dir, exist_ok=True) # Ensure dir exists
            
            hist_df = pd.DataFrame(clean_history)
            hist_csv_path = os.path.join(fold_save_dir, f"sub-{subject_id}_fold-{fold_idx+1}_history.csv")
            hist_df.to_csv(hist_csv_path, index_label='Epoch')
            
            # Plot Curve
            plt.figure(figsize=(12, 5))
            
            # Loss Subplot
            plt.subplot(1, 2, 1)
            if 'loss' in hist_df: plt.plot(hist_df['loss'], label='Train Loss', color='blue')
            if 'val_loss' in hist_df: plt.plot(hist_df['val_loss'], label='Val Loss', color='orange', linestyle='--')
            plt.title(f'Sub-{subject_id} Fold-{fold_idx+1} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Accuracy Subplot
            plt.subplot(1, 2, 2)
            if 'acc' in hist_df: plt.plot(hist_df['acc'], label='Train Acc', color='green')
            if 'val_acc' in hist_df: plt.plot(hist_df['val_acc'], label='Val Acc', color='red', linestyle='--')
            plt.title(f'Sub-{subject_id} Fold-{fold_idx+1} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(fold_save_dir, f"sub-{subject_id}_fold-{fold_idx+1}_curve.png")
            plt.savefig(plot_path)
            
            # Show plot if in a notebook environment
            plt.show() 
            plt.close()
        else:
            print(f"Warning: No history logged for Subject {subject_id} Fold {fold_idx+1}")
        # ---------------------------------------
        
        fold_histories.append(clean_history)
        
        # 5. Test Inference
        pred, real = inference_on_loader(model.model, test_loader, device=f'cuda:{gpu_id}')
        
        acc = accuracy_score(real, pred)
        f1 = f1_score(real, pred, average='macro')

        # --- SAVE BEST MODEL LOGIC ---
        if acc > best_fold_acc:
            best_fold_acc = acc
            # Construct path: Results_TSception_LOFO/sub-XX_best.pth
            save_dir = os.path.dirname(logf)
            save_path = os.path.join(save_dir, f"sub-{subject_id}_best.pth")
            
            # Save the inner PyTorch model weights
            torch.save(model.model.state_dict(), save_path)
            print(f"    [Save] New Best Model for Subject {subject_id} (Fold {fold_idx+1}, Acc: {green(f'{acc:.4f}')}) -> {save_path}")
        # -----------------------------
        
        fold_metrics.append([fold_idx, acc, f1])
        Pred_All.append(pred)
        Real_All.append(real)
        
        fold_idx += 1

    Pred_All, Real_All = np.concatenate(Pred_All), np.concatenate(Real_All)
    final_acc = accuracy_score(Real_All, Pred_All)
    
    print(f"\n>>> Subject {subject_id} Completed <<<")
    print(f"    Aggregated Accuracy: {green(f'{final_acc:.4f}')}")
    
    np.savetxt(logf, np.array([Pred_All, Real_All]).T, delimiter=',', fmt='%d')

    metrics_arr = np.array(fold_metrics)
    metrics_file = logf.replace('.csv', '_metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write("Fold,Accuracy,F1_Score\n")
        for row in metrics_arr:
            f.write(f"{int(row[0])},{row[1]:.5f},{row[2]:.5f}\n")
        f.write(f"\nMEAN,{np.mean(metrics_arr[:, 1]):.5f},{np.mean(metrics_arr[:, 2]):.5f}\n")

    return final_acc

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--folds', type=str, default='0-15', help="Range of subjects to process (e.g. 0-15)")
    args = parser.parse_args()

    if '-' in args.folds:
        start, end = [int(x) for x in args.folds.split('-')]
        subjects_to_run = list(range(start, end))
    else:
        subjects_to_run = [int(x) for x in args.folds.split(',')]

    Run_Name = "Results_TSception_LOFO"
    os.makedirs(Run_Name, exist_ok=True)
    
    X_All, Y_All = load_standardized_h5('Processed/BCIC2020Track3.h5')
    
    global_accuracies = []
    
    print(f"\nStarting TSception Benchmark on {len(subjects_to_run)} Subjects...")
    print(f"Saving results to: {Run_Name}\n")

    for subject_idx in subjects_to_run:
        if subject_idx >= len(X_All):
            print(f"Skipping Subject {subject_idx} (Not in dataset)")
            continue

        log_file = f"{Run_Name}/sub-{subject_idx}_preds.csv"
        
        acc = Train_Subject_LOFO(
            X_All[subject_idx], 
            Y_All[subject_idx], 
            log_file, 
            subject_id=subject_idx, 
            max_epochs=200, 
            gpu_id=args.gpu
        )
        
        global_accuracies.append(acc)
        print("--------------------------------------------------")

    if global_accuracies:
        mean_acc = np.mean(global_accuracies)
        print(f"\n=== BENCHMARK COMPLETE ===")
        print(f"TSception Mean Accuracy across {len(global_accuracies)} subjects: {mean_acc:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.bar(subjects_to_run, global_accuracies, color='purple', alpha=0.7, edgecolor='black')
        plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.2f}')
        plt.xlabel('Subject ID')
        plt.ylabel('Accuracy')
        plt.title('TSception Performance per Subject (No Pretraining)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(f"{Run_Name}/Global_Accuracy_Summary.png")
        print(f"Summary plot saved to {Run_Name}/Global_Accuracy_Summary.png")