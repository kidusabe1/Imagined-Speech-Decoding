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
from sklearn.model_selection import KFold
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
    """Captures loss and metrics at the end of every epoch."""
    def __init__(self):
        self.history = {'loss': [], 'acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        acc = trainer.callback_metrics.get('train_acc')
        
        if loss is not None:
            self.history['loss'].append(loss.item())
        if acc is not None:
            self.history['acc'].append(acc.item())

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
        
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
        self.train_f1  = torchmetrics.F1Score('multiclass', num_classes=config.n_classes, average='macro')

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

def Finetune(config, Data_X, Data_Y, logf, subject_id, max_epochs=200, ckpt_pretrain=None):
    seed_all(42)
    Pred, Real = [], []
    kf = KFold(n_splits=5, shuffle=False)
    
    fold_histories = []

    fold_idx = 0
    for _train_idx, _test_idx in kf.split(Data_X):
        x_train, y_train = Data_X[_train_idx], Data_Y[_train_idx]
        x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

        train_data = BasicDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size=len(x_train), shuffle=True, num_workers=4, pin_memory=True)
        test_data = BasicDataset(x_test, y_test)
        test_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=False, num_workers=4, pin_memory=True)

        model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
        if ckpt_pretrain is not None:
            model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

        print(f"Fold {fold_idx+1}/5 | {yellow(logf)} | Train: {x_train.shape} | Test: {x_test.shape}")
        
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
            logger=False
        )
        
        trainer.fit(model, train_dataloaders=train_loader)
        fold_histories.append(history_cb.history['loss'])
        
        # torch.save(model.model.state_dict(), f"{Run}/{subject_id}_fold{fold_idx}.pth")

        pred, real = inference_on_loader(model.model, test_loader)
        Pred.append(pred)
        Real.append(real)
        fold_idx += 1
        
    Pred, Real = np.concatenate(Pred), np.concatenate(Real)
    
    final_acc = accuracy_score(Real, Pred)
    final_f1 = f1_score(Real, Pred, average='macro')
    
    print(f"\n>>> Subject {subject_id} Completed <<<")
    print(f"    Accuracy: {green(f'{final_acc:.4f}')}")
    print(f"    F1 Score: {green(f'{final_f1:.4f}')}")
    
    np.savetxt(logf, np.array([Pred, Real]).T, delimiter=',', fmt='%d')

    if fold_histories:
        min_len = min([len(h) for h in fold_histories])
        avg_loss = np.mean([h[:min_len] for h in fold_histories], axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_loss, label=f'Sub {subject_id} Avg Loss', color='blue')
        
        min_loss_val = np.min(avg_loss)
        min_loss_idx = np.argmin(avg_loss)
        final_loss_val = avg_loss[-1]
        
        plt.annotate(f'Min: {min_loss_val:.4f}', 
                     xy=(min_loss_idx, min_loss_val), 
                     xytext=(min_loss_idx, min_loss_val + 0.2),
                     arrowprops=dict(facecolor='green', shrink=0.05))
        
        plt.annotate(f'Final: {final_loss_val:.4f}', 
                     xy=(len(avg_loss)-1, final_loss_val), 
                     xytext=(len(avg_loss)-20, final_loss_val + 0.2),
                     arrowprops=dict(facecolor='red', shrink=0.05))

        plt.title(f'Learning Curve - Subject {subject_id} (5-Fold Avg)')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plot_path = f"{os.path.dirname(logf)}/sub_{subject_id}_curve.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"    Learning curve saved to: {plot_path}\n")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--folds', type=str, default='0-15')
    args = args.parse_args()

    if '-' in args.folds:
        start, end = [int(x) for x in args.folds.split('-')]
        args.folds = list(range(start, end))
    else:
        args.folds = [int(x) for x in args.folds.split(',')]

    Run = "Results/FAST/"
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
    global_f1 = []
    
    # Keep track of which subject ID corresponds to which accuracy
    subject_ids = []

    for fold in range(15):
        if fold not in args.folds:
            continue
        flog = f"{Run}/{fold}-Tune.csv"
        
        Finetune(config, X[fold], Y[fold], flog, subject_id=fold, max_epochs=200)

        data = np.loadtxt(flog, delimiter=',', dtype=int)
        pred, label = data[:, 0], data[:, 1]
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        
        global_acc.append(acc)
        global_f1.append(f1)
        subject_ids.append(fold)
        
        mean_acc = np.mean(global_acc)
        mean_f1 = np.mean(global_f1)
        print(f"    [Rolling Average (N={len(global_acc)})] Acc: {green(f'{mean_acc:.4f}')} | F1: {green(f'{mean_f1:.4f}')}")
        print("    --------------------------------------------------")

    # --- Final Summary & Visualization ---
    print("\n========================================")
    print("      ALL SUBJECTS EVALUATION COMPLETE    ")
    print("========================================")
    
    if global_acc:
        avg_acc = np.mean(global_acc)
        std_acc = np.std(global_acc)
        
        print(f"Final Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Final Average F1 Score: {np.mean(global_f1):.4f} ± {np.std(global_f1):.4f}")
        
        # --- NEW: Breakdown Plot ---
        plt.figure(figsize=(12, 6))
        
        # Bar chart for individual subjects
        bars = plt.bar(subject_ids, global_acc, color='skyblue', edgecolor='black', alpha=0.7, label='Subject Accuracy')
        
        # Horizontal line for global average
        plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, label=f'Mean ({avg_acc:.2f})')
        
        # Shaded area for Std Dev
        plt.fill_between([min(subject_ids)-0.5, max(subject_ids)+0.5], 
                         avg_acc - std_acc, avg_acc + std_acc, 
                         color='red', alpha=0.1, label='Std Dev')

        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=9)

        plt.xticks(subject_ids)
        plt.xlabel('Subject ID')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Contribution Breakdown per Subject')
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        final_plot_path = f"{Run}/Global_Accuracy_Breakdown.png"
        plt.savefig(final_plot_path)
        plt.close()
        print(f"\n[Visual] Breakdown plot saved to: {final_plot_path}")
        
    else:
        print("No results found.")