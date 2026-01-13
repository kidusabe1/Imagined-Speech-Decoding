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
import matplotlib.pyplot as plt  # Added for plotting
torch.set_num_threads(8)
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchmetrics
import logging
import h5py
import einops
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score # Added for explicit metric calculation
from transformers import PretrainedConfig
import lightning as pl
from lightning.pytorch.callbacks import Callback # Added for history tracking
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
        # Retrieve metrics logged in training_step
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
        
        # Metrics tracking during training
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
        
        # Log metrics for the callback
        acc = self.train_acc(logits, y)
        f1 = self.train_f1(logits, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

def Finetune(config, Data_X, Data_Y, logf, subject_id, max_epochs=200, ckpt_pretrain=None):
    """
    Train per subject, report metrics, and save learning curve.
    """
    seed_all(42)
    Pred, Real = [], []
    kf = KFold(n_splits=5, shuffle=False)
    
    # Store history for all 5 folds to plot average later
    fold_histories = []

    fold_idx = 0
    for _train_idx, _test_idx in kf.split(Data_X):
        x_train, y_train = Data_X[_train_idx], Data_Y[_train_idx]
        x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

        train_data = BasicDataset(x_train, y_train)
        # Increased workers to 4 for better GPU saturation
        train_loader = DataLoader(train_data, batch_size=len(x_train), shuffle=True, num_workers=4, pin_memory=True)
        test_data = BasicDataset(x_test, y_test)
        test_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=False, num_workers=4, pin_memory=True)

        model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
        if ckpt_pretrain is not None:
            model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

        print(f"Fold {fold_idx+1}/5 | {yellow(logf)} | Train: {x_train.shape} | Test: {x_test.shape}")
        
        # Attach history callback
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
        
        # Save history for this fold
        fold_histories.append(history_cb.history['loss'])
        
        # Optional: Save fold model
        # torch.save(model.model.state_dict(), f"{Run}/{subject_id}_fold{fold_idx}.pth")

        # Inference
        pred, real = inference_on_loader(model.model, test_loader)
        Pred.append(pred)
        Real.append(real)
        fold_idx += 1
        
    # --- Post-Training Analysis ---
    Pred, Real = np.concatenate(Pred), np.concatenate(Real)
    
    # 1. Calculate Final Metrics for this Subject
    final_acc = accuracy_score(Real, Pred)
    final_f1 = f1_score(Real, Pred, average='macro')
    
    print(f"\n>>> Subject {subject_id} Completed <<<")
    print(f"    Accuracy: {green(f'{final_acc:.4f}')}")
    print(f"    F1 Score: {green(f'{final_f1:.4f}')}")
    
    # 2. Save Results to CSV
    np.savetxt(logf, np.array([Pred, Real]).T, delimiter=',', fmt='%d')

    # 3. Generate Learning Curve (Average across 5 folds)
    if fold_histories:
        min_len = min([len(h) for h in fold_histories])
        avg_loss = np.mean([h[:min_len] for h in fold_histories], axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_loss, label=f'Subject {subject_id} Avg Loss', color='blue')
        plt.title(f'Learning Curve - Subject {subject_id} (5-Fold Avg)')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot to Results/FAST/sub_X_curve.png
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
        head='EEGNet_Encoder',
        n_classes=5,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    )
    
    X, Y = load_standardized_h5('Processed/BCIC2020Track3.h5')
    
    for fold in range(15):
        if fold not in args.folds:
            continue
        flog = f"{Run}/{fold}-Tune.csv"
        
        # NOTE: Pass 'fold' as subject_id to track plots
        Finetune(config, X[fold], Y[fold], flog, subject_id=fold, max_epochs=200)

    # --- Global Summary ---
    print("\n========================================")
    print("      ALL SUBJECTS EVALUATION COMPLETE    ")
    print("========================================")
    
    # Re-read all generated CSVs to calculate global stats
    all_accuracies = []
    all_f1s = []
    
    for fold in range(15):
        flog = f"{Run}/{fold}-Tune.csv"
        if not os.path.exists(flog):
            continue
            
        data = np.loadtxt(flog, delimiter=',', dtype=int)
        pred, label = data[:, 0], data[:, 1]
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        
        all_accuracies.append(acc)
        all_f1s.append(f1)

    if all_accuracies:
        print(f"Overall Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        print(f"Overall Average F1 Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    else:
        print("No results found.")