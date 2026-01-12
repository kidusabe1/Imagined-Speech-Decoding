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
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(8)
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchmetrics
import logging
import h5py
import einops
from sklearn.model_selection import KFold
from transformers import PretrainedConfig
import lightning as pl
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)

from FAST import FAST as Tower
from utils import green, yellow
from BCIC2020Track3_preprocess import Electrodes, Zones

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
        self.loss_fn = nn.CrossEntropyLoss()
        self.cosine_lr_list = cosine_scheduler(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=10)
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
        
        # Track metrics for learning curve
        self.train_losses = []
        self.train_accs = []
        self.epoch_loss = 0.0
        self.epoch_correct = 0
        self.epoch_total = 0

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: self.cosine_lr_list[self.global_step-1])
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        # Track batch metrics
        with torch.no_grad():
            preds = torch.argmax(pred, dim=1)
            correct = (preds == y).sum().item()
            self.epoch_loss += loss.item() * x.size(0)
            self.epoch_correct += correct
            self.epoch_total += x.size(0)
        
        # Log loss for progress bar
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def on_train_epoch_end(self):
        # Calculate epoch metrics
        avg_loss = self.epoch_loss / max(self.epoch_total, 1)
        avg_acc = self.epoch_correct / max(self.epoch_total, 1)
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        
        # Reset for next epoch
        self.epoch_loss = 0.0
        self.epoch_correct = 0
        self.epoch_total = 0


class LearningCurveCallback(pl.Callback):
    """Callback to display live training progress."""
    
    def __init__(self, max_epochs, fold_info=""):
        super().__init__()
        self.max_epochs = max_epochs
        self.fold_info = fold_info
        self.start_time = None
    
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training Started {self.fold_info}")
        print(f"{'='*60}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        elapsed = time.time() - self.start_time
        
        if len(pl_module.train_losses) > 0:
            loss = pl_module.train_losses[-1]
            acc = pl_module.train_accs[-1]
            lr = trainer.optimizers[0].param_groups[0]['lr']
            
            # Calculate ETA
            avg_time_per_epoch = elapsed / epoch
            remaining_epochs = self.max_epochs - epoch
            eta = avg_time_per_epoch * remaining_epochs
            
            # Progress bar
            progress = epoch / self.max_epochs
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f"Epoch {epoch:3d}/{self.max_epochs} |{bar}| "
                  f"Loss: {loss:.4f} | Acc: {acc*100:.1f}% | "
                  f"LR: {lr:.2e} | ETA: {eta:.0f}s")
    
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        print(f"{'='*60}")
        print(f"Training Complete! Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        if len(pl_module.train_losses) > 0:
            print(f"Final Loss: {pl_module.train_losses[-1]:.4f} | "
                  f"Final Acc: {pl_module.train_accs[-1]*100:.1f}%")
        print(f"{'='*60}\n")


def save_learning_curve(train_losses, train_accs, save_path):
    """Save learning curve plot to file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Curve', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy curve
    train_accs_pct = [acc * 100 for acc in train_accs]
    ax2.plot(epochs, train_accs_pct, 'g-', linewidth=2, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training Accuracy Curve', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved to: {save_path}")

def Finetune(config, Data_X, Data_Y, logf, max_epochs=200, ckpt_pretrain=None, gpu=0):
    seed_all(42)
    Pred, Real = [], []
    all_fold_losses = []
    all_fold_accs = []
    
    kf = KFold(n_splits=5, shuffle=False)
    fold_num = 0
    
    for _train_idx, _test_idx in kf.split(Data_X):
        fold_num += 1
        x_train, y_train = Data_X[_train_idx], Data_Y[_train_idx]
        x_test, y_test = Data_X[_test_idx], Data_Y[_test_idx]

        train_data = BasicDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size=len(x_train), shuffle=True, num_workers=0, pin_memory=True)
        test_data = BasicDataset(x_test, y_test)
        test_loader = DataLoader(test_data, batch_size=len(x_test), shuffle=False, num_workers=0, pin_memory=True)

        model = EEG_Encoder_Module(config, max_epochs, len(train_loader))
        if ckpt_pretrain is not None:
            model.model.load_state_dict(torch.load(ckpt_pretrain, weights_only=True))

        print(f"\n{yellow(logf)} | {green(str(ckpt_pretrain))}")
        print(f"Train: {x_train.shape} | Test: {x_test.shape} | Labels: {y_train.shape}")
        
        # Setup callback for live logging
        fold_info = f"[K-Fold {fold_num}/5, Test indices: {_test_idx[0]}-{_test_idx[-1]}]"
        learning_curve_callback = LearningCurveCallback(max_epochs, fold_info)
        
        trainer = pl.Trainer(
            strategy='auto', 
            accelerator='gpu', 
            devices=[gpu], 
            max_epochs=max_epochs, 
            callbacks=[learning_curve_callback], 
            enable_progress_bar=False,  # We use custom logging instead
            enable_checkpointing=False, 
            precision='bf16-mixed', 
            logger=False
        )
        trainer.fit(model, train_dataloaders=train_loader)
        
        # Store metrics for this fold
        all_fold_losses.append(model.train_losses)
        all_fold_accs.append(model.train_accs)

        # Save the underlying model weights for SHAP analysis
        model_save_dir = os.path.dirname(logf)
        torch.save(model.model.state_dict(), f"{model_save_dir}/{_test_idx[0]}_model.pth")
        
        # Save learning curve for this fold
        curve_path = f"{model_save_dir}/{_test_idx[0]}_learning_curve.png"
        save_learning_curve(model.train_losses, model.train_accs, curve_path)

        # Test data is used only once
        pred, real = inference_on_loader(model.model, test_loader)
        test_acc = np.mean(pred == real)
        print(f"K-Fold {fold_num} Test Accuracy: {test_acc*100:.2f}%")
        
        Pred.append(pred)
        Real.append(real)
    
    Pred, Real = np.concatenate(Pred), np.concatenate(Real)
    np.savetxt(logf, np.array([Pred, Real]).T, delimiter=',', fmt='%d')
    
    # Save combined learning curve (average across folds)
    if all_fold_losses:
        avg_losses = np.mean(all_fold_losses, axis=0)
        avg_accs = np.mean(all_fold_accs, axis=0)
        combined_curve_path = logf.replace('.csv', '_learning_curve.png')
        save_learning_curve(avg_losses.tolist(), avg_accs.tolist(), combined_curve_path)
        print(f"\nCombined learning curve saved to: {combined_curve_path}")

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
    for fold in range(15):
        if fold not in args.folds:
            continue
        flog = f"{Run}/{fold}-Tune.csv"
        if os.path.exists(flog):
            print(f"Skip {flog}")
            continue
        Finetune(config, X[fold], Y[fold], flog, max_epochs=200, gpu=args.gpu)

    accuracy = []
    for fold in range(15):
        flog = f"{Run}/{fold}-Tune.csv"
        if not os.path.exists(flog):
            print(f"Skip {flog}")
            continue
        data = np.loadtxt(flog, delimiter=',', dtype=int)
        pred, label = data[:, 0], data[:, 1]
        accuracy.append(np.mean(pred == label))

    print(f"Accuracy: {np.mean(accuracy):3f}, Std: {np.std(accuracy):3f}")