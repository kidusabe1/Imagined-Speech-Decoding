"""
PyTorch Lightning Module for FAST training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning as pl

from ..models import FAST


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Create cosine learning rate schedule with optional warmup."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class EEG_Encoder_Module(pl.LightningModule):
    """PyTorch Lightning wrapper for FAST model."""
    
    def __init__(self, config, max_epochs, niter_per_ep):
        super().__init__()
        self.config = config
        self.model = FAST(config)
        self.loss = nn.CrossEntropyLoss()
        self.cosine_lr_list = cosine_scheduler(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=10)
        
        # Training Metrics
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
        self.train_f1 = torchmetrics.F1Score('multiclass', num_classes=config.n_classes, average='macro')
        
        # Validation Metrics
        self.val_acc = torchmetrics.Accuracy('multiclass', num_classes=config.n_classes)
        self.val_f1 = torchmetrics.F1Score('multiclass', num_classes=config.n_classes, average='macro')

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda epoch: self.cosine_lr_list[self.global_step - 1]
        )
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
    """Run inference on a DataLoader and return predictions and labels."""
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
