"""
PyTorch Lightning training callbacks for FAST
"""

from lightning.pytorch.callbacks import Callback


class HistoryCallback(Callback):
    """Captures train/val loss and metrics at the end of every epoch."""
    
    def __init__(self):
        self.history = {'loss': [], 'acc': [], 'f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        acc = trainer.callback_metrics.get('train_acc')
        f1 = trainer.callback_metrics.get('train_f1')
        if loss is not None:
            self.history['loss'].append(loss.item())
        if acc is not None:
            self.history['acc'].append(acc.item())
        if f1 is not None:
            self.history['f1'].append(f1.item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        val_f1 = trainer.callback_metrics.get('val_f1')
        if val_loss is not None:
            self.history['val_loss'].append(val_loss.item())
        if val_acc is not None:
            self.history['val_acc'].append(val_acc.item())
        if val_f1 is not None:
            self.history['val_f1'].append(val_f1.item())
