"""
PyTorch Lightning training callbacks for FAST
"""

from lightning.pytorch.callbacks import Callback


class HistoryCallback(Callback):
    """Captures train/val loss and metrics at the end of every epoch."""
    
    def __init__(self):
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        acc = trainer.callback_metrics.get('train_acc')
        if loss is not None:
            self.history['loss'].append(loss.item())
        if acc is not None:
            self.history['acc'].append(acc.item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_loss is not None:
            self.history['val_loss'].append(val_loss.item())
        if val_acc is not None:
            self.history['val_acc'].append(val_acc.item())
