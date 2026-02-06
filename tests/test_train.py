"""
Comprehensive tests for training module (src/fast/train/).

Covers:
- cosine_scheduler: schedule length, warmup ramp, value ranges, edge cases
- EEG_Encoder_Module: instantiation, training step, validation step, optimizer config
- HistoryCallback: metric recording over epochs
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fast.models.fast import FAST
from fast.data import BasicDataset
from fast.train.trainer import cosine_scheduler, EEG_Encoder_Module
from fast.train.callbacks import HistoryCallback


# ============================================================
# cosine_scheduler tests
# ============================================================

class TestCosineScheduler:
    """Tests for the cosine learning rate schedule."""

    def test_schedule_length(self):
        """Schedule length equals epochs * niter_per_ep."""
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=5)
        assert len(schedule) == 50

    def test_schedule_length_with_warmup(self):
        """Length is correct when warmup is included."""
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=5, warmup_epochs=2)
        assert len(schedule) == 50

    def test_starts_at_base_value_no_warmup(self):
        """Without warmup, schedule starts at base_value."""
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=5, warmup_epochs=0)
        assert schedule[0] == pytest.approx(1.0)

    def test_ends_at_final_value(self):
        """Schedule ends at final_value."""
        schedule = cosine_scheduler(1.0, 0.1, epochs=10, niter_per_ep=5)
        assert schedule[-1] == pytest.approx(0.1, abs=0.01)

    def test_warmup_starts_at_zero(self):
        """With warmup, schedule starts at start_warmup_value (default 0)."""
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=5, warmup_epochs=2)
        assert schedule[0] == pytest.approx(0.0)

    def test_warmup_ramps_to_base(self):
        """At end of warmup, schedule reaches base_value."""
        warmup_epochs = 2
        niter = 5
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=niter, warmup_epochs=warmup_epochs)
        warmup_end_idx = warmup_epochs * niter - 1
        assert schedule[warmup_end_idx] == pytest.approx(1.0, abs=0.05)

    def test_all_values_positive(self):
        """All schedule values are non-negative when base >= final >= 0."""
        schedule = cosine_scheduler(1.0, 0.1, epochs=10, niter_per_ep=5, warmup_epochs=2)
        assert all(v >= 0 for v in schedule)

    def test_monotonic_decay_after_warmup(self):
        """After warmup, schedule generally decreases (cosine decay)."""
        warmup_epochs = 2
        niter = 10
        schedule = cosine_scheduler(1.0, 0.0, epochs=20, niter_per_ep=niter, warmup_epochs=warmup_epochs)
        after_warmup = schedule[warmup_epochs * niter:]
        # First value should be >= last value
        assert after_warmup[0] >= after_warmup[-1]

    def test_single_epoch(self):
        """Works with a single epoch."""
        schedule = cosine_scheduler(1.0, 0.0, epochs=1, niter_per_ep=10)
        assert len(schedule) == 10

    def test_custom_start_warmup_value(self):
        """Custom start_warmup_value is used."""
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, niter_per_ep=5,
                                    warmup_epochs=2, start_warmup_value=0.5)
        assert schedule[0] == pytest.approx(0.5, abs=0.05)


# ============================================================
# EEG_Encoder_Module tests
# ============================================================

class TestEEGEncoderModule:
    """Tests for the Lightning training module."""

    def test_instantiation(self, small_config):
        """Module can be instantiated."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        assert isinstance(module.model, FAST)

    def test_has_loss_function(self, small_config):
        """Module has a CrossEntropyLoss."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        assert isinstance(module.loss, torch.nn.CrossEntropyLoss)

    def test_has_metrics(self, small_config):
        """Module has train and val accuracy/F1 metrics."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        assert hasattr(module, 'train_acc')
        assert hasattr(module, 'train_f1')
        assert hasattr(module, 'val_acc')
        assert hasattr(module, 'val_f1')

    def test_configure_optimizers(self, small_config):
        """configure_optimizers returns optimizer and scheduler."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1
        assert isinstance(optimizers[0], torch.optim.AdamW)

    def test_training_step_returns_loss(self, small_config, dummy_eeg_small):
        """training_step returns a scalar loss."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        labels = torch.randint(0, small_config.n_classes, (dummy_eeg_small.shape[0],))
        batch = (dummy_eeg_small, labels)
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # cross-entropy > 0

    def test_validation_step_returns_loss(self, small_config, dummy_eeg_small):
        """validation_step returns a scalar loss."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        labels = torch.randint(0, small_config.n_classes, (dummy_eeg_small.shape[0],))
        batch = (dummy_eeg_small, labels)
        loss = module.validation_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_cosine_lr_list_length(self, small_config):
        """Cosine LR list has correct length = max_epochs * niter_per_ep."""
        max_epochs = 10
        niter = 5
        module = EEG_Encoder_Module(small_config, max_epochs=max_epochs, niter_per_ep=niter)
        assert len(module.cosine_lr_list) == max_epochs * niter


# ============================================================
# HistoryCallback tests
# ============================================================

class TestHistoryCallback:
    """Tests for the training history callback."""

    def test_initial_state(self):
        """History starts empty."""
        cb = HistoryCallback()
        assert cb.history == {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def test_all_keys_present(self):
        """History has all expected keys."""
        cb = HistoryCallback()
        assert set(cb.history.keys()) == {'loss', 'acc', 'val_loss', 'val_acc'}

    def test_on_train_epoch_end_records(self):
        """on_train_epoch_end appends metrics when available."""
        cb = HistoryCallback()

        # Simulate a trainer with callback_metrics
        class MockTrainer:
            callback_metrics = {'train_loss': torch.tensor(0.5), 'train_acc': torch.tensor(0.8)}

        cb.on_train_epoch_end(MockTrainer(), None)
        assert cb.history['loss'] == [0.5]
        assert cb.history['acc'] == [pytest.approx(0.8)]

    def test_on_validation_epoch_end_records(self):
        """on_validation_epoch_end appends val metrics."""
        cb = HistoryCallback()

        class MockTrainer:
            callback_metrics = {'val_loss': torch.tensor(0.3), 'val_acc': torch.tensor(0.9)}

        cb.on_validation_epoch_end(MockTrainer(), None)
        assert cb.history['val_loss'] == [pytest.approx(0.3)]
        assert cb.history['val_acc'] == [pytest.approx(0.9)]

    def test_missing_metrics_not_recorded(self):
        """If metrics are missing (None), nothing is appended."""
        cb = HistoryCallback()

        class MockTrainer:
            callback_metrics = {}

        cb.on_train_epoch_end(MockTrainer(), None)
        assert cb.history['loss'] == []
        assert cb.history['acc'] == []

    def test_multiple_epochs(self):
        """Multiple calls accumulate metrics."""
        cb = HistoryCallback()

        for epoch in range(3):
            class MockTrainer:
                callback_metrics = {
                    'train_loss': torch.tensor(1.0 / (epoch + 1)),
                    'train_acc': torch.tensor(epoch * 0.3),
                }
            cb.on_train_epoch_end(MockTrainer(), None)

        assert len(cb.history['loss']) == 3
        assert len(cb.history['acc']) == 3
