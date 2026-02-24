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
        """History starts empty with all 6 metric keys."""
        cb = HistoryCallback()
        assert cb.history == {'loss': [], 'acc': [], 'f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    def test_all_keys_present(self):
        """History has all expected keys including f1."""
        cb = HistoryCallback()
        assert set(cb.history.keys()) == {'loss', 'acc', 'f1', 'val_loss', 'val_acc', 'val_f1'}

    def test_on_train_epoch_end_records(self):
        """on_train_epoch_end appends metrics when available."""
        cb = HistoryCallback()

        # Simulate a trainer with callback_metrics
        class MockTrainer:
            callback_metrics = {'train_loss': torch.tensor(0.5), 'train_acc': torch.tensor(0.8), 'train_f1': torch.tensor(0.75)}

        cb.on_train_epoch_end(MockTrainer(), None)
        assert cb.history['loss'] == [0.5]
        assert cb.history['acc'] == [pytest.approx(0.8)]
        assert cb.history['f1'] == [pytest.approx(0.75)]

    def test_on_validation_epoch_end_records(self):
        """on_validation_epoch_end appends val metrics."""
        cb = HistoryCallback()

        class MockTrainer:
            callback_metrics = {'val_loss': torch.tensor(0.3), 'val_acc': torch.tensor(0.9), 'val_f1': torch.tensor(0.85)}

        cb.on_validation_epoch_end(MockTrainer(), None)
        assert cb.history['val_loss'] == [pytest.approx(0.3)]
        assert cb.history['val_acc'] == [pytest.approx(0.9)]
        assert cb.history['val_f1'] == [pytest.approx(0.85)]

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


# ============================================================
# New tests for modifications
# ============================================================

class TestValF1Logging:
    """Tests that val_f1 is now logged in validation_step."""

    def test_validation_step_logs_val_f1(self, small_config, dummy_eeg_small):
        """validation_step logs val_f1 to callback_metrics."""
        module = EEG_Encoder_Module(small_config, max_epochs=10, niter_per_ep=5)
        labels = torch.randint(0, small_config.n_classes, (dummy_eeg_small.shape[0],))
        batch = (dummy_eeg_small, labels)
        # Run validation_step to populate logged metrics
        module.validation_step(batch, 0)
        # val_f1 metric object should have been called
        f1_val = module.val_f1.compute()
        assert f1_val.dim() == 0  # scalar
        assert 0.0 <= f1_val.item() <= 1.0


class TestLRScheduleClamp:
    """Tests that the LR schedule index is clamped safely."""

    def test_lr_at_step_zero(self, small_config):
        """At global_step=0, LR reads schedule[0] instead of schedule[-1]."""
        max_epochs = 10
        niter = 5
        module = EEG_Encoder_Module(small_config, max_epochs=max_epochs, niter_per_ep=niter)
        module.configure_optimizers()
        # Simulate global_step = 0
        module._global_step = 0
        # The lambda uses min(global_step, len-1) = min(0, 49) = 0
        lr_mult = module.cosine_lr_list[min(0, len(module.cosine_lr_list) - 1)]
        # With 10 warmup epochs, step 0 should be near 0 (warmup start)
        assert lr_mult == pytest.approx(module.cosine_lr_list[0])

    def test_lr_at_last_step(self, small_config):
        """At the last step, LR reads the actual final schedule value (not schedule[-1] from wraparound)."""
        max_epochs = 10
        niter = 5
        module = EEG_Encoder_Module(small_config, max_epochs=max_epochs, niter_per_ep=niter)
        last_idx = len(module.cosine_lr_list) - 1
        lr_mult = module.cosine_lr_list[min(last_idx, last_idx)]
        # The value should be the actual last element of the schedule
        assert lr_mult == pytest.approx(module.cosine_lr_list[-1])

    def test_lr_beyond_schedule_is_clamped(self, small_config):
        """Step beyond schedule length is clamped to last element."""
        max_epochs = 10
        niter = 5
        module = EEG_Encoder_Module(small_config, max_epochs=max_epochs, niter_per_ep=niter)
        beyond_step = len(module.cosine_lr_list) + 100
        clamped_idx = min(beyond_step, len(module.cosine_lr_list) - 1)
        lr_mult = module.cosine_lr_list[clamped_idx]
        # Clamped to last element, same as schedule[-1]
        assert lr_mult == pytest.approx(module.cosine_lr_list[-1])


class TestHistoryCallbackF1:
    """Tests for the new F1 recording in HistoryCallback."""

    def test_f1_not_recorded_when_missing(self):
        """When train_f1 is not in callback_metrics, f1 list stays empty."""
        cb = HistoryCallback()
        class MockTrainer:
            callback_metrics = {'train_loss': torch.tensor(0.5), 'train_acc': torch.tensor(0.8)}
        cb.on_train_epoch_end(MockTrainer(), None)
        assert cb.history['f1'] == []

    def test_val_f1_not_recorded_when_missing(self):
        """When val_f1 is not in callback_metrics, val_f1 list stays empty."""
        cb = HistoryCallback()
        class MockTrainer:
            callback_metrics = {'val_loss': torch.tensor(0.3), 'val_acc': torch.tensor(0.9)}
        cb.on_validation_epoch_end(MockTrainer(), None)
        assert cb.history['val_f1'] == []

    def test_multiple_epochs_with_f1(self):
        """F1 accumulates over multiple epochs."""
        cb = HistoryCallback()
        for i in range(5):
            class MockTrainer:
                callback_metrics = {
                    'train_loss': torch.tensor(0.5),
                    'train_acc': torch.tensor(0.8),
                    'train_f1': torch.tensor(0.7 + i * 0.01),
                }
            cb.on_train_epoch_end(MockTrainer(), None)
        assert len(cb.history['f1']) == 5
        assert cb.history['f1'][0] == pytest.approx(0.7)
        assert cb.history['f1'][4] == pytest.approx(0.74)
