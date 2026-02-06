import torch
from fast.models import FAST


def test_fast_forward_shape(model_config, dummy_eeg_batch):
    """FAST model produces (batch_size, n_classes) logits."""
    model = FAST(model_config)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_eeg_batch)
    assert logits.shape == (dummy_eeg_batch.shape[0], model_config.n_classes)


def test_fast_forward_modes(model_config, dummy_eeg_batch):
    """All forward modes produce the correct output shape."""
    model = FAST(model_config)
    model.eval()
    batch_size = dummy_eeg_batch.shape[0]
    n_classes = model_config.n_classes

    with torch.no_grad():
        out_default = model(dummy_eeg_batch, forward_mode="default")
        out_head = model(dummy_eeg_batch, forward_mode="train_head")
        out_transformer = model(dummy_eeg_batch, forward_mode="train_transformer")

    assert out_default.shape == (batch_size, n_classes)
    assert out_head.shape == (batch_size, n_classes)
    assert out_transformer.shape == (batch_size, n_classes)


def test_fast_n_tokens(model_config):
    """Token count matches expected value from config."""
    model = FAST(model_config)
    expected = (model_config.seq_len - model_config.window_len) // model_config.slide_step + 1
    assert model.n_tokens == expected
