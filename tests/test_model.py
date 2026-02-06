"""
Comprehensive tests for FAST model architecture (src/fast/models/fast.py).

Covers:
- FAST end-to-end forward pass (all 3 forward modes)
- Token count calculation from config parameters
- Head module: zone-based electrode indexing and per-zone encoding
- Conv4Layers head architecture
- HeadConv_Paper_Version head architecture
- CVBlock head architecture
- EEGNet_Encoder head architecture
- AttentionBlock (transformer layer)
- Sliding window (unfold) behavior
- batched_forward_head for memory-efficient inference
- Single-sample batch (batch_size=1)
- Gradient flow through the model
- Invalid forward_mode raises NotImplementedError
- Model with different config parameters (small config)
"""

import pytest
import torch
from transformers import PretrainedConfig

from fast.models.fast import (
    FAST, AttentionBlock, CVBlock, Conv4Layers,
    EEGNet_Encoder, HeadConv_Paper_Version, Head,
)
from fast.data.preprocess import Electrodes, Zones


# ============================================================
# FAST full model tests
# ============================================================

class TestFASTForward:
    """Tests for FAST model forward pass."""

    def test_default_forward_shape(self, model_config, dummy_eeg_batch):
        """Default forward produces (B, n_classes) logits."""
        model = FAST(model_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_batch)
        assert logits.shape == (4, model_config.n_classes)

    def test_train_head_mode_shape(self, model_config, dummy_eeg_batch):
        """train_head mode produces (B, n_classes) logits."""
        model = FAST(model_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_batch, forward_mode='train_head')
        assert logits.shape == (4, model_config.n_classes)

    def test_train_transformer_mode_shape(self, model_config, dummy_eeg_batch):
        """train_transformer mode produces (B, n_classes) logits."""
        model = FAST(model_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_batch, forward_mode='train_transformer')
        assert logits.shape == (4, model_config.n_classes)

    def test_invalid_forward_mode_raises(self, model_config, dummy_eeg_batch):
        """Unknown forward_mode raises NotImplementedError."""
        model = FAST(model_config)
        with pytest.raises(NotImplementedError):
            model(dummy_eeg_batch, forward_mode='nonexistent')

    def test_single_sample_batch(self, model_config):
        """Model works with batch_size=1."""
        model = FAST(model_config)
        model.eval()
        x = torch.randn(1, len(Electrodes), model_config.seq_len)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, model_config.n_classes)

    def test_small_config_forward(self, small_config, dummy_eeg_small):
        """Model works with a minimal config (fewer zones, layers)."""
        model = FAST(small_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_small)
        assert logits.shape == (2, small_config.n_classes)

    def test_output_dtype_float32(self, model_config, dummy_eeg_batch):
        """Logits are float32."""
        model = FAST(model_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_batch)
        assert logits.dtype == torch.float32

    def test_output_contains_no_nan(self, model_config, dummy_eeg_batch):
        """Output should not contain NaN values."""
        model = FAST(model_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_batch)
        assert not torch.isnan(logits).any()

    def test_output_contains_no_inf(self, model_config, dummy_eeg_batch):
        """Output should not contain Inf values."""
        model = FAST(model_config)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_eeg_batch)
        assert not torch.isinf(logits).any()

    def test_model_name_attribute(self, model_config):
        """Model has name='FAST' class attribute."""
        model = FAST(model_config)
        assert model.name == 'FAST'


class TestFASTTokens:
    """Tests for token count calculation."""

    def test_n_tokens_default_config(self, model_config):
        """n_tokens = (800 - 250) / 125 + 1 = 5."""
        model = FAST(model_config)
        assert model.n_tokens == 5

    def test_n_tokens_small_config(self, small_config):
        """n_tokens = (500 - 250) / 125 + 1 = 3."""
        model = FAST(small_config)
        assert model.n_tokens == 3

    def test_pos_embedding_matches_tokens(self, model_config):
        """Positional embedding has n_tokens + 1 positions (for CLS)."""
        model = FAST(model_config)
        assert model.pos_embedding.shape[1] == model.n_tokens + 1
        assert model.pos_embedding.shape[2] == model_config.dim_token

    def test_cls_token_shape(self, model_config):
        """CLS token has shape (1, 1, dim_token)."""
        model = FAST(model_config)
        assert model.cls_token.shape == (1, 1, model_config.dim_token)


class TestFASTGradients:
    """Tests for gradient flow."""

    def test_gradients_flow_default_mode(self, small_config, dummy_eeg_small):
        """All parameters receive gradients in default mode."""
        model = FAST(small_config)
        model.train()
        logits = model(dummy_eeg_small)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_train_transformer_freezes_head(self, small_config, dummy_eeg_small):
        """In train_transformer mode, head forward runs under no_grad."""
        model = FAST(small_config)
        model.train()
        logits = model(dummy_eeg_small, forward_mode='train_transformer')
        loss = logits.sum()
        loss.backward()
        # Head params should have no gradient (torch.no_grad in forward)
        for name, param in model.head.named_parameters():
            assert param.grad is None, f"Head param {name} should not have grad in train_transformer mode"

    def test_train_head_mode_has_gradients(self, small_config, dummy_eeg_small):
        """In train_head mode, head parameters DO receive gradients."""
        model = FAST(small_config)
        model.train()
        logits = model(dummy_eeg_small, forward_mode='train_head')
        loss = logits.sum()
        loss.backward()
        head_has_grad = any(p.grad is not None for p in model.head.parameters())
        assert head_has_grad, "Head should have gradients in train_head mode"


class TestFASTForwardHead:
    """Tests for forward_head and batched_forward_head."""

    def test_forward_head_output_shape(self, small_config, dummy_eeg_small):
        """forward_head returns (B, N, Z, F) features."""
        model = FAST(small_config)
        model.eval()
        with torch.no_grad():
            features = model.forward_head(dummy_eeg_small)
        B = dummy_eeg_small.shape[0]
        N = model.n_tokens
        Z = len(small_config.zone_dict)
        F = small_config.dim_cnn
        assert features.shape == (B, N, Z, F)

    def test_step_override_changes_token_count(self, small_config, dummy_eeg_small):
        """Overriding slide_step produces different number of tokens."""
        model = FAST(small_config)
        model.eval()
        with torch.no_grad():
            feat_default = model.forward_head(dummy_eeg_small)
            # Smaller step -> more tokens
            feat_fine = model.forward_head(dummy_eeg_small, step_override=50)
        assert feat_fine.shape[1] > feat_default.shape[1]

    def test_batched_forward_head_matches_full(self, small_config, dummy_eeg_small):
        """batched_forward_head gives same result as forward_head."""
        model = FAST(small_config)
        model.eval()
        with torch.no_grad():
            full = model.forward_head(dummy_eeg_small)
            batched = model.batched_forward_head(
                dummy_eeg_small, step=small_config.slide_step, batch_size=1
            )
        torch.testing.assert_close(full, batched)


# ============================================================
# Individual component tests
# ============================================================

class TestAttentionBlock:
    """Tests for the transformer attention block."""

    def test_output_shape_preserved(self):
        """AttentionBlock preserves input shape."""
        block = AttentionBlock(embed_dim=32, hidden_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output differs from input (not identity) but has same shape."""
        block = AttentionBlock(embed_dim=16, hidden_dim=32, num_heads=2, dropout=0.0)
        x = torch.randn(1, 5, 16)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_single_token_sequence(self):
        """Works with sequence length of 1."""
        block = AttentionBlock(embed_dim=16, hidden_dim=32, num_heads=2)
        x = torch.randn(2, 1, 16)
        out = block(x)
        assert out.shape == (2, 1, 16)

    def test_dropout_zero_deterministic(self):
        """With dropout=0, output is deterministic."""
        block = AttentionBlock(embed_dim=16, hidden_dim=32, num_heads=2, dropout=0.0)
        block.eval()
        x = torch.randn(1, 3, 16)
        out1 = block(x)
        out2 = block(x)
        torch.testing.assert_close(out1, out2)


class TestConv4Layers:
    """Tests for Conv4Layers head."""

    def test_output_shape(self):
        """Conv4Layers produces (B, dim) output."""
        head = Conv4Layers(channels=6, dim=32)
        x = torch.randn(4, 6, 250)
        out = head(x)
        assert out.shape == (4, 32)

    def test_single_channel(self):
        """Works with single channel."""
        head = Conv4Layers(channels=1, dim=16)
        x = torch.randn(2, 1, 250)
        out = head(x)
        assert out.shape == (2, 16)

    def test_varying_time_lengths(self):
        """Output dim stays constant regardless of temporal length."""
        head = Conv4Layers(channels=4, dim=32)
        for T in [100, 250, 500]:
            x = torch.randn(2, 4, T)
            out = head(x)
            assert out.shape == (2, 32), f"Failed for T={T}"

    def test_different_dim_values(self):
        """Works with different output dimensions."""
        for dim in [8, 16, 32, 64]:
            head = Conv4Layers(channels=4, dim=dim)
            x = torch.randn(2, 4, 250)
            out = head(x)
            assert out.shape == (2, dim)


class TestHeadConvPaperVersion:
    """Tests for HeadConv_Paper_Version head."""

    def test_output_shape(self):
        """Produces (B, feature_dim) output."""
        head = HeadConv_Paper_Version(in_channels=6, feature_dim=32)
        x = torch.randn(4, 6, 250)
        out = head(x)
        assert out.shape == (4, 32)

    def test_different_feature_dims(self):
        """Works with various feature_dim values."""
        for dim in [16, 32, 64]:
            head = HeadConv_Paper_Version(in_channels=4, feature_dim=dim)
            x = torch.randn(2, 4, 250)
            out = head(x)
            assert out.shape == (2, dim)


class TestCVBlock:
    """Tests for CVBlock (EEGNet-style) head."""

    def test_output_shape_3d_input(self):
        """CVBlock handles 3D input (B, C, T) by auto-unsqueezing."""
        block = CVBlock(n_channels=6, dim_token=32)
        x = torch.randn(4, 6, 250)
        out = block(x)
        assert out.shape == (4, 32)

    def test_output_shape_4d_input(self):
        """CVBlock handles pre-unsqueezed 4D input (B, 1, C, T)."""
        block = CVBlock(n_channels=6, dim_token=32)
        x = torch.randn(4, 1, 6, 250)
        out = block(x)
        assert out.shape == (4, 32)

    def test_3d_and_4d_give_same_result(self):
        """3D and 4D input produce identical output."""
        block = CVBlock(n_channels=4, dim_token=16)
        block.eval()
        x3d = torch.randn(2, 4, 250)
        x4d = x3d.unsqueeze(1)
        with torch.no_grad():
            out3d = block(x3d)
            out4d = block(x4d)
        torch.testing.assert_close(out3d, out4d)


class TestEEGNetEncoder:
    """Tests for EEGNet_Encoder head."""

    def test_output_shape(self):
        """Produces (B, feature_dim) output."""
        enc = EEGNet_Encoder(in_channels=6, feature_dim=32)
        x = torch.randn(4, 6, 250)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_different_kernel_lengths(self):
        """Works with different kernel_length values."""
        for kl in [32, 64, 128]:
            enc = EEGNet_Encoder(in_channels=4, feature_dim=16, kernel_length=kl)
            x = torch.randn(2, 4, 250)
            out = enc(x)
            assert out.shape == (2, 16)

    def test_adaptive_pool_handles_varying_lengths(self):
        """AdaptiveAvgPool2d handles different temporal lengths."""
        enc = EEGNet_Encoder(in_channels=4, feature_dim=16)
        for T in [200, 250, 400]:
            x = torch.randn(2, 4, T)
            out = enc(x)
            assert out.shape == (2, 16)


class TestHead:
    """Tests for the zone-based Head module."""

    def test_output_shape(self):
        """Head produces (B, n_zones, feature_dim) output."""
        electrodes = ['A', 'B', 'C', 'D', 'E', 'F']
        zones = {'Z1': ['A', 'B', 'C'], 'Z2': ['D', 'E', 'F']}
        head = Head('Conv4Layers', electrodes, zones, feature_dim=32)
        x = torch.randn(4, 6, 250)
        out = head(x)
        assert out.shape == (4, 2, 32)

    def test_electrode_indexing(self):
        """Electrode indices correctly map channel names to positions."""
        electrodes = ['A', 'B', 'C', 'D']
        zones = {'Z1': ['B', 'D']}
        head = Head('Conv4Layers', electrodes, zones, feature_dim=16)
        assert head.index_dict['Z1'].tolist() == [1, 3]

    def test_zone_with_single_channel(self):
        """Zone with single channel works."""
        electrodes = ['A', 'B', 'C']
        zones = {'Z1': ['A'], 'Z2': ['B', 'C']}
        head = Head('Conv4Layers', electrodes, zones, feature_dim=16)
        x = torch.randn(2, 3, 250)
        out = head(x)
        assert out.shape == (2, 2, 16)

    def test_many_zones(self):
        """Head works with many zones (like the real 8-zone config)."""
        head = Head('Conv4Layers', Electrodes, Zones, feature_dim=32)
        x = torch.randn(2, len(Electrodes), 250)
        out = head(x)
        assert out.shape == (2, len(Zones), 32)

    def test_encoder_per_zone_has_correct_channels(self):
        """Each zone encoder is constructed with the correct number of input channels."""
        electrodes = ['A', 'B', 'C', 'D', 'E']
        zones = {'Z1': ['A', 'B'], 'Z2': ['C', 'D', 'E']}
        head = Head('Conv4Layers', electrodes, zones, feature_dim=16)
        # Conv4Layers.cnn2 kernel has shape (dim, n_channels, 1)
        assert head.encoders['Z1'].cnn2.weight.shape[2] == 2
        assert head.encoders['Z2'].cnn2.weight.shape[2] == 3
