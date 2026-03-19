"""Tests for llm_quant.kv_cache_quant — KV cache quantization."""

import torch
import pytest
from llm_quant.kv_cache_quant import (
    KVQuantGranularity,
    KVCacheQuantizer,
    PerTokenKVCacheQuantizer,
    PerChannelKVCacheQuantizer,
    PerGroupKVCacheQuantizer,
    QuantizedKVCache,
    kv_cache_memory_ratio,
)


B, H, T, D = 2, 4, 16, 32  # batch, heads, seq_len, head_dim


# ──────────────────────────────────────────────────────────────────────────────
# Per-token quantizer
# ──────────────────────────────────────────────────────────────────────────────

class TestPerTokenQuantizer:
    def test_quantize_shape(self):
        q_er = PerTokenKVCacheQuantizer(n_bits=8)
        x = torch.randn(B, H, T, D)
        q, scale, zp = q_er.quantize(x)
        assert q.shape == x.shape
        assert scale.shape == (B, H, T, 1)
        assert zp.shape == (B, H, T, 1)

    def test_int_range_8bit(self):
        q_er = PerTokenKVCacheQuantizer(n_bits=8)
        x = torch.randn(B, H, T, D)
        q, _, _ = q_er.quantize(x)
        assert q.min() >= -128
        assert q.max() <= 127

    def test_int_range_4bit(self):
        q_er = PerTokenKVCacheQuantizer(n_bits=4)
        x = torch.randn(B, H, T, D)
        q, _, _ = q_er.quantize(x)
        assert q.min() >= -8
        assert q.max() <= 7

    def test_roundtrip_close(self):
        q_er = PerTokenKVCacheQuantizer(n_bits=8)
        x = torch.randn(B, H, T, D)
        q, scale, zp = q_er.quantize(x)
        x_deq = q_er.dequantize(q, scale, zp)
        mse = (x - x_deq).pow(2).mean()
        assert mse < 0.01, f"Per-token 8-bit MSE too high: {mse:.6f}"

    def test_4bit_roundtrip(self):
        q_er = PerTokenKVCacheQuantizer(n_bits=4)
        x = torch.randn(B, H, T, D)
        x_deq = q_er.fake_quantize(x)
        mse = (x - x_deq).pow(2).mean()
        assert mse < 0.1, f"Per-token 4-bit MSE: {mse:.4f}"

    def test_symmetric_zero_zp(self):
        q_er = PerTokenKVCacheQuantizer(n_bits=8, symmetric=True)
        x = torch.randn(B, H, T, D)
        _, _, zp = q_er.quantize(x)
        assert (zp == 0).all()

    def test_asymmetric(self):
        q_er = KVCacheQuantizer(n_bits=8, symmetric=False,
                                granularity=KVQuantGranularity.PER_TOKEN)
        x = torch.randn(B, H, T, D) + 2.0  # positive-biased
        q, scale, zp = q_er.quantize(x)
        x_deq = q_er.dequantize(q, scale, zp)
        mse = (x - x_deq).pow(2).mean()
        assert mse < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Per-channel quantizer
# ──────────────────────────────────────────────────────────────────────────────

class TestPerChannelQuantizer:
    def test_quantize_shape(self):
        q_er = PerChannelKVCacheQuantizer(n_bits=8)
        x = torch.randn(B, H, T, D)
        q, scale, zp = q_er.quantize(x)
        assert q.shape == x.shape
        assert scale.shape == (B, H, 1, D)
        assert zp.shape == (B, H, 1, D)

    def test_roundtrip_close(self):
        q_er = PerChannelKVCacheQuantizer(n_bits=8)
        x = torch.randn(B, H, T, D)
        q, scale, zp = q_er.quantize(x)
        x_deq = q_er.dequantize(q, scale, zp)
        mse = (x - x_deq).pow(2).mean()
        assert mse < 0.01, f"Per-channel 8-bit MSE: {mse:.6f}"

    def test_4bit_works(self):
        q_er = PerChannelKVCacheQuantizer(n_bits=4)
        x = torch.randn(B, H, T, D)
        x_deq = q_er.fake_quantize(x)
        assert x_deq.shape == x.shape

    def test_different_channels_different_scales(self):
        q_er = PerChannelKVCacheQuantizer(n_bits=8)
        x = torch.randn(1, 1, T, D)
        # Make one channel have much larger range
        x[:, :, :, 0] *= 100.0
        _, scale, _ = q_er.quantize(x)
        # scale for channel 0 should be much larger
        assert scale[0, 0, 0, 0] > scale[0, 0, 0, 1] * 10


# ──────────────────────────────────────────────────────────────────────────────
# Per-group quantizer
# ──────────────────────────────────────────────────────────────────────────────

class TestPerGroupQuantizer:
    def test_quantize_shape(self):
        gs = 8
        q_er = PerGroupKVCacheQuantizer(n_bits=8, group_size=gs)
        x = torch.randn(B, H, T, D)
        q, scale, zp = q_er.quantize(x)
        assert q.shape == x.shape
        n_groups = (T + gs - 1) // gs
        assert scale.shape == (B, H, n_groups, D)
        assert zp.shape == (B, H, n_groups, D)

    def test_roundtrip_close(self):
        q_er = PerGroupKVCacheQuantizer(n_bits=8, group_size=8)
        x = torch.randn(B, H, T, D)
        q, scale, zp = q_er.quantize(x)
        x_deq = q_er.dequantize(q, scale, zp)
        mse = (x - x_deq).pow(2).mean()
        assert mse < 0.01, f"Per-group 8-bit MSE: {mse:.6f}"

    def test_non_divisible_seq_len(self):
        """seq_len=13 is not divisible by group_size=8."""
        q_er = PerGroupKVCacheQuantizer(n_bits=8, group_size=8)
        x = torch.randn(B, H, 13, D)
        q, scale, zp = q_er.quantize(x)
        x_deq = q_er.dequantize(q, scale, zp)
        assert x_deq.shape == x.shape

    def test_4bit_group(self):
        q_er = PerGroupKVCacheQuantizer(n_bits=4, group_size=4)
        x = torch.randn(B, H, T, D)
        q, _, _ = q_er.quantize(x)
        assert q.min() >= -8
        assert q.max() <= 7


# ──────────────────────────────────────────────────────────────────────────────
# QuantizedKVCache
# ──────────────────────────────────────────────────────────────────────────────

class TestQuantizedKVCache:
    def test_update_single(self):
        cache = QuantizedKVCache(num_layers=2)
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        fk, fv = cache.update(0, k, v)
        assert fk.shape == (B, H, 1, D)
        assert fv.shape == (B, H, 1, D)

    def test_update_multiple(self):
        cache = QuantizedKVCache(num_layers=2)
        for t in range(5):
            k = torch.randn(B, H, 1, D)
            v = torch.randn(B, H, 1, D)
            fk, fv = cache.update(0, k, v)
        assert fk.shape == (B, H, 5, D)
        assert fv.shape == (B, H, 5, D)

    def test_seq_len_property(self):
        cache = QuantizedKVCache(num_layers=1)
        assert cache.seq_len == 0
        cache.update(0, torch.randn(B, H, 3, D), torch.randn(B, H, 3, D))
        assert cache.seq_len == 3
        cache.update(0, torch.randn(B, H, 2, D), torch.randn(B, H, 2, D))
        assert cache.seq_len == 5

    def test_reset(self):
        cache = QuantizedKVCache(num_layers=2)
        cache.update(0, torch.randn(B, H, 4, D), torch.randn(B, H, 4, D))
        cache.reset()
        assert cache.seq_len == 0
        assert cache.get(0) is None

    def test_get_layer(self):
        cache = QuantizedKVCache(num_layers=2)
        cache.update(0, torch.randn(B, H, 3, D), torch.randn(B, H, 3, D))
        cache.update(1, torch.randn(B, H, 2, D), torch.randn(B, H, 2, D))
        result0 = cache.get(0)
        result1 = cache.get(1)
        assert result0 is not None
        assert result1 is not None
        assert result0[0].shape == (B, H, 3, D)
        assert result1[0].shape == (B, H, 2, D)

    def test_with_per_channel(self):
        q_er = PerChannelKVCacheQuantizer(n_bits=8)
        cache = QuantizedKVCache(num_layers=1, quantizer=q_er)
        k = torch.randn(B, H, 4, D)
        v = torch.randn(B, H, 4, D)
        fk, fv = cache.update(0, k, v)
        assert fk.shape == k.shape
        assert fv.shape == v.shape

    def test_with_per_group(self):
        q_er = PerGroupKVCacheQuantizer(n_bits=4, group_size=4)
        cache = QuantizedKVCache(num_layers=1, quantizer=q_er)
        for _ in range(3):
            k = torch.randn(B, H, 4, D)
            v = torch.randn(B, H, 4, D)
            fk, fv = cache.update(0, k, v)
        assert fk.shape == (B, H, 12, D)

    def test_quantize_quality(self):
        """Cached values should be close to originals (8-bit)."""
        cache = QuantizedKVCache(num_layers=1)
        k_orig = torch.randn(B, H, 8, D)
        v_orig = torch.randn(B, H, 8, D)
        fk, fv = cache.update(0, k_orig, v_orig)
        k_mse = (k_orig - fk).pow(2).mean()
        v_mse = (v_orig - fv).pow(2).mean()
        assert k_mse < 0.01
        assert v_mse < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Memory ratio utility
# ──────────────────────────────────────────────────────────────────────────────

class TestMemoryRatio:
    def test_4bit_less_than_1(self):
        ratio = kv_cache_memory_ratio(n_bits=4)
        assert ratio < 1.0

    def test_8bit_approx_half(self):
        ratio = kv_cache_memory_ratio(n_bits=8, head_dim=128)
        assert 0.4 < ratio < 0.6

    def test_per_channel_lower_overhead(self):
        r_token = kv_cache_memory_ratio(n_bits=4,
                                        granularity=KVQuantGranularity.PER_TOKEN)
        r_channel = kv_cache_memory_ratio(n_bits=4,
                                          granularity=KVQuantGranularity.PER_CHANNEL,
                                          seq_len=2048)
        # Per-channel has lower overhead for long sequences
        assert r_channel < r_token
