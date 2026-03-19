"""Tests for llm_quant.core.quantizer — basic quantization primitives."""

import pytest
import torch
import torch.nn as nn
from llm_quant.core.quantizer import (
    ste_round,
    compute_qparams,
    quantize_tensor,
    dequantize_tensor,
    fake_quantize,
    QScheme,
    QuantizedLinear,
)


# ── STE Round ──────────────────────────────────────────────────────────────

class TestSTERound:
    def test_rounds_correctly(self):
        x = torch.tensor([1.3, 2.7, -0.5, 0.0])
        y = ste_round(x)
        expected = torch.tensor([1.0, 3.0, 0.0, 0.0])
        torch.testing.assert_close(y, expected)

    def test_gradient_passes_through(self):
        x = torch.tensor([1.6, 2.3], requires_grad=True)
        y = ste_round(x)
        y.sum().backward()
        assert x.grad is not None
        torch.testing.assert_close(x.grad, torch.ones(2))


# ── Qparams ───────────────────────────────────────────────────────────────

class TestComputeQparams:
    def test_per_tensor_symmetric(self):
        x = torch.randn(4, 8)
        scale, zp = compute_qparams(x, n_bits=8, symmetric=True)
        assert scale.shape == torch.Size([])  or scale.numel() == 1
        assert (zp == 0).all()

    def test_per_channel(self):
        x = torch.randn(16, 32)
        scale, zp = compute_qparams(x, n_bits=8, symmetric=True,
                                     scheme=QScheme.PER_CHANNEL)
        assert scale.shape[0] == 16

    def test_per_group(self):
        x = torch.randn(16, 64)
        scale, zp = compute_qparams(x, n_bits=4, symmetric=True,
                                     scheme=QScheme.PER_GROUP, group_size=32)
        # 64 / 32 = 2 groups
        assert scale.shape == (16, 2, 1)

    def test_asymmetric_per_tensor(self):
        x = torch.randn(4, 8)
        scale, zp = compute_qparams(x, n_bits=8, symmetric=False)
        assert scale.numel() == 1
        assert zp.numel() == 1


# ── Quantize / Dequantize ─────────────────────────────────────────────────

class TestQuantizeDequantize:
    def test_round_trip_low_noise(self):
        """Quantize 8-bit → dequantize should be close to original."""
        x = torch.randn(32, 64)
        scale, zp = compute_qparams(x, n_bits=8, symmetric=True)
        q = quantize_tensor(x, scale, zp, n_bits=8, symmetric=True)
        x_hat = dequantize_tensor(q, scale, zp)
        # 8-bit quantization should be very close
        assert ((x - x_hat).abs().max() < 0.1).item()

    def test_clamps_to_range(self):
        x = torch.tensor([100.0, -100.0])
        scale, zp = compute_qparams(x, n_bits=4, symmetric=True)
        q = quantize_tensor(x, scale, zp, n_bits=4, symmetric=True)
        assert q.max() <= 7
        assert q.min() >= -8


# ── Fake Quantize ─────────────────────────────────────────────────────────

class TestFakeQuantize:
    def test_output_shape(self):
        x = torch.randn(8, 64)
        y = fake_quantize(x, n_bits=8)
        assert y.shape == x.shape

    def test_per_group(self):
        x = torch.randn(16, 128)
        y = fake_quantize(x, n_bits=4, scheme=QScheme.PER_GROUP, group_size=32)
        assert y.shape == x.shape

    def test_gradient_flows(self):
        x = torch.randn(8, 32, requires_grad=True)
        y = fake_quantize(x, n_bits=8)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_8bit_low_error(self):
        x = torch.randn(32, 64)
        y = fake_quantize(x, n_bits=8)
        mse = ((x - y) ** 2).mean()
        assert mse < 0.01  # 8-bit should be very accurate


# ── QuantizedLinear ────────────────────────────────────────────────────────

class TestQuantizedLinear:
    def test_from_linear(self):
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_linear(linear, n_bits=8)
        assert ql.in_features == 64
        assert ql.out_features == 32
        assert ql.weight_int.dtype == torch.int8

    def test_forward_shape(self):
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_linear(linear, n_bits=8)
        x = torch.randn(2, 10, 64)
        y = ql(x)
        assert y.shape == (2, 10, 32)

    def test_8bit_close_to_original(self):
        torch.manual_seed(42)
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_linear(linear, n_bits=8)
        x = torch.randn(4, 64)
        y_orig = linear(x)
        y_quant = ql(x)
        # 8-bit should be quite close
        assert ((y_orig - y_quant).abs().max() < 0.5).item()

    def test_group_quantize(self):
        linear = nn.Linear(128, 64)
        ql = QuantizedLinear.from_linear(linear, n_bits=4, group_size=32)
        x = torch.randn(2, 128)
        y = ql(x)
        assert y.shape == (2, 64)

    def test_no_bias(self):
        linear = nn.Linear(32, 16, bias=False)
        ql = QuantizedLinear.from_linear(linear, n_bits=8)
        assert ql.bias is None
        x = torch.randn(2, 32)
        y = ql(x)
        assert y.shape == (2, 16)
