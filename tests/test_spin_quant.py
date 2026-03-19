"""Tests for llm_quant.spin_quant — SpinQuant."""

import pytest
import torch
import torch.nn as nn
from llm_quant.spin_quant import SpinQuant, SpinQuantRotation
from llm_quant.core.quantizer import QuantizedLinear


class TestSpinQuantRotation:
    def test_forward_shape(self):
        rot = SpinQuantRotation(32, n_bits=4, group_size=16)
        weight = torch.randn(16, 32)
        W_q, R = rot(weight)
        assert W_q.shape == weight.shape
        assert R.shape == (32, 32)

    def test_rotation_is_orthogonal(self):
        rot = SpinQuantRotation(16)
        rot.rotation.params.data.normal_(std=0.1)
        _, R = rot(torch.randn(8, 16))
        I = torch.eye(16)
        torch.testing.assert_close(R @ R.t(), I, atol=1e-5, rtol=1e-5)

    def test_loss_with_data(self):
        rot = SpinQuantRotation(32, n_bits=4, group_size=16)
        weight = torch.randn(16, 32)
        x = torch.randn(2, 8, 32)
        W_q, R, loss = rot(weight, x)
        assert loss.item() >= 0
        # Loss should be differentiable
        loss.backward()
        assert rot.rotation.params.grad is not None


class TestSpinQuantOptimise:
    def test_optimise_reduces_error(self):
        torch.manual_seed(42)
        sq = SpinQuant(n_bits=4, group_size=16, lr=1e-2)
        weight = torch.randn(16, 32)
        calib = [torch.randn(2, 8, 32) for _ in range(4)]

        R = sq.optimise_rotation(weight, calib, steps=50)
        assert R.shape == (32, 32)
        # R should be orthogonal
        I = torch.eye(32)
        torch.testing.assert_close(R @ R.t(), I, atol=1e-4, rtol=1e-4)

    def test_full_pipeline(self):
        torch.manual_seed(42)
        sq = SpinQuant(n_bits=4, group_size=16)
        linear = nn.Linear(32, 16)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]

        R = sq.optimise_rotation(linear.weight.data, calib, steps=20)
        sq.apply_rotation(linear, R)
        q_linear = sq.quantize_linear(linear)

        assert isinstance(q_linear, QuantizedLinear)
        x = torch.randn(2, 32)
        y = q_linear(x)
        assert y.shape == (2, 16)
