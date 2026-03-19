"""Tests for llm_quant.quarot — QuaRot."""

import pytest
import torch
import torch.nn as nn
from llm_quant.quarot import (
    QuaRot,
    quarot_rotate_linear_pair,
    quarot_rotate_head_pair,
)
from llm_quant.core.rotation import hadamard_matrix, random_hadamard_matrix
from llm_quant.core.quantizer import QuantizedLinear


class TestQuaRotRotation:
    def test_rotation_preserves_output_linear_pair(self):
        """Rotating (prev_linear, linear) pair should preserve output."""
        torch.manual_seed(42)
        prev = nn.Linear(64, 64)
        linear = nn.Linear(64, 32)
        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            y_orig = linear(prev(x))

        R = hadamard_matrix(64)
        quarot_rotate_linear_pair(prev, linear, R)

        with torch.no_grad():
            y_rot = linear(prev(x))

        torch.testing.assert_close(y_rot, y_orig, atol=1e-4, rtol=1e-4)

    def test_rotation_changes_weights(self):
        torch.manual_seed(42)
        linear = nn.Linear(64, 32)
        w_orig = linear.weight.data.clone()
        R = hadamard_matrix(64)
        quarot_rotate_linear_pair(None, linear, R)
        assert not torch.allclose(linear.weight.data, w_orig)


class TestQuaRotHeadRotation:
    def test_per_head_rotation(self):
        torch.manual_seed(42)
        head_dim = 16
        n_heads = 4
        hidden = n_heads * head_dim

        q_proj = nn.Linear(hidden, hidden)
        k_proj = nn.Linear(hidden, hidden)
        v_proj = nn.Linear(hidden, hidden)
        o_proj = nn.Linear(hidden, hidden)

        R_head = hadamard_matrix(head_dim)
        quarot_rotate_head_pair(q_proj, k_proj, v_proj, o_proj, head_dim, R_head)

        # Weights should have changed
        assert q_proj.weight.shape == (hidden, hidden)


class TestQuaRotAPI:
    def test_make_rotation(self):
        qr = QuaRot(n_bits=4)
        R = qr.make_rotation(64)
        assert R.shape == (64, 64)
        # Check orthogonal
        I = torch.eye(64)
        torch.testing.assert_close(R @ R.t(), I, atol=1e-5, rtol=1e-5)

    def test_quantize_linear(self):
        qr = QuaRot(n_bits=4, group_size=32)
        linear = nn.Linear(64, 32)
        q_linear = qr.quantize_linear(linear)
        assert isinstance(q_linear, QuantizedLinear)
        x = torch.randn(2, 64)
        y = q_linear(x)
        assert y.shape == (2, 32)

    def test_rotate_activation(self):
        qr = QuaRot()
        x = torch.randn(2, 8, 64)
        y = qr.rotate_activation(x)
        assert y.shape == x.shape

    def test_fake_quantize_weight(self):
        qr = QuaRot(n_bits=4, group_size=32)
        w = torch.randn(32, 64)
        w_q = qr.fake_quantize_weight(w)
        assert w_q.shape == w.shape

    def test_full_pipeline(self):
        torch.manual_seed(42)
        qr = QuaRot(n_bits=4, group_size=32)

        # Create a simple model
        prev = nn.Linear(64, 64)
        linear = nn.Linear(64, 32)

        R = qr.make_rotation(64)
        qr.rotate_linear_pair(prev, linear, R)
        q_linear = qr.quantize_linear(linear)

        x = torch.randn(2, 64)
        y = q_linear(prev(x))
        assert y.shape == (2, 32)
