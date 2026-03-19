"""Tests for llm_quant.core.rotation — Hadamard, Cayley, transforms."""

import math
import pytest
import torch
from llm_quant.core.rotation import (
    hadamard_matrix,
    random_hadamard_matrix,
    random_orthogonal,
    cayley_transform,
    CayleyRotation,
    fast_hadamard_transform,
    rotate_weight_right,
    rotate_weight_left,
    apply_rotation_to_linear,
)
import torch.nn as nn


class TestHadamard:
    def test_is_orthogonal(self):
        H = hadamard_matrix(16)
        I = torch.eye(16)
        torch.testing.assert_close(H @ H.t(), I, atol=1e-5, rtol=1e-5)

    def test_sizes(self):
        for n in [2, 4, 8, 32, 64]:
            H = hadamard_matrix(n)
            assert H.shape == (n, n)

    def test_not_power_of_2_raises(self):
        with pytest.raises(AssertionError):
            hadamard_matrix(3)

    def test_values_bounded(self):
        H = hadamard_matrix(8)
        expected_val = 1.0 / math.sqrt(8)
        assert torch.allclose(H.abs(), torch.full_like(H, expected_val), atol=1e-6)


class TestRandomHadamard:
    def test_is_orthogonal(self):
        H = random_hadamard_matrix(32)
        I = torch.eye(32)
        torch.testing.assert_close(H @ H.t(), I, atol=1e-5, rtol=1e-5)

    def test_different_from_deterministic(self):
        H1 = hadamard_matrix(16)
        g = torch.Generator().manual_seed(123)
        H2 = random_hadamard_matrix(16, generator=g)
        # Should differ (due to random signs) but both orthogonal
        assert not torch.allclose(H1, H2)


class TestRandomOrthogonal:
    def test_is_orthogonal(self):
        Q = random_orthogonal(32)
        I = torch.eye(32)
        torch.testing.assert_close(Q @ Q.t(), I, atol=1e-5, rtol=1e-5)


class TestCayleyTransform:
    def test_zero_gives_identity(self):
        A = torch.zeros(8, 8)
        Q = cayley_transform(A)
        torch.testing.assert_close(Q, torch.eye(8), atol=1e-6, rtol=1e-6)

    def test_skew_gives_orthogonal(self):
        A = torch.randn(8, 8)
        A = A - A.t()  # make skew-symmetric
        Q = cayley_transform(A)
        I = torch.eye(8)
        torch.testing.assert_close(Q @ Q.t(), I, atol=1e-5, rtol=1e-5)


class TestCayleyRotation:
    def test_init_identity(self):
        cr = CayleyRotation(16)
        Q = cr()
        I = torch.eye(16)
        torch.testing.assert_close(Q, I, atol=1e-6, rtol=1e-6)

    def test_output_orthogonal(self):
        cr = CayleyRotation(16)
        # Randomise parameters
        cr.params.data.normal_()
        Q = cr()
        I = torch.eye(16)
        torch.testing.assert_close(Q @ Q.t(), I, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        cr = CayleyRotation(8)
        cr.params.data.normal_(std=0.1)
        Q = cr()
        loss = Q.sum()
        loss.backward()
        assert cr.params.grad is not None


class TestFastHadamard:
    def test_matches_matmul(self):
        x = torch.randn(4, 16)
        H = hadamard_matrix(16)
        y_matmul = x @ H.t()
        y_fast = fast_hadamard_transform(x, normalize=True)
        torch.testing.assert_close(y_fast, y_matmul, atol=1e-5, rtol=1e-5)

    def test_3d_input(self):
        x = torch.randn(2, 8, 32)
        y = fast_hadamard_transform(x)
        assert y.shape == (2, 8, 32)


class TestRotateWeight:
    def test_right_rotation(self):
        W = torch.randn(16, 32)
        R = random_orthogonal(32)
        W_rot = rotate_weight_right(W, R)
        expected = W @ R
        torch.testing.assert_close(W_rot, expected)

    def test_left_rotation(self):
        W = torch.randn(16, 32)
        R = random_orthogonal(16)
        W_rot = rotate_weight_left(W, R)
        expected = R.t() @ W
        torch.testing.assert_close(W_rot, expected)

    def test_apply_to_linear(self):
        linear = nn.Linear(32, 16)
        R = random_orthogonal(32)
        W_orig = linear.weight.data.clone()
        apply_rotation_to_linear(linear, R, side="right")
        expected = W_orig @ R
        torch.testing.assert_close(linear.weight.data, expected)
