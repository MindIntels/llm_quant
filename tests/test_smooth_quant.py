"""Tests for llm_quant.smooth_quant — SmoothQuant."""

import pytest
import torch
import torch.nn as nn
from llm_quant.smooth_quant import (
    SmoothQuant,
    compute_smooth_scales,
    collect_act_max,
    smooth_linear_pair,
)
from llm_quant.core.quantizer import QuantizedLinear


class TestComputeSmoothScales:
    def test_shape(self):
        act_max = torch.rand(64) + 0.1
        weight = torch.randn(32, 64)
        scales = compute_smooth_scales(act_max, weight, alpha=0.5)
        assert scales.shape == (64,)

    def test_positive(self):
        act_max = torch.rand(64) + 0.1
        weight = torch.randn(32, 64)
        scales = compute_smooth_scales(act_max, weight, alpha=0.5)
        assert (scales > 0).all()

    def test_alpha_zero_weight_only(self):
        """alpha=0 → scales = 1/w_max (all difficulty on weight)."""
        act_max = torch.ones(8) * 5.0
        weight = torch.randn(4, 8)
        s0 = compute_smooth_scales(act_max, weight, alpha=0.0)
        s1 = compute_smooth_scales(act_max, weight, alpha=1.0)
        # At alpha=0 scales should be smaller (less protection for act)
        # At alpha=1 scales should be proportional to act_max
        assert not torch.allclose(s0, s1)


class TestCollectActMax:
    def test_single_batch(self):
        acts = [torch.tensor([[1.0, -3.0], [2.0, 1.0]])]
        amax = collect_act_max(acts)
        torch.testing.assert_close(amax, torch.tensor([2.0, 3.0]))

    def test_multi_batch(self):
        acts = [
            torch.tensor([[1.0, -2.0]]),
            torch.tensor([[5.0, 1.0]]),
        ]
        amax = collect_act_max(acts)
        torch.testing.assert_close(amax, torch.tensor([5.0, 2.0]))


class TestSmoothLinearPair:
    def test_preserves_output(self):
        """Smoothing should preserve the mathematical output."""
        torch.manual_seed(42)
        ln = nn.LayerNorm(64)
        linear = nn.Linear(64, 32)
        x = torch.randn(2, 10, 64)

        # Record original output
        with torch.no_grad():
            y_orig = linear(ln(x))

        # Smooth
        act = ln(x)
        act_max = act.reshape(-1, 64).abs().amax(dim=0)
        scales = compute_smooth_scales(act_max, linear.weight, alpha=0.5)
        smooth_linear_pair(ln, linear, scales)

        # After smoothing the output should be the same
        with torch.no_grad():
            y_smooth = linear(ln(x))
        torch.testing.assert_close(y_smooth, y_orig, atol=1e-4, rtol=1e-4)


class TestSmoothQuantPipeline:
    def test_full_pipeline(self):
        torch.manual_seed(42)
        ln = nn.LayerNorm(64)
        linear = nn.Linear(64, 32)
        x = torch.randn(4, 16, 64)

        sq = SmoothQuant(n_bits=8, alpha=0.5)
        acts = [ln(x).detach()]
        sq.calibrate(acts)

        # Smooth
        sq.smooth(ln, linear)

        # Quantize
        q_linear = sq.quantize_linear(linear)
        assert isinstance(q_linear, QuantizedLinear)

        # Should produce correct-shaped output
        y = q_linear(ln(x))
        assert y.shape == (4, 16, 32)

    def test_act_max_stored(self):
        sq = SmoothQuant()
        acts = [torch.randn(2, 32)]
        sq.calibrate(acts)
        assert sq.act_max.shape == (32,)
