"""Tests for llm_quant.awq — AWQ."""

import pytest
import torch
import torch.nn as nn
from llm_quant.awq import AWQ, compute_saliency, awq_grid_search
from llm_quant.core.quantizer import QuantizedLinear


class TestSaliency:
    def test_shape(self):
        acts = [torch.randn(2, 8, 32) for _ in range(4)]
        sal = compute_saliency(acts)
        assert sal.shape == (32,)

    def test_positive(self):
        acts = [torch.randn(2, 8, 32).abs() + 0.1 for _ in range(4)]
        sal = compute_saliency(acts)
        assert (sal > 0).all()

    def test_channels_with_large_activation(self):
        """Channels with larger activations should have higher saliency."""
        act = torch.zeros(1, 1, 4)
        act[0, 0, 2] = 100.0  # channel 2 is dominant
        sal = compute_saliency([act])
        assert sal[2] > sal[0]
        assert sal[2] > sal[1]


class TestGridSearch:
    def test_returns_scales(self):
        torch.manual_seed(42)
        weight = torch.randn(32, 64)
        acts = [torch.randn(2, 8, 64) for _ in range(4)]
        sal = compute_saliency(acts)
        scales = awq_grid_search(weight, acts, sal, n_bits=4, group_size=32)
        assert scales.shape == (64,)
        assert (scales > 0).all()

    def test_fewer_grid_points(self):
        torch.manual_seed(42)
        weight = torch.randn(16, 32)
        acts = [torch.randn(2, 4, 32) for _ in range(4)]
        sal = compute_saliency(acts)
        scales = awq_grid_search(weight, acts, sal, n_grid=5)
        assert scales.shape == (32,)


class TestAWQPipeline:
    def test_calibrate_and_search(self):
        torch.manual_seed(42)
        awq = AWQ(n_bits=4, group_size=32, n_grid=10)
        acts = [torch.randn(2, 8, 64) for _ in range(4)]
        awq.calibrate(acts)

        weight = torch.randn(32, 64)
        scales = awq.search_scales(weight)
        assert scales.shape == (64,)

    def test_apply_scales_with_layernorm(self):
        torch.manual_seed(42)
        ln = nn.LayerNorm(32)
        linear = nn.Linear(32, 16)
        x = torch.randn(2, 8, 32)

        awq = AWQ(n_bits=4, group_size=16)
        acts = [ln(x).detach()]
        awq.calibrate(acts)
        scales = awq.search_scales(linear.weight)
        awq.apply_scales(ln, linear, scales)

        # Should still produce valid output
        y = linear(ln(x))
        assert y.shape == (2, 8, 16)

    def test_quantize_linear(self):
        awq = AWQ(n_bits=4, group_size=32)
        linear = nn.Linear(64, 32)
        q_linear = awq.quantize_linear(linear)
        assert isinstance(q_linear, QuantizedLinear)

    def test_full_pipeline(self):
        torch.manual_seed(42)
        awq = AWQ(n_bits=4, group_size=16, n_grid=10)
        ln = nn.LayerNorm(32)
        linear = nn.Linear(32, 16)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]

        awq.calibrate([ln(c).detach() for c in calib])
        scales = awq.search_scales(linear.weight)
        awq.apply_scales(ln, linear, scales)
        q_linear = awq.quantize_linear(linear)

        x = torch.randn(2, 32)
        y = q_linear(x)
        assert y.shape == (2, 16)
