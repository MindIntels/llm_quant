"""Tests for llm_quant.spqr — SpQR."""

import pytest
import torch
import torch.nn as nn
from llm_quant.spqr import (
    SpQR,
    SpQRLinear,
    compute_weight_sensitivity,
    detect_outliers,
)


class TestWeightSensitivity:
    def test_shape(self):
        weight = torch.randn(32, 64)
        acts = [torch.randn(2, 8, 64) for _ in range(4)]
        sens = compute_weight_sensitivity(weight, acts, n_bits=4, group_size=32)
        assert sens.shape == (32, 64)

    def test_non_negative(self):
        weight = torch.randn(16, 32)
        acts = [torch.randn(2, 4, 32) for _ in range(4)]
        sens = compute_weight_sensitivity(weight, acts)
        assert (sens >= 0).all()


class TestDetectOutliers:
    def test_fraction(self):
        sens = torch.randn(100, 100).abs()
        mask = detect_outliers(sens, outlier_fraction=0.01)
        # ~1% should be True
        actual_frac = mask.float().mean().item()
        assert 0.005 < actual_frac < 0.03  # loose tolerance

    def test_higher_fraction(self):
        sens = torch.randn(100, 100).abs()
        mask = detect_outliers(sens, outlier_fraction=0.1)
        actual_frac = mask.float().mean().item()
        assert 0.05 < actual_frac < 0.15

    def test_outliers_are_high_sensitivity(self):
        sens = torch.zeros(10, 10)
        sens[0, 0] = 1000.0  # obvious outlier
        mask = detect_outliers(sens, outlier_fraction=0.01)
        assert mask[0, 0].item() is True


class TestSpQRLinear:
    def test_pack_and_forward(self):
        weight = torch.randn(32, 64)
        mask = torch.zeros(32, 64, dtype=torch.bool)
        mask[0, 0] = True
        mask[5, 10] = True

        layer = SpQRLinear(64, 32, n_bits=4, group_size=32)
        layer.pack(weight, mask)

        x = torch.randn(2, 64)
        y = layer(x)
        assert y.shape == (2, 32)

    def test_sparsity(self):
        mask = torch.zeros(32, 64, dtype=torch.bool)
        mask[:, :3] = True  # 3/64 ≈ 4.7%

        layer = SpQRLinear(64, 32)
        layer.outlier_mask.copy_(mask)
        assert 0.04 < layer.sparsity < 0.05

    def test_outlier_values_restored(self):
        """Outlier positions should return values close to original."""
        torch.manual_seed(42)
        weight = torch.randn(16, 32)
        mask = torch.zeros(16, 32, dtype=torch.bool)
        mask[0, 0] = True
        mask[1, 1] = True

        layer = SpQRLinear(32, 16, n_bits=8, group_size=0)
        layer.pack(weight, mask)

        # Reconstruct full weight
        w_recon = layer.weight_int.float() * layer.scale
        w_recon[layer.outlier_mask] = layer.outlier_vals

        # Outlier positions should be exact
        assert torch.allclose(w_recon[0, 0], weight[0, 0])
        assert torch.allclose(w_recon[1, 1], weight[1, 1])


class TestSpQRPipeline:
    def test_full_pipeline(self):
        torch.manual_seed(42)
        spqr = SpQR(n_bits=4, group_size=32, outlier_fraction=0.01)

        linear = nn.Linear(64, 32)
        calib = [torch.randn(2, 8, 64) for _ in range(4)]
        spqr.calibrate(calib)

        spqr_linear = spqr.quantize_linear(linear)
        assert isinstance(spqr_linear, SpQRLinear)

        x = torch.randn(2, 64)
        y = spqr_linear(x)
        assert y.shape == (2, 32)

    def test_sensitivity_computed(self):
        spqr = SpQR(n_bits=4)
        weight = torch.randn(16, 32)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]
        spqr.calibrate(calib)
        sens = spqr.compute_sensitivity(weight)
        assert sens.shape == (16, 32)
