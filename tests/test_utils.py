"""Tests for llm_quant.core.utils — observers, metrics, helpers."""

import pytest
import torch
import torch.nn as nn
from llm_quant.core.utils import (
    MinMaxObserver,
    ActivationCollector,
    quantization_mse,
    quantization_snr,
    kurtosis,
    make_dummy_linear,
    make_calibration_data,
)


class TestMinMaxObserver:
    def test_single_update(self):
        obs = MinMaxObserver()
        x = torch.tensor([[-1.0, 2.0], [0.5, -3.0]])
        obs.update(x)
        mn, mx = obs.get_range()
        assert mn.item() == pytest.approx(-3.0)
        assert mx.item() == pytest.approx(2.0)

    def test_multiple_updates(self):
        obs = MinMaxObserver()
        obs.update(torch.tensor([1.0, 5.0]))
        obs.update(torch.tensor([-2.0, 3.0]))
        mn, mx = obs.get_range()
        assert mn.item() == pytest.approx(-2.0)
        assert mx.item() == pytest.approx(5.0)

    def test_per_channel(self):
        obs = MinMaxObserver(per_channel=True, channel_dim=-1)
        x = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 4.0, -5.0]])
        obs.update(x)
        mn, mx = obs.get_range()
        assert mn.shape == (3,)
        torch.testing.assert_close(mn, torch.tensor([-1.0, -2.0, -5.0]))
        torch.testing.assert_close(mx, torch.tensor([1.0, 4.0, 3.0]))

    def test_reset(self):
        obs = MinMaxObserver()
        obs.update(torch.tensor([1.0]))
        obs.reset()
        with pytest.raises(AssertionError):
            obs.get_range()


class TestActivationCollector:
    def test_collects_inputs(self):
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        model.eval()

        # Name the second linear as '2'
        collector = ActivationCollector(model, ["2"])
        x = torch.randn(2, 8)
        model(x)
        inputs = collector.get_inputs()
        assert "2" in inputs
        assert len(inputs["2"]) == 1
        assert inputs["2"][0].shape == (2, 8)
        collector.remove_hooks()


class TestMetrics:
    def test_mse_zero_for_same(self):
        x = torch.randn(10, 10)
        assert quantization_mse(x, x).item() == pytest.approx(0.0)

    def test_mse_positive(self):
        x = torch.randn(10, 10)
        y = x + 0.1
        assert quantization_mse(x, y).item() > 0

    def test_snr_high_for_small_noise(self):
        x = torch.randn(100, 100)
        y = x + torch.randn_like(x) * 0.001
        snr = quantization_snr(x, y)
        assert snr.item() > 40  # > 40 dB for tiny noise

    def test_snr_low_for_big_noise(self):
        x = torch.randn(100, 100)
        y = x + torch.randn_like(x) * 10.0
        snr = quantization_snr(x, y)
        assert snr.item() < 10

    def test_kurtosis_gaussian(self):
        # Gaussian has excess kurtosis ≈ 0
        torch.manual_seed(0)
        x = torch.randn(10000, 4)
        k = kurtosis(x, dim=0)
        assert k.shape == (4,)
        assert (k.abs() < 0.5).all()  # loose tolerance for finite samples


class TestHelpers:
    def test_make_dummy_linear(self):
        linear = make_dummy_linear(128, 64)
        assert isinstance(linear, nn.Linear)
        assert linear.weight.shape == (64, 128)

    def test_make_calibration_data(self):
        data = make_calibration_data(batch_size=2, seq_len=8, hidden=16, n_batches=4)
        assert len(data) == 4
        assert data[0].shape == (2, 8, 16)

    def test_calibration_deterministic(self):
        d1 = make_calibration_data(seed=42)
        d2 = make_calibration_data(seed=42)
        torch.testing.assert_close(d1[0], d2[0])
