"""Tests for llm_quant.mixed_precision."""

import pytest
import torch
import torch.nn as nn

from llm_quant.mixed_precision import (
    MixedPrecision,
    MixedPrecisionLinear,
    select_protected,
)


class TestSelectProtected:
    def test_mask_passthrough(self):
        mask = torch.tensor([True, False, True, False])
        out = select_protected(4, protect_mask=mask)
        assert torch.equal(out, mask)

    def test_top_fraction(self):
        imp = torch.tensor([0.1, 5.0, 0.2, 9.0, 0.3])
        out = select_protected(5, importance=imp, protect_frac=0.4)  # top 2
        assert out.sum() == 2
        assert out[3] and out[1]            # the two biggest scores
        assert not out[0]

    def test_at_least_one_protected(self):
        imp = torch.randn(1000)
        out = select_protected(1000, importance=imp, protect_frac=0.0001)
        assert out.sum() == 1

    def test_mask_wins_over_importance(self):
        imp = torch.tensor([9.0, 0.0, 0.0])
        mask = torch.tensor([False, True, False])
        out = select_protected(3, importance=imp, protect_mask=mask, protect_frac=0.5)
        assert torch.equal(out, mask)

    def test_needs_one_of_them(self):
        with pytest.raises(ValueError):
            select_protected(8)

    def test_bad_mask_length(self):
        with pytest.raises(ValueError):
            select_protected(4, protect_mask=torch.ones(3, dtype=torch.bool))

    def test_bad_frac(self):
        with pytest.raises(ValueError):
            select_protected(4, importance=torch.randn(4), protect_frac=1.5)


class TestMixedPrecision:
    def test_protected_rows_reconstruct_better(self):
        # the whole point: rows kept at high bits should be closer to the
        # original weight than the rows dropped to low bits.
        torch.manual_seed(0)
        w = torch.randn(16, 64)
        lin = nn.Linear(64, 16, bias=False)
        lin.weight.data = w
        mask = torch.zeros(16, dtype=torch.bool)
        mask[:4] = True

        mp = MixedPrecision(bits_high=8, bits_low=2)
        ml = mp.quantize_linear(lin, protect_mask=mask)

        deq = (ml.weight_int.float() - ml.zero_point) * ml.scale
        row_err = ((deq - w) ** 2).mean(dim=1)
        assert row_err[:4].mean() < row_err[4:].mean()

    def test_effective_bits_in_range(self):
        lin = nn.Linear(32, 100)
        mp = MixedPrecision(bits_high=8, bits_low=4, protect_frac=0.1)
        ml = mp.quantize_linear(lin, importance=torch.randn(100))
        assert 4.0 <= ml.effective_bits <= 8.0
        # 10% protected -> roughly 4 + 0.1*(8-4)
        assert abs(ml.effective_bits - 4.4) < 1e-6

    def test_all_protected_is_uniform_high(self):
        lin = nn.Linear(16, 8)
        mp = MixedPrecision(bits_high=8, bits_low=3)
        ml = mp.quantize_linear(lin, protect_mask=torch.ones(8, dtype=torch.bool))
        assert ml.effective_bits == 8.0

    def test_none_protected_is_uniform_low(self):
        lin = nn.Linear(16, 8)
        mp = MixedPrecision(bits_high=8, bits_low=3)
        ml = mp.quantize_linear(lin, protect_mask=torch.zeros(8, dtype=torch.bool))
        assert ml.effective_bits == 3.0

    def test_forward_shape_and_finite(self):
        lin = nn.Linear(64, 32)
        mp = MixedPrecision(bits_high=8, bits_low=4, protect_frac=0.05)
        ml = mp.quantize_linear(lin, importance=torch.randn(32))
        y = ml(torch.randn(4, 64))
        assert y.shape == (4, 32)
        assert torch.isfinite(y).all()

    def test_dead_channels_dont_nan(self):
        # real models have channels that never fire (all-zero rows). protecting
        # or dropping them should stay finite, not blow up the scale.
        torch.manual_seed(0)
        lin = nn.Linear(64, 16)
        with torch.no_grad():
            lin.weight[0] = 0.0
            lin.weight[7] = 0.0
        imp = torch.randn(16)
        imp[0] = 1e9  # force a dead row into the protected tier
        mp = MixedPrecision(bits_high=8, bits_low=4, protect_frac=0.25)
        ml = mp.quantize_linear(lin, importance=imp)
        assert torch.isfinite(ml.scale).all()
        assert torch.isfinite(ml(torch.randn(2, 64))).all()

    def test_bits_high_must_be_at_least_bits_low(self):
        with pytest.raises(ValueError):
            MixedPrecision(bits_high=3, bits_low=8)

    def test_bias_preserved(self):
        lin = nn.Linear(16, 8)
        mp = MixedPrecision()
        ml = mp.quantize_linear(lin, protect_mask=torch.zeros(8, dtype=torch.bool))
        assert torch.equal(ml.bias.data, lin.bias.data)

    def test_returns_mixed_precision_linear(self):
        lin = nn.Linear(16, 8)
        mp = MixedPrecision()
        ml = mp.quantize_linear(lin, importance=torch.randn(8))
        assert isinstance(ml, MixedPrecisionLinear)
