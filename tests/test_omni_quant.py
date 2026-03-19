"""Tests for llm_quant.omni_quant — OmniQuant."""

import pytest
import torch
import torch.nn as nn
from llm_quant.omni_quant import (
    OmniQuant,
    LearnableClipping,
    LearnableSmoothing,
    OmniQuantBlock,
)
from llm_quant.core.quantizer import QuantizedLinear


class TestLearnableClipping:
    def test_clips_values(self):
        lc = LearnableClipping(2, init_val=1.0)
        weight = torch.tensor([[-2.0, 3.0, 0.5, -0.1],
                               [5.0, -4.0, 0.2, 1.5]])
        clipped = lc(weight)
        alpha = lc.get_alpha()  # should be ≈ 1.0
        assert (clipped.abs() <= alpha + 1e-5).all()

    def test_gradient_flows(self):
        lc = LearnableClipping(4)
        w = torch.randn(4, 8)
        clipped = lc(w)
        clipped.sum().backward()
        assert lc.log_alpha.grad is not None


class TestLearnableSmoothing:
    def test_shape(self):
        ls = LearnableSmoothing(32)
        x = torch.randn(2, 8, 32)
        w = torch.randn(16, 32)
        x_s, w_s = ls(x, w)
        assert x_s.shape == x.shape
        assert w_s.shape == w.shape

    def test_init_identity(self):
        ls = LearnableSmoothing(16)
        x = torch.randn(2, 16)
        w = torch.randn(8, 16)
        x_s, w_s = ls(x, w)
        # log_scale=0 → s=1 → no change
        torch.testing.assert_close(x_s, x)
        torch.testing.assert_close(w_s, w)

    def test_preserves_output(self):
        torch.manual_seed(42)
        ls = LearnableSmoothing(32)
        ls.log_scale.data.normal_(std=0.5)
        x = torch.randn(4, 32)
        w = torch.randn(16, 32)
        x_s, w_s = ls(x, w)
        y_orig = x @ w.t()
        y_smooth = x_s @ w_s.t()
        torch.testing.assert_close(y_smooth, y_orig, atol=1e-5, rtol=1e-5)


class TestOmniQuantBlock:
    def test_loss_computable(self):
        block = OmniQuantBlock(32, 16, n_bits=4, group_size=16)
        weight = torch.randn(16, 32)
        x = torch.randn(2, 8, 32)
        y_q, loss = block(weight, x)
        assert y_q.shape == (2, 8, 16)
        assert loss.item() >= 0

    def test_lwc_only(self):
        block = OmniQuantBlock(32, 16, use_lwc=True, use_let=False)
        weight = torch.randn(16, 32)
        x = torch.randn(2, 8, 32)
        _, loss = block(weight, x)
        loss.backward()
        assert block.lwc.log_alpha.grad is not None

    def test_let_only(self):
        block = OmniQuantBlock(32, 16, use_lwc=False, use_let=True)
        weight = torch.randn(16, 32)
        x = torch.randn(2, 8, 32)
        _, loss = block(weight, x)
        loss.backward()
        assert block.let.log_scale.grad is not None


class TestOmniQuantPipeline:
    def test_optimise_returns_results(self):
        torch.manual_seed(42)
        oq = OmniQuant(n_bits=4, group_size=16, lr=1e-2)
        weight = torch.randn(16, 32)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]

        results = oq.optimise(weight, calib, steps=20)
        assert "scales" in results
        assert "clip_alpha" in results
        assert results["scales"].shape == (32,)
        assert results["clip_alpha"].shape == (16, 1)

    def test_full_pipeline(self):
        torch.manual_seed(42)
        oq = OmniQuant(n_bits=4, group_size=16)
        linear = nn.Linear(32, 16)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]

        results = oq.optimise(linear.weight.data, calib, steps=20)
        oq.apply_transforms(linear, results)
        q_linear = oq.quantize_linear(linear)

        assert isinstance(q_linear, QuantizedLinear)
        x = torch.randn(2, 32)
        y = q_linear(x)
        assert y.shape == (2, 16)
