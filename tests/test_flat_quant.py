"""Tests for llm_quant.flat_quant — FlatQuant."""

import pytest
import torch
import torch.nn as nn
from llm_quant.flat_quant import FlatQuant, FlatTransform, FlatQuantSolver
from llm_quant.core.quantizer import QuantizedLinear


class TestFlatTransform:
    def test_shape(self):
        ft = FlatTransform(32)
        T = ft()
        assert T.shape == (32, 32)

    def test_init_identity(self):
        ft = FlatTransform(16)
        T = ft()
        # Initial: rotation=I, scale=1 → T = I
        torch.testing.assert_close(T, torch.eye(16), atol=1e-6, rtol=1e-6)

    def test_gradient_flows(self):
        ft = FlatTransform(16)
        T = ft()
        loss = T.sum()
        loss.backward()
        assert ft.rotation.params.grad is not None
        assert ft.log_scale.grad is not None


class TestFlatQuantSolver:
    def test_loss_computable(self):
        solver = FlatQuantSolver(32, 16, n_bits=4, group_size=16)
        weight = torch.randn(16, 32)
        x = torch.randn(2, 8, 32)
        y_q, loss = solver(weight, x)
        assert y_q.shape == (2, 8, 16)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        solver = FlatQuantSolver(32, 16, n_bits=4, group_size=16)
        weight = torch.randn(16, 32)
        x = torch.randn(2, 8, 32)
        _, loss = solver(weight, x)
        loss.backward()
        # Check gradients exist for transform params
        assert solver.t_act.rotation.params.grad is not None
        assert solver.t_wt.log_scale.grad is not None


class TestFlatQuantPipeline:
    def test_optimise_returns_transforms(self):
        torch.manual_seed(42)
        fq = FlatQuant(n_bits=4, group_size=16, lr=1e-2)
        weight = torch.randn(16, 32)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]

        T_act, T_wt = fq.optimise(weight, calib, steps=20)
        assert T_act.shape == (32, 32)
        assert T_wt.shape == (32, 32)

    def test_full_pipeline(self):
        torch.manual_seed(42)
        fq = FlatQuant(n_bits=4, group_size=16)
        linear = nn.Linear(32, 16)
        calib = [torch.randn(2, 4, 32) for _ in range(4)]

        T_act, T_wt = fq.optimise(linear.weight.data, calib, steps=20)
        fq.apply_transforms(linear, T_act, T_wt)
        q_linear = fq.quantize_linear(linear)

        assert isinstance(q_linear, QuantizedLinear)
        x = torch.randn(2, 32)
        y = q_linear(x)
        assert y.shape == (2, 16)
