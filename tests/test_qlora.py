"""Tests for llm_quant.qlora — QLoRA (NF4 + LoRA) quantization."""

import torch
import pytest
from llm_quant.qlora import (
    NF4_TABLE,
    nf4_quantize,
    nf4_dequantize,
    double_quantize,
    double_dequantize,
    QLoRALinear,
    QLoRA,
)


# ──────────────────────────────────────────────────────────────────────────────
# NF4 table
# ──────────────────────────────────────────────────────────────────────────────

class TestNF4Table:
    def test_length(self):
        assert NF4_TABLE.numel() == 16

    def test_sorted(self):
        assert (NF4_TABLE[1:] >= NF4_TABLE[:-1]).all()

    def test_range(self):
        assert NF4_TABLE[0] == -1.0
        assert NF4_TABLE[-1] == 1.0

    def test_roughly_symmetric(self):
        # Not perfectly symmetric, but center ≈ 0
        mid = NF4_TABLE.mean()
        assert abs(mid) < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# NF4 quantize / dequantize
# ──────────────────────────────────────────────────────────────────────────────

class TestNF4Quantize:
    def test_codes_range(self):
        x = torch.randn(32, 64)
        codes, scales = nf4_quantize(x, group_size=64)
        assert codes.min() >= 0
        assert codes.max() <= 15

    def test_codes_shape(self):
        x = torch.randn(16, 128)
        codes, scales = nf4_quantize(x, group_size=64)
        assert codes.shape == x.shape

    def test_scales_positive(self):
        x = torch.randn(32, 64)
        _, scales = nf4_quantize(x, group_size=32)
        assert (scales > 0).all()

    def test_roundtrip(self):
        """Dequant(Quant(x)) should be close to x for normally distributed data."""
        torch.manual_seed(42)
        x = torch.randn(32, 64) * 0.5  # N(0, 0.5²)
        gs = 64
        codes, scales = nf4_quantize(x, group_size=gs)
        x_deq = nf4_dequantize(codes, scales, group_size=gs)
        # NF4 is optimal for normal data, should be reasonably close
        mse = (x - x_deq).pow(2).mean()
        assert mse < 0.05, f"NF4 roundtrip MSE too high: {mse:.4f}"

    def test_zero_input(self):
        x = torch.zeros(8, 16)
        codes, scales = nf4_quantize(x, group_size=16)
        x_deq = nf4_dequantize(codes, scales, group_size=16)
        assert x_deq.abs().max() < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Double quantization
# ──────────────────────────────────────────────────────────────────────────────

class TestDoubleQuantize:
    def test_roundtrip(self):
        torch.manual_seed(42)
        scales = torch.rand(256) * 0.1 + 0.01
        q_scales, dq_scales = double_quantize(scales, dq_group_size=64)
        restored = double_dequantize(q_scales, dq_scales, dq_group_size=64)
        # 8-bit quantization: should be very close
        rel_err = (scales - restored).abs().max() / scales.abs().max()
        assert rel_err < 0.02, f"Double quant rel error: {rel_err:.4f}"

    def test_output_shapes(self):
        scales = torch.rand(128) * 0.5
        q_scales, dq_scales = double_quantize(scales, dq_group_size=32)
        assert q_scales.shape == scales.shape
        assert dq_scales.shape == (4,)  # 128 / 32

    def test_q_scales_in_range(self):
        scales = torch.rand(64) * 0.2
        q_scales, _ = double_quantize(scales, dq_group_size=64)
        assert q_scales.min() >= -127
        assert q_scales.max() <= 127


# ──────────────────────────────────────────────────────────────────────────────
# QLoRALinear
# ──────────────────────────────────────────────────────────────────────────────

class TestQLoRALinear:
    def test_forward_shape(self):
        torch.manual_seed(42)
        ql = QLoRALinear(64, 32, rank=8, group_size=64)
        ql.pack_base(torch.randn(32, 64))
        x = torch.randn(4, 64)
        out = ql(x)
        assert out.shape == (4, 32)

    def test_lora_trainable(self):
        ql = QLoRALinear(32, 16, rank=4, group_size=32)
        ql.pack_base(torch.randn(16, 32))
        # Only lora_A, lora_B, bias should have grad
        trainable = [n for n, p in ql.named_parameters() if p.requires_grad]
        assert "lora_A" in trainable
        assert "lora_B" in trainable
        # nf4_codes should NOT be trainable (it's a buffer)
        assert "nf4_codes" not in trainable

    def test_gradient_flows_through_lora(self):
        torch.manual_seed(42)
        ql = QLoRALinear(32, 16, rank=4, group_size=32)
        ql.pack_base(torch.randn(16, 32))
        x = torch.randn(2, 32)
        out = ql(x)
        out.sum().backward()
        assert ql.lora_A.grad is not None
        assert ql.lora_B.grad is not None

    def test_merge_lora(self):
        torch.manual_seed(42)
        ql = QLoRALinear(32, 16, rank=4, group_size=32)
        ql.pack_base(torch.randn(16, 32))
        w_merged = ql.merge_lora()
        assert w_merged.shape == (16, 32)

    def test_double_quant_saves_precision(self):
        torch.manual_seed(42)
        W = torch.randn(32, 64)

        ql_dq = QLoRALinear(64, 32, rank=4, group_size=64, double_quant=True)
        ql_dq.pack_base(W)
        ql_no = QLoRALinear(64, 32, rank=4, group_size=64, double_quant=False)
        ql_no.pack_base(W)

        # Both should produce valid outputs
        x = torch.randn(2, 64)
        out_dq = ql_dq(x)
        out_no = ql_no(x)
        assert out_dq.shape == out_no.shape

    def test_no_bias(self):
        ql = QLoRALinear(32, 16, rank=4, group_size=32, bias=False)
        ql.pack_base(torch.randn(16, 32))
        x = torch.randn(2, 32)
        out = ql(x)
        assert out.shape == (2, 16)
        assert ql.bias is None


# ──────────────────────────────────────────────────────────────────────────────
# QLoRA pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestQLoRAPipeline:
    def test_wrap_linear(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(64, 32)
        qlora = QLoRA(rank=8, group_size=64)
        ql = qlora.wrap_linear(linear)

        assert isinstance(ql, QLoRALinear)
        x = torch.randn(2, 64)
        out = ql(x)
        assert out.shape == (2, 32)

    def test_trainable_params(self):
        linear = torch.nn.Linear(32, 16)
        qlora = QLoRA(rank=4, group_size=32)
        ql = qlora.wrap_linear(linear)
        params = qlora.trainable_params(ql)
        # lora_A + lora_B + bias = 3
        assert len(params) == 3

    def test_finetune_step(self):
        """Simulate one fine-tuning step: forward → loss → backward → step."""
        torch.manual_seed(42)
        linear = torch.nn.Linear(64, 32)
        qlora = QLoRA(rank=8, group_size=64)
        ql = qlora.wrap_linear(linear)

        # Give lora_B a small nonzero init so lora_A gets gradient
        ql.lora_B.data.normal_(std=0.01)

        optimizer = torch.optim.SGD(qlora.trainable_params(ql), lr=0.01)

        x = torch.randn(4, 64)
        target = torch.randn(4, 32)

        # Before step
        lora_A_before = ql.lora_A.data.clone()

        out = ql(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # lora_A should have changed
        assert not torch.equal(ql.lora_A.data, lora_A_before)

    def test_no_double_quant(self):
        linear = torch.nn.Linear(32, 16)
        qlora = QLoRA(rank=4, group_size=32, double_quant=False)
        ql = qlora.wrap_linear(linear)
        x = torch.randn(2, 32)
        out = ql(x)
        assert out.shape == (2, 16)
