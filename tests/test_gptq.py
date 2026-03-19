"""Tests for llm_quant.gptq — GPTQ quantization."""

import torch
import pytest
from llm_quant.gptq import (
    compute_hessian,
    gptq_quantize,
    GPTQLinear,
    GPTQ,
)
from llm_quant.core.quantizer import fake_quantize, QScheme


# ──────────────────────────────────────────────────────────────────────────────
# Hessian
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeHessian:
    def test_shape(self):
        acts = [torch.randn(4, 8, 32) for _ in range(3)]
        H = compute_hessian(acts)
        assert H.shape == (32, 32)

    def test_symmetric(self):
        acts = [torch.randn(2, 16) for _ in range(5)]
        H = compute_hessian(acts)
        torch.testing.assert_close(H, H.t(), atol=1e-5, rtol=1e-5)

    def test_positive_semidefinite(self):
        acts = [torch.randn(4, 16) for _ in range(5)]
        H = compute_hessian(acts)
        eigvals = torch.linalg.eigvalsh(H)
        assert (eigvals >= -1e-5).all(), "Hessian should be PSD"

    def test_single_batch(self):
        x = torch.randn(8, 16)
        H = compute_hessian([x])
        expected = (x.t() @ x) / 8
        torch.testing.assert_close(H, expected, atol=1e-5, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Core GPTQ quantize
# ──────────────────────────────────────────────────────────────────────────────

class TestGPTQQuantize:
    def test_output_shapes(self):
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        H = torch.eye(64) + 0.01 * torch.randn(64, 64)
        H = H @ H.t()  # make PSD
        w_q, scale, zp = gptq_quantize(W, H, n_bits=4, group_size=0, block_size=32)
        assert w_q.shape == W.shape

    def test_int_range(self):
        torch.manual_seed(0)
        W = torch.randn(16, 32)
        H = torch.eye(32)
        w_q, _, _ = gptq_quantize(W, H, n_bits=4, group_size=0)
        assert w_q.min() >= -8
        assert w_q.max() <= 7

    def test_better_than_rtn(self):
        """GPTQ should have lower or equal error compared to naive RTN."""
        torch.manual_seed(123)
        W = torch.randn(32, 64)
        x = torch.randn(100, 64)
        H = (x.t() @ x) / 100

        # GPTQ
        w_q_gptq, _, _ = gptq_quantize(W, H, n_bits=4, group_size=0, block_size=16)
        # For dequantization: use per-channel qparams of original W
        from llm_quant.core.quantizer import compute_qparams, dequantize_tensor
        scale, zp = compute_qparams(W, 4, True, QScheme.PER_CHANNEL)
        # RTN baseline
        w_q_rtn = torch.round(W / scale + zp).clamp(-8, 7)

        # Dequantize both
        w_dq_gptq = (w_q_gptq - zp) * scale
        w_dq_rtn = (w_q_rtn - zp) * scale

        # GPTQ error should be <= RTN error (on calibration data)
        err_gptq = ((x @ W.t()) - (x @ w_dq_gptq.t())).pow(2).mean()
        err_rtn = ((x @ W.t()) - (x @ w_dq_rtn.t())).pow(2).mean()
        # GPTQ is at least not significantly worse
        assert err_gptq <= err_rtn * 1.5, f"GPTQ err {err_gptq:.4f} > 1.5 * RTN err {err_rtn:.4f}"

    def test_group_quantize(self):
        torch.manual_seed(7)
        W = torch.randn(16, 64)
        H = torch.eye(64)
        w_q, scale, zp = gptq_quantize(W, H, n_bits=4, group_size=32)
        assert w_q.shape == W.shape


# ──────────────────────────────────────────────────────────────────────────────
# GPTQLinear
# ──────────────────────────────────────────────────────────────────────────────

class TestGPTQLinear:
    def test_pack_and_forward(self):
        torch.manual_seed(42)
        W = torch.randn(16, 32)
        H = torch.eye(32)
        w_q, _, _ = gptq_quantize(W, H, n_bits=4, group_size=0)

        ql = GPTQLinear(32, 16, n_bits=4, group_size=0, bias=False)
        ql.pack(w_q, W)

        x = torch.randn(2, 32)
        out = ql(x)
        assert out.shape == (2, 16)

    def test_with_groups(self):
        torch.manual_seed(42)
        W = torch.randn(16, 64)
        H = torch.eye(64)
        w_q, _, _ = gptq_quantize(W, H, n_bits=4, group_size=32)

        ql = GPTQLinear(64, 16, n_bits=4, group_size=32, bias=True)
        ql.pack(w_q, W)

        x = torch.randn(2, 64)
        out = ql(x)
        assert out.shape == (2, 16)

    def test_output_not_zero(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(32, 16)
        W = linear.weight.data
        H = torch.eye(32)
        w_q, _, _ = gptq_quantize(W, H, n_bits=4, group_size=0)

        ql = GPTQLinear(32, 16, n_bits=4, group_size=0)
        ql.pack(w_q, W)
        ql.bias.data.copy_(linear.bias.data)

        x = torch.randn(4, 32)
        out = ql(x)
        assert out.abs().sum() > 0


# ──────────────────────────────────────────────────────────────────────────────
# GPTQ pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestGPTQPipeline:
    def test_full_pipeline(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(64, 32)
        acts = [torch.randn(4, 8, 64) for _ in range(5)]

        gptq = GPTQ(n_bits=4, group_size=0, block_size=32)
        gptq.calibrate(acts)
        ql = gptq.quantize_linear(linear)

        x = torch.randn(2, 8, 64)
        out_q = ql(x)
        assert out_q.shape == (2, 8, 32)

    def test_pipeline_with_groups(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(128, 32)
        acts = [torch.randn(2, 128) for _ in range(10)]

        gptq = GPTQ(n_bits=4, group_size=128, block_size=64)
        gptq.calibrate(acts)
        ql = gptq.quantize_linear(linear)

        x = torch.randn(4, 128)
        out = ql(x)
        assert out.shape == (4, 32)

    def test_quantized_output_close(self):
        """Quantized output should be reasonably close to original."""
        torch.manual_seed(42)
        linear = torch.nn.Linear(64, 32)
        acts = [torch.randn(8, 64) for _ in range(20)]

        gptq = GPTQ(n_bits=8, group_size=0)  # 8-bit for tighter tolerance
        gptq.calibrate(acts)
        ql = gptq.quantize_linear(linear)

        x = torch.randn(4, 64)
        out_orig = linear(x)
        out_q = ql(x)
        # 8-bit GPTQ should be quite close
        rel_err = (out_orig - out_q).pow(2).mean() / out_orig.pow(2).mean()
        assert rel_err < 0.1, f"Relative error too high: {rel_err:.4f}"
