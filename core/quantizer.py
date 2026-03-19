"""
llm_quant.core.quantizer — Low-level symmetric / asymmetric quantization.

Provides:
- STE (Straight-Through Estimator) rounding
- Per-tensor / per-channel / per-group quantization parameters
- Fake-quantize (quantize → dequantize) simulation
- QuantizedLinear: a drop-in nn.Linear replacement that stores INT weights
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class QScheme(Enum):
    """Quantization scheme."""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight-Through Estimator (gradient passes through)."""
    return x + (x.round() - x).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Quantization parameters
# ──────────────────────────────────────────────────────────────────────────────

def compute_qparams(
    x: torch.Tensor,
    n_bits: int = 8,
    symmetric: bool = True,
    scheme: QScheme = QScheme.PER_TENSOR,
    group_size: int = 128,
    channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and zero-point for quantization.

    Returns
    -------
    scale : Tensor  — per-element scale factor
    zero_point : Tensor — per-element zero point (0 for symmetric)
    """
    if symmetric:
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** n_bits - 1

    if scheme == QScheme.PER_TENSOR:
        if symmetric:
            amax = x.abs().max().clamp(min=1e-12)
            scale = amax / qmax
            zero_point = torch.zeros(1, device=x.device, dtype=x.dtype)
        else:
            x_min = x.min()
            x_max = x.max()
            scale = (x_max - x_min).clamp(min=1e-12) / (qmax - qmin)
            zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)

    elif scheme == QScheme.PER_CHANNEL:
        assert x.ndim >= 2, "per-channel requires at least 2-D tensor"
        reduce_dims = [i for i in range(x.ndim) if i != channel_dim]
        if symmetric:
            amax = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12)
            scale = amax / qmax
            zero_point = torch.zeros_like(scale)
        else:
            x_min = x.amin(dim=reduce_dims, keepdim=True)
            x_max = x.amax(dim=reduce_dims, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-12) / (qmax - qmin)
            zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)

    elif scheme == QScheme.PER_GROUP:
        assert x.ndim == 2, "per-group requires a 2-D weight matrix"
        out_features, in_features = x.shape
        n_groups = math.ceil(in_features / group_size)
        # Pad if necessary
        padded = in_features % group_size
        if padded != 0:
            x = F.pad(x, (0, group_size - padded), value=0.0)
        x_grouped = x.reshape(out_features, n_groups, group_size)
        if symmetric:
            amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = amax / qmax
            zero_point = torch.zeros_like(scale)
        else:
            x_min = x_grouped.amin(dim=-1, keepdim=True)
            x_max = x_grouped.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-12) / (qmax - qmin)
            zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return scale, zero_point


# ──────────────────────────────────────────────────────────────────────────────
# Quantize / Dequantize
# ──────────────────────────────────────────────────────────────────────────────

def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    n_bits: int = 8,
    symmetric: bool = True,
) -> torch.Tensor:
    """Quantize floating-point tensor to integer representation."""
    if symmetric:
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** n_bits - 1

    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return q


def dequantize_tensor(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize integer tensor back to floating-point."""
    return (q - zero_point) * scale


def fake_quantize(
    x: torch.Tensor,
    n_bits: int = 8,
    symmetric: bool = True,
    scheme: QScheme = QScheme.PER_TENSOR,
    group_size: int = 128,
) -> torch.Tensor:
    """Simulated quantization: quantize → dequantize (differentiable via STE).

    Returns a floating-point tensor that simulates quantization noise.
    """
    scale, zp = compute_qparams(x, n_bits, symmetric, scheme, group_size)

    if scheme == QScheme.PER_GROUP and x.ndim == 2:
        out_features, in_features = x.shape
        n_groups = math.ceil(in_features / group_size)
        padded = in_features % group_size
        if padded != 0:
            x_padded = F.pad(x, (0, group_size - padded), value=0.0)
        else:
            x_padded = x
        x_grouped = x_padded.reshape(out_features, n_groups, group_size)
        q = ste_round(x_grouped / scale + zp)
        if symmetric:
            qmin = -(2 ** (n_bits - 1))
            qmax = 2 ** (n_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** n_bits - 1
        q = q.clamp(qmin, qmax)
        dq = (q - zp) * scale
        dq = dq.reshape(out_features, -1)[:, :in_features]
        return dq

    q = ste_round(x / scale + zp)
    if symmetric:
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** n_bits - 1
    q = q.clamp(qmin, qmax)
    return (q - zp) * scale


# ──────────────────────────────────────────────────────────────────────────────
# QuantizedLinear
# ──────────────────────────────────────────────────────────────────────────────

class QuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` storing INT weights.

    Keeps ``float`` scale/zero_point; performs dequantize-on-the-fly during forward.

    Parameters
    ----------
    in_features, out_features : int
    n_bits : int
    symmetric : bool
    group_size : int  (0 = per-channel)
    bias : bool
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bits: int = 8,
        symmetric: bool = True,
        group_size: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.group_size = group_size

        # Start with float weights (call .quantize_weights() after assigning)
        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1))
        self.register_buffer("zero_point", torch.zeros(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def quantize_weights(self, weight: torch.Tensor) -> None:
        """Quantize a float weight matrix and store INT + scale/zp."""
        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        scale, zp = compute_qparams(
            weight, self.n_bits, self.symmetric, scheme, self.group_size,
        )
        if scheme == QScheme.PER_GROUP:
            out_features, in_features = weight.shape
            n_groups = math.ceil(in_features / self.group_size)
            padded = in_features % self.group_size
            if padded != 0:
                w_pad = F.pad(weight, (0, self.group_size - padded), value=0.0)
            else:
                w_pad = weight
            w_grouped = w_pad.reshape(out_features, n_groups, self.group_size)
            q = quantize_tensor(w_grouped, scale, zp, self.n_bits, self.symmetric)
            q = q.reshape(out_features, -1)[:, :in_features]
        else:
            q = quantize_tensor(weight, scale, zp, self.n_bits, self.symmetric)
        self.weight_int.copy_(q.to(torch.int8).reshape_as(self.weight_int))
        scale_flat = scale.reshape(self.out_features, -1)
        zp_flat = zp.reshape(self.out_features, -1)
        # Re-register buffers if shape changed (per-group case)
        if scale_flat.shape != self.scale.shape:
            self.scale = scale_flat.clone()
            self.zero_point = zp_flat.clone()
        else:
            self.scale.copy_(scale_flat)
            self.zero_point.copy_(zp_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on-the-fly
        if self.group_size > 0:
            n_groups = self.scale.shape[1]
            w_int = self.weight_int.float()
            # Pad if necessary
            padded_in = n_groups * self.group_size
            if padded_in > self.in_features:
                w_int = F.pad(w_int, (0, padded_in - self.in_features))
            w_grouped = w_int.reshape(self.out_features, n_groups, self.group_size)
            scale = self.scale.unsqueeze(-1)  # [out, n_groups, 1]
            zp = self.zero_point.unsqueeze(-1)
            w_float = (w_grouped - zp) * scale
            w_float = w_float.reshape(self.out_features, -1)[:, :self.in_features]
        else:
            w_float = dequantize_tensor(
                self.weight_int.float(), self.scale, self.zero_point,
            )
        return F.linear(x, w_float, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_bits: int = 8,
                    symmetric: bool = True, group_size: int = 0) -> "QuantizedLinear":
        """Create a QuantizedLinear from an existing nn.Linear."""
        ql = cls(
            linear.in_features, linear.out_features,
            n_bits=n_bits, symmetric=symmetric, group_size=group_size,
            bias=linear.bias is not None,
        )
        ql.quantize_weights(linear.weight.data)
        if linear.bias is not None:
            ql.bias.data.copy_(linear.bias.data)
        return ql

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.n_bits}, sym={self.symmetric}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )
