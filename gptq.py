"""
llm_quant.gptq — GPTQ: Accurate Post-Training Quantization for GPTs
                  (Frantar et al., 2023).

Core idea:
    Quantize weights one column (or group of columns) at a time, using the
    inverse Hessian to optimally compensate remaining weights for the error
    introduced by quantization.  This is a second-order, one-shot PTQ method.

Algorithm (per row-block of size ``block_size``):
    1. Collect the Hessian  H = X^T X  from calibration activations.
    2. Compute  H_inv = (H + damp * I)^{-1}  via Cholesky.
    3. For each column j in the current block:
        a. Quantize weight column:  w_q = Q(W[:, j])
        b. Compute error:           δ = (W[:, j] - w_q) / H_inv[j, j]
        c. Compensate remaining:    W[:, j+1:] -= δ @ H_inv[j, j+1:]
    4. Move to next block.

Key components
--------------
- ``compute_hessian``:    build the H = X^T X matrix from calibration data
- ``gptq_quantize``:      core column-wise quantization with Hessian updates
- ``GPTQLinear``:          quantized linear storing INT weights + metadata
- ``GPTQ``:               high-level pipeline API

References
----------
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers", ICLR 2023.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .core.quantizer import (
    QScheme,
    compute_qparams,
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
)


# ──────────────────────────────────────────────────────────────────────────────
# Hessian computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_hessian(
    activations: List[torch.Tensor],
) -> torch.Tensor:
    """Compute Hessian approximation  H = (1/N) Σ X^T X  from calibration data.

    Parameters
    ----------
    activations : list of Tensor [B, S, C] or [B, C]
        Calibration activations feeding into the linear layer.

    Returns
    -------
    H : Tensor [C, C]
        Hessian approximation (positive semi-definite).
    """
    H = None
    n_samples = 0
    for act in activations:
        flat = act.reshape(-1, act.shape[-1]).float()  # [N_i, C]
        n_samples += flat.shape[0]
        hh = flat.t() @ flat  # [C, C]
        if H is None:
            H = hh
        else:
            H = H + hh
    return H / n_samples


# ──────────────────────────────────────────────────────────────────────────────
# Core GPTQ quantization
# ──────────────────────────────────────────────────────────────────────────────

def gptq_quantize(
    weight: torch.Tensor,
    H: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the GPTQ algorithm on a single weight matrix.

    Parameters
    ----------
    weight : Tensor [out_features, in_features]
    H      : Tensor [in_features, in_features]  — Hessian approximation
    n_bits : int   — target bit-width
    group_size : int — group quantization size (0 = per-channel)
    block_size : int — columns processed per block
    damp_percent : float — diagonal damping ratio
    symmetric : bool — symmetric quantization

    Returns
    -------
    w_q   : Tensor [out_features, in_features]  — quantized (int) weight
    scale : Tensor — quantization scales
    zp    : Tensor — quantization zero-points
    """
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype

    W = weight.clone().float()
    H = H.float().to(device)

    # Damping
    diag_mean = H.diagonal().mean()
    H += damp_percent * diag_mean * torch.eye(in_features, device=device)

    # Cholesky of H (upper triangular)
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fall back to pseudo-inverse if Cholesky fails
        H_inv = torch.linalg.pinv(H)

    # Pre-allocate output
    w_q = torch.zeros_like(W)

    # Determine quantization groups
    use_group = group_size > 0
    scheme = QScheme.PER_GROUP if use_group else QScheme.PER_CHANNEL

    # Pre-compute per-group or per-channel qparams
    if use_group:
        n_groups = math.ceil(in_features / group_size)
    else:
        n_groups = 1

    # Process in blocks
    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        W_block = W[:, block_start:block_end].clone()
        H_inv_block = H_inv[block_start:block_end, block_start:block_end]
        err = torch.zeros_like(W_block)

        for j in range(block_end - block_start):
            col_idx = block_start + j
            w_col = W_block[:, j]  # [out_features]

            # Compute qparams for this column's group
            if use_group:
                group_idx = col_idx // group_size
                g_start = group_idx * group_size
                g_end = min(g_start + group_size, in_features)
                w_group = W[:, g_start:g_end]
                amax = w_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                if symmetric:
                    qmax = 2 ** (n_bits - 1) - 1
                    scale = amax / qmax
                    zp = torch.zeros_like(scale)
                else:
                    qmax = 2 ** n_bits - 1
                    w_min = w_group.amin(dim=1, keepdim=True)
                    w_max = w_group.amax(dim=1, keepdim=True)
                    scale = (w_max - w_min).clamp(min=1e-12) / qmax
                    zp = torch.round(-w_min / scale).clamp(0, qmax)
            else:
                amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                if symmetric:
                    qmax = 2 ** (n_bits - 1) - 1
                    scale = amax / qmax
                    zp = torch.zeros_like(scale)
                else:
                    qmax = 2 ** n_bits - 1
                    w_min = W.amin(dim=1, keepdim=True)
                    w_max = W.amax(dim=1, keepdim=True)
                    scale = (w_max - w_min).clamp(min=1e-12) / qmax
                    zp = torch.round(-w_min / scale).clamp(0, qmax)

            # Quantize & dequantize the column
            if symmetric:
                qmin = -(2 ** (n_bits - 1))
                qmax_val = 2 ** (n_bits - 1) - 1
            else:
                qmin = 0
                qmax_val = 2 ** n_bits - 1

            col_q = torch.round(w_col.unsqueeze(1) / scale + zp).clamp(qmin, qmax_val)
            col_dq = ((col_q - zp) * scale).squeeze(1)

            w_q[:, col_idx] = col_q.squeeze(1)

            # Error and Hessian compensation
            h_jj = H_inv_block[j, j].clamp(min=1e-12)
            delta = (w_col - col_dq) / h_jj   # [out_features]
            err[:, j] = delta

            # Compensate future columns within this block
            if j + 1 < block_end - block_start:
                W_block[:, j + 1:] -= delta.unsqueeze(1) * H_inv_block[j, j + 1:block_end - block_start].unsqueeze(0)

        # Compensate columns beyond this block
        if block_end < in_features:
            W[:, block_end:] -= err @ H_inv[block_start:block_end, block_end:]

    # Compute final qparams for the quantized weight (for storage)
    scale_out, zp_out = compute_qparams(
        dequantize_gptq(w_q, weight, n_bits, symmetric, group_size),
        n_bits, symmetric, scheme, group_size,
    )

    return w_q, scale_out, zp_out


def dequantize_gptq(
    w_q: torch.Tensor,
    weight_orig: torch.Tensor,
    n_bits: int,
    symmetric: bool,
    group_size: int,
) -> torch.Tensor:
    """Re-dequantize the integer weight matrix back to float using proper qparams.

    Recompute scale/zp from the original weight (used at the time of quantization)
    to dequantize consistently.
    """
    # Just return the scale-based dequant of the quantized weight
    scheme = QScheme.PER_GROUP if group_size > 0 else QScheme.PER_CHANNEL
    scale, zp = compute_qparams(weight_orig, n_bits, symmetric, scheme, group_size)

    if group_size > 0:
        out_features, in_features = w_q.shape
        n_groups = math.ceil(in_features / group_size)
        padded = in_features % group_size
        if padded != 0:
            w_pad = F.pad(w_q, (0, group_size - padded), value=0.0)
        else:
            w_pad = w_q
        w_grouped = w_pad.reshape(out_features, n_groups, group_size)
        dq = (w_grouped - zp) * scale
        return dq.reshape(out_features, -1)[:, :in_features]
    else:
        return (w_q - zp) * scale


# ──────────────────────────────────────────────────────────────────────────────
# GPTQLinear — drop-in replacement
# ──────────────────────────────────────────────────────────────────────────────

class GPTQLinear(nn.Module):
    """A linear layer whose weights are quantized via GPTQ.

    Stores INT weights and performs dequantize-on-the-fly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bits: int = 4,
        symmetric: bool = True,
        group_size: int = 128,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.group_size = group_size

        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1))
        self.register_buffer("zero_point", torch.zeros(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def pack(self, w_q: torch.Tensor, weight_orig: torch.Tensor) -> None:
        """Store quantized weights and compute proper scale/zp."""
        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        scale, zp = compute_qparams(
            weight_orig, self.n_bits, self.symmetric, scheme, self.group_size,
        )
        self.weight_int.copy_(w_q.to(torch.int8).reshape_as(self.weight_int))
        scale_flat = scale.reshape(self.out_features, -1)
        zp_flat = zp.reshape(self.out_features, -1)
        if scale_flat.shape != self.scale.shape:
            self.scale = scale_flat.clone()
            self.zero_point = zp_flat.clone()
        else:
            self.scale.copy_(scale_flat)
            self.zero_point.copy_(zp_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size > 0:
            n_groups = self.scale.shape[1]
            w_int = self.weight_int.float()
            padded_in = n_groups * self.group_size
            if padded_in > self.in_features:
                w_int = F.pad(w_int, (0, padded_in - self.in_features))
            w_grouped = w_int.reshape(self.out_features, n_groups, self.group_size)
            scale = self.scale.unsqueeze(-1)
            zp = self.zero_point.unsqueeze(-1)
            w_float = (w_grouped - zp) * scale
            w_float = w_float.reshape(self.out_features, -1)[:, :self.in_features]
        else:
            w_float = (self.weight_int.float() - self.zero_point) * self.scale
        return F.linear(x, w_float, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.n_bits}, group={self.group_size}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────────────────

class GPTQ:
    """High-level GPTQ pipeline.

    Usage
    -----
        gptq = GPTQ(n_bits=4, group_size=128)
        gptq.calibrate(activations)
        q_linear = gptq.quantize_linear(linear)
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        block_size: int = 128,
        damp_percent: float = 0.01,
        symmetric: bool = True,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.block_size = block_size
        self.damp_percent = damp_percent
        self.symmetric = symmetric
        self._hessian: Optional[torch.Tensor] = None

    def calibrate(self, activations: List[torch.Tensor]) -> None:
        """Compute Hessian from calibration data."""
        self._hessian = compute_hessian(activations)

    def quantize_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a weight matrix using GPTQ's OBQ procedure.

        Returns (w_q, scale, zp).
        """
        assert self._hessian is not None, "Call .calibrate() first"
        return gptq_quantize(
            weight,
            self._hessian,
            n_bits=self.n_bits,
            group_size=self.group_size,
            block_size=self.block_size,
            damp_percent=self.damp_percent,
            symmetric=self.symmetric,
        )

    def quantize_linear(self, linear: nn.Linear) -> GPTQLinear:
        """Quantize an nn.Linear layer using GPTQ.

        Returns a GPTQLinear with packed INT weights.
        """
        w_q, scale, zp = self.quantize_weight(linear.weight.data)
        ql = GPTQLinear(
            linear.in_features,
            linear.out_features,
            n_bits=self.n_bits,
            symmetric=self.symmetric,
            group_size=self.group_size,
            bias=linear.bias is not None,
        )
        ql.pack(w_q, linear.weight.data)
        if linear.bias is not None:
            ql.bias.data.copy_(linear.bias.data)
        return ql
