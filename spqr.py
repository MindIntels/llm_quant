"""
llm_quant.spqr — SpQR: Sparse-Quantized Representation (Dettmers et al., 2024).

Core idea:
    A small fraction (~1%) of weights are **outliers** that dominate
    quantization error.  SpQR identifies them and stores them in full
    precision (sparse), while the remaining ~99% are group-quantized
    to INT3/INT4.

    Result: near-lossless INT3 quantization with <5% sparse overhead.

Key steps
---------
1. **Sensitivity analysis**: compute per-weight quantization error
   weighted by the Hessian diagonal (from calibration data).
2. **Outlier detection**: flag weights whose sensitivity exceeds a
   percentile threshold.
3. **Hybrid storage**: outliers → FP16 sparse; rest → INT group-quantized.

Key components
--------------
- ``compute_weight_sensitivity``: Hessian-weighted per-element error
- ``detect_outliers``:            threshold-based outlier mask
- ``SpQRLinear``:                 hybrid sparse + quantized Linear
- ``SpQR``:                      high-level API
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .core.quantizer import (
    compute_qparams, quantize_tensor, dequantize_tensor,
    fake_quantize, QScheme,
)


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity & Outlier Detection
# ──────────────────────────────────────────────────────────────────────────────

def compute_weight_sensitivity(
    weight: torch.Tensor,
    activations: List[torch.Tensor],
    n_bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """Hessian-weighted per-element quantization error.

    Sensitivity(i,j) = (W_ij - Q(W_ij))² · H_jj

    where H_jj ≈ mean( x_j² ) is the diagonal Hessian approximation.
    """
    # Compute diagonal Hessian: H_jj = E[x_j^2]
    h_diag = torch.zeros(weight.shape[1], device=weight.device)
    count = 0
    for act in activations:
        flat = act.reshape(-1, act.shape[-1]).to(weight.device)
        h_diag += (flat ** 2).sum(dim=0)
        count += flat.shape[0]
    h_diag /= count

    # Quantization error per element
    scheme = QScheme.PER_GROUP if group_size > 0 else QScheme.PER_CHANNEL
    w_q = fake_quantize(weight, n_bits, True, scheme, group_size)
    quant_error = (weight - w_q) ** 2

    # Weight by Hessian
    sensitivity = quant_error * h_diag.unsqueeze(0)
    return sensitivity


def detect_outliers(
    sensitivity: torch.Tensor,
    outlier_fraction: float = 0.01,
) -> torch.Tensor:
    """Return a boolean mask of outlier weights.

    Parameters
    ----------
    sensitivity : [out, in] per-element sensitivity scores.
    outlier_fraction : fraction of weights to mark as outliers (e.g. 0.01 = 1%).

    Returns
    -------
    mask : [out, in] boolean.  True = outlier (keep in FP).
    """
    threshold = torch.quantile(
        sensitivity.flatten().float(),
        1.0 - outlier_fraction,
    )
    return sensitivity >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# SpQR Linear Layer
# ──────────────────────────────────────────────────────────────────────────────

class SpQRLinear(nn.Module):
    """Hybrid sparse-FP + quantized-INT Linear layer.

    Stores:
    - ``weight_int``:   [out, in]  quantized INT weights (outliers zeroed).
    - ``scale``, ``zp``: quantization parameters.
    - ``outlier_mask``: [out, in]  boolean mask.
    - ``outlier_vals``: 1-D tensor of FP outlier values.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_bits: int = 4, group_size: int = 128, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.group_size = group_size

        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1))
        self.register_buffer("zero_point", torch.zeros(out_features, 1))
        self.register_buffer("outlier_mask", torch.zeros(out_features, in_features, dtype=torch.bool))
        self.register_buffer("outlier_vals", torch.zeros(0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def pack(self, weight: torch.Tensor, outlier_mask: torch.Tensor) -> None:
        """Pack weight into quantized + sparse outlier representation."""
        self.outlier_mask.copy_(outlier_mask)

        # Store outlier values
        self.outlier_vals = weight[outlier_mask].clone()

        # Zero out outliers for INT quantization
        w_for_quant = weight.clone()
        w_for_quant[outlier_mask] = 0.0

        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        scale, zp = compute_qparams(w_for_quant, self.n_bits, True, scheme, self.group_size)

        if scheme == QScheme.PER_GROUP:
            import math
            import torch.nn.functional as F_pad
            out_f, in_f = w_for_quant.shape
            n_groups = math.ceil(in_f / self.group_size)
            padded = in_f % self.group_size
            if padded != 0:
                w_pad = torch.nn.functional.pad(w_for_quant, (0, self.group_size - padded))
            else:
                w_pad = w_for_quant
            w_grouped = w_pad.reshape(out_f, n_groups, self.group_size)
            q = quantize_tensor(w_grouped, scale, zp, self.n_bits, True)
            q = q.reshape(out_f, -1)[:, :in_f]
        else:
            q = quantize_tensor(w_for_quant, scale, zp, self.n_bits, True)

        self.weight_int.copy_(q.to(torch.int8).reshape_as(self.weight_int))
        scale_flat = scale.reshape(self.out_features, -1)
        zp_flat = zp.reshape(self.out_features, -1)
        if scale_flat.shape != self.scale.shape:
            self.scale = scale_flat.clone()
            self.zero_point = zp_flat.clone()
        else:
            self.scale.copy_(scale_flat)
            self.zero_point.copy_(zp_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize INT weights
        if self.group_size > 0:
            import math
            n_groups = self.scale.shape[1] if self.scale.ndim > 1 else 1
            w_int = self.weight_int.float()
            padded_in = n_groups * self.group_size
            if padded_in > self.in_features:
                w_int = F.pad(w_int, (0, padded_in - self.in_features))
            w_grouped = w_int.reshape(self.out_features, n_groups, self.group_size)
            scale = self.scale.unsqueeze(-1) if self.scale.ndim == 2 else self.scale
            zp = self.zero_point.unsqueeze(-1) if self.zero_point.ndim == 2 else self.zero_point
            w_float = (w_grouped - zp) * scale
            w_float = w_float.reshape(self.out_features, -1)[:, :self.in_features]
        else:
            w_float = dequantize_tensor(
                self.weight_int.float(), self.scale, self.zero_point,
            )
            if w_float.shape[1] != self.in_features:
                w_float = w_float.reshape(self.out_features, -1)[:, :self.in_features]

        # Restore outlier values
        w_float[self.outlier_mask] = self.outlier_vals.to(w_float.dtype)

        return F.linear(x, w_float, self.bias)

    @property
    def sparsity(self) -> float:
        """Fraction of weights stored as FP outliers."""
        return self.outlier_mask.float().mean().item()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.n_bits}, group={self.group_size}, "
            f"outliers={self.sparsity:.2%}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SpQR High-Level API
# ──────────────────────────────────────────────────────────────────────────────

class SpQR:
    """SpQR: Sparse-Quantized Representation.

    Usage:
        spqr = SpQR(n_bits=4, group_size=128, outlier_fraction=0.01)
        spqr.calibrate(activations)
        spqr_linear = spqr.quantize_linear(linear)
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        outlier_fraction: float = 0.01,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.outlier_fraction = outlier_fraction
        self._activations: Optional[List[torch.Tensor]] = None

    def calibrate(self, activations: List[torch.Tensor]) -> None:
        self._activations = activations

    def compute_sensitivity(self, weight: torch.Tensor) -> torch.Tensor:
        assert self._activations is not None, "Call calibrate() first"
        return compute_weight_sensitivity(
            weight, self._activations, self.n_bits, self.group_size,
        )

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> SpQRLinear:
        """Quantize a Linear layer using SpQR."""
        assert self._activations is not None, "Call calibrate() first"

        sensitivity = self.compute_sensitivity(linear.weight)
        mask = detect_outliers(sensitivity, self.outlier_fraction)

        spqr_layer = SpQRLinear(
            linear.in_features, linear.out_features,
            n_bits=self.n_bits, group_size=self.group_size,
            bias=linear.bias is not None,
        )
        spqr_layer.pack(linear.weight.data, mask)
        if linear.bias is not None:
            spqr_layer.bias.data.copy_(linear.bias.data)
        return spqr_layer
