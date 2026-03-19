"""
llm_quant.smooth_quant — SmoothQuant (Xiao et al., 2023).

Core idea:
    Activation outliers make weight-only quantization fail.
    SmoothQuant migrates the quantization difficulty from activations
    to weights by a per-channel scaling factor:

        Y = (X · diag(s)^{-1}) · (diag(s) · W^T)

    where s_j = max|X_j|^α / max|W_j|^{1-α}  balances outlier magnitude.

Workflow
--------
1. **Calibration**: collect per-channel activation max across a few batches.
2. **Compute smoothing scales** ``s`` with migration strength ``alpha``.
3. **Apply smoothing**: scale the preceding LayerNorm (or Linear) and the
   current Linear in-place.
4. **Quantize** both activations and weights with simple per-tensor /
   per-channel W8A8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from .core.quantizer import fake_quantize, QScheme, QuantizedLinear, compute_qparams, quantize_tensor, dequantize_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Smoothing Scale Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_smooth_scales(
    act_max: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute per-channel smoothing scales.

    Parameters
    ----------
    act_max : [C]  per-channel max of absolute activations from calibration.
    weight  : [out, C]  weight matrix of the target Linear layer.
    alpha   : migration strength (0 → all on weight, 1 → all on activation).

    Returns
    -------
    scales : [C]  smoothing factors.
    """
    w_max = weight.abs().amax(dim=0).clamp(min=1e-12)
    act_max = act_max.clamp(min=1e-12)
    scales = (act_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-12)
    return scales


# ──────────────────────────────────────────────────────────────────────────────
# Apply Smoothing
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def smooth_linear_pair(
    prev_layer: nn.Module,
    linear: nn.Linear,
    scales: torch.Tensor,
) -> None:
    """Apply smoothing scales in-place.

    ``prev_layer`` is typically a LayerNorm or another Linear whose output
    feeds ``linear``.

    Transforms:
        prev_layer.weight /= scales   (or prev_layer.gamma /= scales)
        prev_layer.bias   /= scales   (if present)
        linear.weight     *= scales   (column-wise)
    """
    s = scales.to(linear.weight.device)

    # Scale the previous layer's output
    if isinstance(prev_layer, nn.LayerNorm):
        prev_layer.weight.data.div_(s)
        if prev_layer.bias is not None:
            prev_layer.bias.data.div_(s)
    elif isinstance(prev_layer, nn.Linear):
        # Scale output channels of previous linear
        prev_layer.weight.data.div_(s.unsqueeze(0))
        if prev_layer.bias is not None:
            prev_layer.bias.data.div_(s)
    else:
        raise TypeError(f"Unsupported prev_layer type: {type(prev_layer)}")

    # Scale the current linear's input channels
    linear.weight.data.mul_(s.unsqueeze(0))  # broadcast over out_features


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def collect_act_max(
    activations: List[torch.Tensor],
) -> torch.Tensor:
    """Compute per-channel max(|activation|) across calibration batches.

    Parameters
    ----------
    activations : list of tensors, each ``[B, S, C]`` or ``[B, C]``.

    Returns
    -------
    act_max : [C]
    """
    amax = None
    for act in activations:
        # Flatten batch dims → [*, C]
        flat = act.reshape(-1, act.shape[-1])
        batch_max = flat.abs().amax(dim=0)
        if amax is None:
            amax = batch_max
        else:
            amax = torch.maximum(amax, batch_max)
    return amax


# ──────────────────────────────────────────────────────────────────────────────
# Full SmoothQuant Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class SmoothQuant:
    """SmoothQuant: smooth + quantize a (prev_layer, linear) pair.

    Usage:
        sq = SmoothQuant(n_bits=8, alpha=0.5)
        sq.calibrate(activations)                  # list[Tensor]
        sq.smooth(layer_norm, linear)              # in-place smoothing
        q_linear = sq.quantize_linear(linear)      # QuantizedLinear
    """

    def __init__(self, n_bits: int = 8, alpha: float = 0.5,
                 symmetric: bool = True, per_channel: bool = True):
        self.n_bits = n_bits
        self.alpha = alpha
        self.symmetric = symmetric
        self.per_channel = per_channel
        self._act_max: Optional[torch.Tensor] = None

    def calibrate(self, activations: List[torch.Tensor]) -> None:
        """Collect activation statistics."""
        self._act_max = collect_act_max(activations)

    @property
    def act_max(self) -> torch.Tensor:
        assert self._act_max is not None, "Call calibrate() first"
        return self._act_max

    def compute_scales(self, weight: torch.Tensor) -> torch.Tensor:
        return compute_smooth_scales(self.act_max, weight, self.alpha)

    @torch.no_grad()
    def smooth(self, prev_layer: nn.Module, linear: nn.Linear) -> torch.Tensor:
        """Compute and apply smoothing in-place.  Returns the scales."""
        scales = self.compute_scales(linear.weight)
        smooth_linear_pair(prev_layer, linear, scales)
        return scales

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        """Quantize a (already smoothed) Linear to QuantizedLinear."""
        return QuantizedLinear.from_linear(
            linear, n_bits=self.n_bits, symmetric=self.symmetric,
        )

    def fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate INT8 activation quantization."""
        scheme = QScheme.PER_CHANNEL if self.per_channel else QScheme.PER_TENSOR
        return fake_quantize(x, self.n_bits, self.symmetric, scheme)
