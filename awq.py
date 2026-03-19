"""
llm_quant.awq — AWQ: Activation-Aware Weight Quantization (Lin et al., 2024).

Core idea:
    Not all weight channels are equally important. 1% of "salient" channels
    (determined by activation magnitude) contribute most to output quality.
    Protect them with per-channel scaling, then quantize all weights with
    group-wise INT4.

Key insight:
    Scale salient channels UP before quantization (reducing their relative
    quantization error) and compensate by scaling activations DOWN.
    Optimal scale: s* = argmin || Q(s·W) · (x/s) − W·x ||²

    Solved via grid search over a discrete set of candidate scales.

Key components
--------------
- ``compute_saliency``:  rank channels by activation magnitude
- ``awq_grid_search``:   find optimal per-channel scales via grid search
- ``AWQ``:               high-level API
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from .core.quantizer import fake_quantize, QScheme, QuantizedLinear


# ──────────────────────────────────────────────────────────────────────────────
# Saliency computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_saliency(
    activations: List[torch.Tensor],
) -> torch.Tensor:
    """Compute per-channel saliency as mean absolute activation.

    Parameters
    ----------
    activations : list of [B, S, C] or [B, C] tensors.

    Returns
    -------
    saliency : [C]  higher = more important.
    """
    total = None
    count = 0
    for act in activations:
        flat = act.reshape(-1, act.shape[-1])
        batch_mean = flat.abs().mean(dim=0)
        if total is None:
            total = batch_mean
        else:
            total = total + batch_mean
        count += 1
    return total / count


# ──────────────────────────────────────────────────────────────────────────────
# Grid search for optimal per-channel scales
# ──────────────────────────────────────────────────────────────────────────────

def awq_grid_search(
    weight: torch.Tensor,
    activations: List[torch.Tensor],
    saliency: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
) -> torch.Tensor:
    """Find optimal per-channel scaling factors via grid search.

    For each candidate alpha in [0, 1], compute
        s = saliency^alpha
    then evaluate quantization error of ``Q(diag(s) @ W)`` on calibration data.

    Parameters
    ----------
    weight : [out, in]
    activations : list of [B, S, in]
    saliency : [in]
    n_grid : number of alpha candidates.

    Returns
    -------
    best_scales : [in]
    """
    best_error = float("inf")
    best_scales = torch.ones_like(saliency)

    # Stack calibration data
    x_cat = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)

    scheme = QScheme.PER_GROUP if group_size > 0 else QScheme.PER_CHANNEL

    for i in range(n_grid + 1):
        alpha = i / n_grid
        scales = saliency.pow(alpha).clamp(min=1e-4)
        scales = scales / scales.mean()  # normalise to avoid magnitude shift

        # Scale weight up, activation down
        w_scaled = weight * scales.unsqueeze(0)          # [out, in]
        x_scaled = x_cat / scales.unsqueeze(0)           # [N, in]

        # Fake quantize weight
        w_q = fake_quantize(w_scaled, n_bits, True, scheme, group_size)

        # Compute reconstruction error
        y_ref = x_cat @ weight.t()
        y_q = x_scaled @ w_q.t()
        err = ((y_ref - y_q) ** 2).mean().item()

        if err < best_error:
            best_error = err
            best_scales = scales.clone()

    return best_scales


# ──────────────────────────────────────────────────────────────────────────────
# AWQ High-Level API
# ──────────────────────────────────────────────────────────────────────────────

class AWQ:
    """AWQ: Activation-Aware Weight Quantization.

    Usage:
        awq = AWQ(n_bits=4, group_size=128)
        awq.calibrate(activations)
        scales = awq.search_scales(linear.weight)
        awq.apply_scales(prev_layer, linear, scales)
        q_linear = awq.quantize_linear(linear)
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        n_grid: int = 20,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.n_grid = n_grid
        self._activations: Optional[List[torch.Tensor]] = None
        self._saliency: Optional[torch.Tensor] = None

    def calibrate(self, activations: List[torch.Tensor]) -> None:
        """Store calibration activations and compute saliency."""
        self._activations = activations
        self._saliency = compute_saliency(activations)

    @property
    def saliency(self) -> torch.Tensor:
        assert self._saliency is not None, "Call calibrate() first"
        return self._saliency

    def search_scales(self, weight: torch.Tensor) -> torch.Tensor:
        """Grid-search optimal per-channel scales for *weight*."""
        assert self._activations is not None, "Call calibrate() first"
        return awq_grid_search(
            weight, self._activations, self.saliency,
            self.n_bits, self.group_size, self.n_grid,
        )

    @torch.no_grad()
    def apply_scales(
        self,
        prev_layer: Optional[nn.Module],
        linear: nn.Linear,
        scales: torch.Tensor,
    ) -> None:
        """Apply scaling in-place: scale weight up, prev_layer down."""
        s = scales.to(linear.weight.device)
        linear.weight.data.mul_(s.unsqueeze(0))

        if prev_layer is not None:
            if isinstance(prev_layer, nn.LayerNorm):
                prev_layer.weight.data.div_(s)
                if prev_layer.bias is not None:
                    prev_layer.bias.data.div_(s)
            elif isinstance(prev_layer, nn.Linear):
                prev_layer.weight.data.div_(s.unsqueeze(0))
                if prev_layer.bias is not None:
                    prev_layer.bias.data.div_(s)

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        return QuantizedLinear.from_linear(
            linear, n_bits=self.n_bits, symmetric=self.symmetric,
            group_size=self.group_size,
        )
