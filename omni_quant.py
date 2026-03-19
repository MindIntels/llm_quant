"""
llm_quant.omni_quant — OmniQuant (Shao et al., 2024).

Core idea:
    Jointly optimise two lightweight "equivalent transforms" to make
    quantization easier — without changing the model's FP output:

    1. **Learnable Weight Clipping (LWC)**: per-channel clipping bounds
       ``[−α, α]`` that discard extreme weight values before quantization.
    2. **Learnable Equivalent Transform (LET)**: per-channel scaling
       (like SmoothQuant) but with learned scales optimised end-to-end.

    Both are trained with block-wise knowledge distillation on a small
    calibration set (≈128 samples, ~1 hour on 1 GPU).

Key components
--------------
- ``LearnableClipping``:   learnable per-channel clip bounds
- ``LearnableSmoothing``:  learnable smoothing scales (generalised SmoothQuant)
- ``OmniQuantBlock``:      per-block optimisation module
- ``OmniQuant``:           high-level API
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from .core.quantizer import fake_quantize, QScheme, QuantizedLinear


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Weight Clipping (LWC)
# ──────────────────────────────────────────────────────────────────────────────

class LearnableClipping(nn.Module):
    """Per-channel learnable clipping bounds for weight quantization.

    Stores log(alpha) to keep alpha > 0 via exp().
    Clips weight to [-alpha, alpha] per output channel before quantization.
    """

    def __init__(self, out_features: int, init_val: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.full((out_features, 1), fill_value=torch.tensor(init_val).log().item())
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()  # [out, 1]
        return torch.min(torch.max(weight, -alpha), alpha)

    def get_alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Equivalent Transform (LET)
# ──────────────────────────────────────────────────────────────────────────────

class LearnableSmoothing(nn.Module):
    """Per-channel learnable smoothing scales (generalised SmoothQuant).

    Stores log(s) to keep s > 0.  Applied as:
        x_smooth = x / s
        w_smooth = w * s  (column-wise)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        s = self.log_scale.exp()
        x_s = x / s                    # [..., C] / [C]
        w_s = weight * s.unsqueeze(0)   # [out, C] * [1, C]
        return x_s, w_s

    def get_scales(self) -> torch.Tensor:
        return self.log_scale.exp().detach()


# ──────────────────────────────────────────────────────────────────────────────
# OmniQuant Block Optimiser
# ──────────────────────────────────────────────────────────────────────────────

class OmniQuantBlock(nn.Module):
    """Per-linear-layer optimisation module.

    Jointly learns clipping + smoothing to minimise block-wise
    reconstruction error under quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_bits: int = 4,
        group_size: int = 128,
        use_lwc: bool = True,
        use_let: bool = True,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size

        self.lwc = LearnableClipping(out_features) if use_lwc else None
        self.let = LearnableSmoothing(in_features) if use_let else None

    def forward(self, weight: torch.Tensor, x: torch.Tensor):
        """Apply transforms + fake quantize.  Return (y_q, loss)."""
        w = weight
        act = x

        # 1. Learnable smoothing
        if self.let is not None:
            act, w = self.let(act, w)

        # 2. Learnable clipping
        if self.lwc is not None:
            w = self.lwc(w)

        # 3. Fake quantize weight
        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        w_q = fake_quantize(w, self.n_bits, True, scheme, self.group_size)

        y_ref = x @ weight.t()
        y_q = act @ w_q.t()

        loss = ((y_ref - y_q) ** 2).mean()
        return y_q, loss


# ──────────────────────────────────────────────────────────────────────────────
# OmniQuant High-Level API
# ──────────────────────────────────────────────────────────────────────────────

class OmniQuant:
    """OmniQuant: learnable clipping + smoothing + quantization.

    Usage:
        oq = OmniQuant(n_bits=4, group_size=128)
        results = oq.optimise(linear.weight, calib_data, steps=200)
        oq.apply_transforms(linear, results)
        q_linear = oq.quantize_linear(linear)
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        lr: float = 1e-2,
        use_lwc: bool = True,
        use_let: bool = True,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.lr = lr
        self.use_lwc = use_lwc
        self.use_let = use_let

    def optimise(
        self,
        weight: torch.Tensor,
        calibration_data: List[torch.Tensor],
        steps: int = 200,
    ) -> dict:
        """Optimise clipping + smoothing for a single weight matrix.

        Returns dict with keys:
            'scales': smoothing scales (if LET)
            'clip_alpha': clipping bounds (if LWC)
        """
        in_f = weight.shape[1]
        out_f = weight.shape[0]
        block = OmniQuantBlock(in_f, out_f, self.n_bits, self.group_size,
                               self.use_lwc, self.use_let)
        block = block.to(weight.device)
        optimizer = torch.optim.Adam(block.parameters(), lr=self.lr)
        weight_f = weight.detach().float()

        for step in range(steps):
            idx = step % len(calibration_data)
            x = calibration_data[idx].to(weight.device).float()
            if x.ndim == 2:
                x = x.unsqueeze(0)
            optimizer.zero_grad()
            _, loss = block(weight_f, x)
            loss.backward()
            optimizer.step()

        results = {}
        if block.let is not None:
            results["scales"] = block.let.get_scales()
        if block.lwc is not None:
            results["clip_alpha"] = block.lwc.get_alpha()
        return results

    @torch.no_grad()
    def apply_transforms(
        self,
        linear: nn.Linear,
        results: dict,
        prev_layer: Optional[nn.Module] = None,
    ) -> None:
        """Apply learned transforms to the linear layer in-place."""
        if "scales" in results:
            s = results["scales"].to(linear.weight.device)
            linear.weight.data.mul_(s.unsqueeze(0))
            if prev_layer is not None:
                if isinstance(prev_layer, nn.LayerNorm):
                    prev_layer.weight.data.div_(s)
                    if prev_layer.bias is not None:
                        prev_layer.bias.data.div_(s)

        if "clip_alpha" in results:
            alpha = results["clip_alpha"].to(linear.weight.device)
            linear.weight.data.clamp_(-alpha, alpha)

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        return QuantizedLinear.from_linear(
            linear, n_bits=self.n_bits, symmetric=self.symmetric,
            group_size=self.group_size,
        )
