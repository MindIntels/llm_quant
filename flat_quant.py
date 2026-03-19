"""
llm_quant.flat_quant — FlatQuant (Sun et al., 2024).

Core idea:
    Learn per-layer affine transforms (rotation + scaling) that "flatten"
    the weight/activation distributions before quantization, making them
    more uniform and easier to quantize.

    For each Linear layer:
        Y = X · W^T
    becomes:
        Y = (X · R_x · D_x) · (D_w · R_w · W)^T

    where R_x, R_w are orthogonal (Cayley), D_x, D_w are diagonal scales.
    The transforms are jointly optimised to minimise quantization error on
    calibration data.

Key components
--------------
- ``FlatTransform``:   per-layer learnable (rotation + diag scale)
- ``FlatQuantSolver``: grid search + gradient descent for optimal transforms
- ``FlatQuant``:       high-level API for the full pipeline
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from .core.quantizer import fake_quantize, QScheme, QuantizedLinear
from .core.rotation import CayleyRotation


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Flat Transform
# ──────────────────────────────────────────────────────────────────────────────

class FlatTransform(nn.Module):
    """Per-layer affine transform: ``T = D · Cayley(A)``.

    *D* is a learnable positive diagonal (stored as log-scale).
    *Cayley(A)* is a learnable orthogonal rotation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.rotation = CayleyRotation(dim, init="identity")
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self) -> torch.Tensor:
        """Return the transform matrix ``D @ R``, shape [dim, dim]."""
        R = self.rotation()           # [dim, dim] orthogonal
        D = self.log_scale.exp()      # [dim]
        return D.unsqueeze(1) * R     # diag(D) @ R

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ──────────────────────────────────────────────────────────────────────────────
# FlatQuant Solver
# ──────────────────────────────────────────────────────────────────────────────

class FlatQuantSolver(nn.Module):
    """Optimise per-layer flat transforms for a single Linear layer.

    Learns ``T_act`` (activation-side) and ``T_wt`` (weight-side) transforms
    that minimise:
        L = || X @ W^T  −  FQ(X @ T_act) @ FQ(T_wt @ W)^T ||²
    """

    def __init__(self, in_features: int, out_features: int,
                 n_bits: int = 4, group_size: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features

        self.t_act = FlatTransform(in_features)
        self.t_wt = FlatTransform(in_features)  # applied on input-dim of W

    def forward(self, weight: torch.Tensor, x: torch.Tensor):
        """Compute quantized output and reconstruction loss.

        Parameters
        ----------
        weight : [out, in]
        x : [B, S, in]
        """
        T_act = self.t_act()  # [in, in]
        T_wt = self.t_wt()   # [in, in]

        # Transform activation and weight
        x_flat = x @ T_act          # [B, S, in]
        w_flat = weight @ T_wt      # [out, in]

        # Fake quantize
        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        x_q = fake_quantize(x_flat, self.n_bits, True, QScheme.PER_TENSOR)
        w_q = fake_quantize(w_flat, self.n_bits, True, scheme, self.group_size)

        # Compute outputs
        y_ref = x @ weight.t()                    # reference
        y_q = x_q @ w_q.t()                       # quantized

        loss = ((y_ref - y_q) ** 2).mean()
        return y_q, loss


# ──────────────────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────────────────

class FlatQuant:
    """FlatQuant: learnable affine flatten + quantization.

    Usage:
        fq = FlatQuant(n_bits=4, group_size=128)
        T_act, T_wt = fq.optimise(linear.weight, calib_data, steps=200)
        fq.apply_transforms(linear, T_act, T_wt)
        q_linear = fq.quantize_linear(linear)
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        lr: float = 1e-2,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.lr = lr

    def optimise(
        self,
        weight: torch.Tensor,
        calibration_data: List[torch.Tensor],
        steps: int = 200,
    ):
        """Learn optimal transforms for *weight*.

        Returns (T_act, T_wt) as [in, in] matrices.
        """
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        solver = FlatQuantSolver(in_features, out_features,
                                 self.n_bits, self.group_size)
        solver = solver.to(weight.device)
        optimizer = torch.optim.Adam(solver.parameters(), lr=self.lr)
        weight_f = weight.detach().float()

        for step in range(steps):
            idx = step % len(calibration_data)
            x = calibration_data[idx].to(weight.device).float()
            if x.ndim == 2:
                x = x.unsqueeze(0)
            optimizer.zero_grad()
            _, loss = solver(weight_f, x)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            T_act = solver.t_act()
            T_wt = solver.t_wt()
        return T_act, T_wt

    @torch.no_grad()
    def apply_transforms(
        self,
        linear: nn.Linear,
        T_act: torch.Tensor,
        T_wt: torch.Tensor,
    ) -> None:
        """Fuse weight-side transform into linear: ``W' = W @ T_wt``."""
        linear.weight.data = linear.weight.data @ T_wt.to(linear.weight.device)

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        return QuantizedLinear.from_linear(
            linear, n_bits=self.n_bits, symmetric=self.symmetric,
            group_size=self.group_size,
        )
