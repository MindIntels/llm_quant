"""
llm_quant.spin_quant — SpinQuant (Liu et al., 2024).

Core idea:
    Replace QuaRot's **fixed** Hadamard with **learnable** orthogonal rotations
    trained to minimise quantization error.  Uses the Cayley parameterisation
    to stay on the orthogonal manifold during gradient descent.

Key steps
---------
1. Initialise rotation R = Cayley(A) with A = 0 (identity) or from Hadamard.
2. For each calibration batch, compute:
        loss = || W·x  −  Deq(Q(R^T · W · R)) · R^T · x ||²
   and backprop through Cayley(A).
3. After optimisation, fuse the learned R into weights (same as QuaRot).
4. RTN-quantize the rotated weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from .core.quantizer import fake_quantize, QScheme, QuantizedLinear
from .core.rotation import CayleyRotation, apply_rotation_to_linear


# ──────────────────────────────────────────────────────────────────────────────
# SpinQuant Rotation Optimiser
# ──────────────────────────────────────────────────────────────────────────────

class SpinQuantRotation(nn.Module):
    """Learnable orthogonal rotation for a single weight matrix.

    Wraps ``CayleyRotation`` and provides a loss that measures
    the quantization error of the rotated weight.
    """

    def __init__(self, dim: int, n_bits: int = 4, group_size: int = 128):
        super().__init__()
        self.rotation = CayleyRotation(dim, init="identity")
        self.n_bits = n_bits
        self.group_size = group_size

    def forward(self, weight: torch.Tensor, x: Optional[torch.Tensor] = None):
        """Compute rotated-and-quantized weight, return (W_q, R).

        If *x* is provided, also returns the reconstruction loss.
        """
        R = self.rotation()  # orthogonal matrix
        W_rot = weight @ R   # rotate columns

        # Fake quantize (differentiable via STE)
        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        W_q = fake_quantize(W_rot, self.n_bits, True, scheme, self.group_size)

        if x is not None:
            # x: [B, S, in_features]
            y_ref = x @ weight.t()  # original output
            y_q = (x @ R) @ W_q.t()  # quantized output (note the R on activation)
            loss = ((y_ref - y_q) ** 2).mean()
            return W_q, R, loss

        return W_q, R


# ──────────────────────────────────────────────────────────────────────────────
# SpinQuant Solver
# ──────────────────────────────────────────────────────────────────────────────

class SpinQuant:
    """SpinQuant: learnable rotation + RTN quantization.

    Usage:
        sq = SpinQuant(n_bits=4, group_size=128)
        R = sq.optimise_rotation(linear.weight, calibration_data, steps=200)
        sq.apply_rotation(linear, R)
        q_linear = sq.quantize_linear(linear)
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

    def optimise_rotation(
        self,
        weight: torch.Tensor,
        calibration_data: List[torch.Tensor],
        steps: int = 200,
    ) -> torch.Tensor:
        """Learn the best orthogonal rotation for *weight*.

        Parameters
        ----------
        weight : [out, in]  float weight matrix (detached).
        calibration_data : list of [B, S, in] activation tensors.
        steps : optimisation steps.

        Returns
        -------
        R : [in, in] optimised orthogonal matrix.
        """
        dim = weight.shape[1]
        rot_module = SpinQuantRotation(dim, self.n_bits, self.group_size)
        rot_module = rot_module.to(weight.device)

        optimizer = torch.optim.Adam(rot_module.parameters(), lr=self.lr)

        weight_f = weight.detach().float()

        for step in range(steps):
            idx = step % len(calibration_data)
            x = calibration_data[idx].to(weight.device).float()
            if x.ndim == 2:
                x = x.unsqueeze(0)

            optimizer.zero_grad()
            _, _, loss = rot_module(weight_f, x)
            loss.backward()
            optimizer.step()

        # Extract final rotation
        with torch.no_grad():
            R = rot_module.rotation()
        return R

    @torch.no_grad()
    def apply_rotation(self, linear: nn.Linear, R: torch.Tensor) -> None:
        """Fuse learned rotation into weight in-place."""
        apply_rotation_to_linear(linear, R, side="right")

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        return QuantizedLinear.from_linear(
            linear, n_bits=self.n_bits, symmetric=self.symmetric,
            group_size=self.group_size,
        )
