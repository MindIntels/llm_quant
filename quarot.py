"""
llm_quant.quarot — QuaRot (Ashkboos et al., 2024).

Core idea:
    Apply a **fixed** random Hadamard rotation to activations and weights
    so that outlier channels are spread across all dimensions.  After
    rotation, simple round-to-nearest (RTN) quantization works well.

Key steps
---------
1. Insert Hadamard rotations between every (LayerNorm, Linear) pair.
2. For self-attention:  rotate Q, K before dot-product, undo on V output.
3. Fuse rotations into adjacent weight matrices (zero runtime overhead).
4. Quantize rotated weights with RTN (per-channel or per-group INT4).

This module focuses on the **rotation + quantization** logic, not the
full Transformer surgery (which is model-specific).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import List, Optional, Dict

from .core.quantizer import fake_quantize, QScheme, QuantizedLinear
from .core.rotation import (
    hadamard_matrix,
    random_hadamard_matrix,
    fast_hadamard_transform,
    apply_rotation_to_linear,
)


# ──────────────────────────────────────────────────────────────────────────────
# QuaRot rotation strategy
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def quarot_rotate_linear_pair(
    prev: nn.Module,
    linear: nn.Linear,
    R: torch.Tensor,
) -> None:
    """Fuse rotation R into a (prev, linear) pair in-place.

    For a chain ``linear(prev(x))``:
        y = W_lin @ (W_prev @ x + b_prev)
    We insert ``R^T @ R = I`` between prev and linear:
        y = (W_lin @ R^T) @ (R @ (W_prev @ x + b_prev))
    So:
        W_prev' = R @ W_prev,   b_prev' = R @ b_prev
        W_lin'  = W_lin @ R^T
    """
    dev = linear.weight.device
    R = R.to(dev, dtype=linear.weight.dtype)

    if isinstance(prev, nn.LayerNorm):
        # Can't easily fuse R into LayerNorm params.
        # For LN → Linear: absorb R^T into linear's weight only.
        # R gets applied to activations at runtime (or handled elsewhere).
        linear.weight.data = linear.weight.data @ R.t()
    elif isinstance(prev, nn.Linear):
        # prev output rotated by R:  W_prev' = R @ W_prev
        prev.weight.data = R @ prev.weight.data
        if prev.bias is not None:
            prev.bias.data = R @ prev.bias.data
        # linear input absorbs R^T:  W_lin' = W_lin @ R^T
        linear.weight.data = linear.weight.data @ R.t()
    else:
        # Default: just rotate linear's input side
        linear.weight.data = linear.weight.data @ R.t()


@torch.no_grad()
def quarot_rotate_head_pair(
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    head_dim: int,
    R_head: Optional[torch.Tensor] = None,
) -> None:
    """Apply per-head Hadamard rotation to Q/K projections.

    This spreads outliers within each attention head.

    Q' = Q @ blkdiag(R_head, ..., R_head)
    K' = K @ blkdiag(R_head, ..., R_head)
    V output absorbs R_head^T on the complementary side.
    """
    if R_head is None:
        R_head = hadamard_matrix(head_dim, dtype=q_proj.weight.dtype)
    R_head = R_head.to(q_proj.weight.device)

    num_heads_q = q_proj.out_features // head_dim
    num_heads_kv = k_proj.out_features // head_dim

    def rotate_per_head(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
        # weight shape: [out, in]
        # Reshape → [n_heads, head_dim, in], rotate output side per head
        w = weight.view(n_heads, head_dim, -1)  # [H, D, in]
        # R_head: [D, D],  multiply: R @ w_per_head → [H, D, in]
        w = torch.einsum("de,hei->hdi", R_head, w)
        return w.reshape_as(weight)

    q_proj.weight.data = rotate_per_head(q_proj.weight.data, num_heads_q)
    k_proj.weight.data = rotate_per_head(k_proj.weight.data, num_heads_kv)

    # Compensate on o_proj: o_proj sees V * Attn → need R_head^T on input side
    # o_proj.weight shape: [hidden, n_heads_q * head_dim]
    w_o = o_proj.weight.data.view(-1, num_heads_q, head_dim)  # [out, H, D]
    w_o = torch.einsum("ohd,Dd->ohD", w_o, R_head.t())       # multiply input side
    o_proj.weight.data = w_o.reshape_as(o_proj.weight.data)


# ──────────────────────────────────────────────────────────────────────────────
# QuaRot Quantizer
# ──────────────────────────────────────────────────────────────────────────────

class QuaRot:
    """QuaRot: Hadamard rotation + RTN quantization.

    Usage:
        qr = QuaRot(n_bits=4, group_size=128)
        R = qr.make_rotation(hidden_dim)
        qr.rotate_linear_pair(layer_norm, linear, R)
        q_linear = qr.quantize_linear(linear)
    """

    def __init__(
        self,
        n_bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        random_sign: bool = True,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.random_sign = random_sign

    def make_rotation(self, dim: int,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        """Create the Hadamard rotation matrix."""
        if self.random_sign:
            return random_hadamard_matrix(dim, generator=generator)
        return hadamard_matrix(dim)

    @torch.no_grad()
    def rotate_linear_pair(self, prev: nn.Module, linear: nn.Linear,
                           R: torch.Tensor) -> None:
        quarot_rotate_linear_pair(prev, linear, R)

    @torch.no_grad()
    def rotate_attention(self, q_proj, k_proj, v_proj, o_proj,
                         head_dim: int) -> None:
        quarot_rotate_head_pair(q_proj, k_proj, v_proj, o_proj, head_dim)

    @torch.no_grad()
    def quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        return QuantizedLinear.from_linear(
            linear, n_bits=self.n_bits, symmetric=self.symmetric,
            group_size=self.group_size,
        )

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fast Hadamard transform to activations at runtime."""
        return fast_hadamard_transform(x, normalize=True)

    def fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        scheme = QScheme.PER_GROUP if self.group_size > 0 else QScheme.PER_CHANNEL
        return fake_quantize(weight, self.n_bits, self.symmetric, scheme, self.group_size)
