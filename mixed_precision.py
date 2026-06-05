"""
Mixed-precision quantization.

Most PTQ methods pick one bit-width and apply it to the whole layer. But output
channels don't all matter equally - if you already know which ones carry the
signal (from a sensitivity scan, an activation study, whatever you trust), you
can spend more bits on those and fewer on the rest. The average bit-width stays
low while the channels that actually move the output keep their precision.

You decide what "important" means and hand it in. Either:
  - a per-channel importance score, and the top `protect_frac` get protected, or
  - an explicit boolean mask of which channels to protect.

Protection is per output channel (rows of the weight), which lines up with the
per-channel scale used elsewhere in the library.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .core.quantizer import compute_qparams, quantize_tensor, QScheme


def select_protected(
    out_features: int,
    importance: Optional[torch.Tensor] = None,
    protect_mask: Optional[torch.Tensor] = None,
    protect_frac: float = 0.01,
) -> torch.Tensor:
    """Decide which output channels to keep at high precision.

    Returns a boolean tensor of shape [out_features]. If `protect_mask` is given
    it wins. Otherwise the top `protect_frac` of channels by `importance` are
    marked (at least one channel always survives).
    """
    if protect_mask is not None:
        mask = protect_mask.flatten().bool()
        if mask.numel() != out_features:
            raise ValueError(
                f"protect_mask has {mask.numel()} entries, expected {out_features}"
            )
        return mask

    if importance is None:
        raise ValueError("pass either importance or protect_mask")

    importance = importance.flatten()
    if importance.numel() != out_features:
        raise ValueError(
            f"importance has {importance.numel()} entries, expected {out_features}"
        )
    if not 0.0 < protect_frac <= 1.0:
        raise ValueError(f"protect_frac must be in (0, 1], got {protect_frac}")

    k = max(1, round(out_features * protect_frac))
    mask = torch.zeros(out_features, dtype=torch.bool, device=importance.device)
    top = torch.topk(importance, k).indices
    mask[top] = True
    return mask


def _quantize_per_channel(weight, n_bits, symmetric):
    # quantize every row at one bit-width; scale/zp come back as [out, 1]
    scale, zp = compute_qparams(weight, n_bits, symmetric, QScheme.PER_CHANNEL)
    q = quantize_tensor(weight, scale, zp, n_bits, symmetric)
    return q, scale, zp


class MixedPrecisionLinear(nn.Module):
    """Linear layer where each output channel can have its own bit-width.

    Weights are stored as int8 (same as the other quantized layers here - the
    sub-8-bit rows just don't use the full range) together with a per-row scale,
    zero-point and a record of how many bits each row was quantized to.
    """

    def __init__(self, in_features, out_features, symmetric=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symmetric

        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1))
        self.register_buffer("zero_point", torch.zeros(out_features, 1))
        self.register_buffer("bits", torch.zeros(out_features, dtype=torch.int8))
        self.register_buffer("protect_mask", torch.zeros(out_features, dtype=torch.bool))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def pack(self, q, scale, zp, bits, mask):
        self.weight_int.copy_(q.to(torch.int8))
        self.scale.copy_(scale)
        self.zero_point.copy_(zp)
        self.bits.copy_(bits.to(torch.int8))
        self.protect_mask.copy_(mask)

    def forward(self, x):
        w = (self.weight_int.float() - self.zero_point) * self.scale
        return F.linear(x, w, self.bias)

    @property
    def effective_bits(self) -> float:
        """Average bit-width across the layer (channel-count weighted)."""
        return self.bits.float().mean().item()

    def extra_repr(self):
        n_prot = int(self.protect_mask.sum())
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"protected={n_prot}/{self.out_features}, "
            f"effective_bits={self.effective_bits:.2f}"
        )


class MixedPrecision:
    """Quantize a Linear, protecting important output channels at higher bits.

        mp = MixedPrecision(bits_high=8, bits_low=4, protect_frac=0.01)
        ml = mp.quantize_linear(linear, importance=scores)   # top 1% -> int8
        # or:
        ml = mp.quantize_linear(linear, protect_mask=mask)   # mask -> int8
    """

    def __init__(self, bits_high=8, bits_low=4, symmetric=True, protect_frac=0.01):
        if bits_high < bits_low:
            raise ValueError(
                f"bits_high ({bits_high}) must be >= bits_low ({bits_low})"
            )
        self.bits_high = bits_high
        self.bits_low = bits_low
        self.symmetric = symmetric
        self.protect_frac = protect_frac

    @torch.no_grad()
    def quantize_linear(self, linear, importance=None, protect_mask=None):
        weight = linear.weight.data
        out_features, in_features = weight.shape

        mask = select_protected(out_features, importance, protect_mask, self.protect_frac)
        mask = mask.to(weight.device)
        m = mask.view(-1, 1)

        # quantize the whole matrix at both bit-widths, then keep each row from
        # whichever tier it belongs to. wasteful by a constant factor but dead
        # simple and obviously correct.
        q_hi, s_hi, z_hi = _quantize_per_channel(weight, self.bits_high, self.symmetric)
        q_lo, s_lo, z_lo = _quantize_per_channel(weight, self.bits_low, self.symmetric)

        q = torch.where(m, q_hi, q_lo)
        scale = torch.where(m, s_hi, s_lo)
        zp = torch.where(m, z_hi, z_lo)
        bits = torch.where(mask, torch.full_like(mask, self.bits_high, dtype=torch.long),
                           torch.full_like(mask, self.bits_low, dtype=torch.long))

        ml = MixedPrecisionLinear(
            in_features, out_features,
            symmetric=self.symmetric,
            bias=linear.bias is not None,
        )
        ml.pack(q, scale, zp, bits, mask)
        if linear.bias is not None:
            ml.bias.data.copy_(linear.bias.data)
        return ml
