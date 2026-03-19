"""
llm_quant.kv_cache_quant — KV-Cache Quantization for LLM Inference.

During autoregressive decoding, the KV cache grows linearly with sequence
length and becomes the dominant memory bottleneck.  Quantizing the cached
key/value tensors to INT4/INT8 can drastically reduce memory with minimal
quality loss.

Supported granularities
-----------------------
- **Per-token** (row-wise):  each generated token has its own scale/zp.
  The scale is computed over the entire hidden dimension for that position.
  Shape: scale [B, H, T, 1] for cache [B, H, T, D].

- **Per-channel** (column-wise):  each head-dim channel has its own scale/zp
  shared across all tokens.  Better when channel ranges differ a lot.
  Shape: scale [B, H, 1, D] for cache [B, H, T, D].

- **Per-group** (token-groups):  tokens are grouped into contiguous blocks
  of ``group_size`` and each group gets its own scale/zp.  A compromise
  between per-token and per-tensor.
  Shape: scale [B, H, n_groups, D] for cache [B, H, T, D].

Key components
--------------
- ``KVCacheQuantizer``:         stateless quantize/dequantize for KV tensors
- ``QuantizedKVCache``:         drop-in replacement for the KV cache list,
                               auto-quantizes on append and dequantizes on read
- ``PerTokenKVCacheQuantizer``: convenience subclass (per-token)
- ``PerChannelKVCacheQuantizer``: convenience subclass (per-channel)
- ``PerGroupKVCacheQuantizer``: convenience subclass (per-group)

References
----------
- Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference
  with KV Cache Quantization", 2024.
- Yue et al., "WKVQuant: Quantizing Weight and Key/Value Cache for Large
  Language Models Gains More", 2024.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from enum import Enum


# ──────────────────────────────────────────────────────────────────────────────
# Quantization granularity enum
# ──────────────────────────────────────────────────────────────────────────────

class KVQuantGranularity(Enum):
    """Quantization granularity for KV cache."""
    PER_TOKEN = "per_token"         # scale per (batch, head, token)
    PER_CHANNEL = "per_channel"     # scale per (batch, head, channel)
    PER_GROUP = "per_group"         # scale per group of tokens


# ──────────────────────────────────────────────────────────────────────────────
# Core KV cache quantization/dequantization
# ──────────────────────────────────────────────────────────────────────────────

class KVCacheQuantizer:
    """Stateless quantizer for key/value cache tensors.

    Supports INT4 / INT8 with per-token, per-channel, or per-group granularity.

    Parameters
    ----------
    n_bits : int — quantization bit-width (typically 4 or 8)
    symmetric : bool — if True, use symmetric quantization
    granularity : KVQuantGranularity — per-token, per-channel, or per-group
    group_size : int — number of tokens per group (only used for PER_GROUP)

    Input format
    ------------
    Cache tensor: [B, num_heads, seq_len, head_dim]  (standard multi-head layout)
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        granularity: KVQuantGranularity = KVQuantGranularity.PER_TOKEN,
        group_size: int = 32,
    ):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.granularity = granularity
        self.group_size = group_size

        if symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1

    def _compute_qparams(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero-point for the given cache tensor.

        Parameters
        ----------
        x : Tensor [B, H, T, D]

        Returns
        -------
        scale, zero_point — broadcastable to x's shape
        """
        if self.granularity == KVQuantGranularity.PER_TOKEN:
            # Reduce over head_dim (last dim) → scale [B, H, T, 1]
            reduce_dim = -1
            keepdim = True
        elif self.granularity == KVQuantGranularity.PER_CHANNEL:
            # Reduce over seq_len (dim 2) → scale [B, H, 1, D]
            reduce_dim = -2
            keepdim = True
        elif self.granularity == KVQuantGranularity.PER_GROUP:
            # Handle in a special path
            return self._compute_qparams_grouped(x)
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")

        if self.symmetric:
            amax = x.abs().amax(dim=reduce_dim, keepdim=keepdim).clamp(min=1e-12)
            scale = amax / self.qmax
            zero_point = torch.zeros_like(scale)
        else:
            x_min = x.amin(dim=reduce_dim, keepdim=keepdim)
            x_max = x.amax(dim=reduce_dim, keepdim=keepdim)
            scale = (x_max - x_min).clamp(min=1e-12) / (self.qmax - self.qmin)
            zero_point = torch.round(-x_min / scale).clamp(self.qmin, self.qmax)

        return scale, zero_point

    def _compute_qparams_grouped(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-group qparams: group tokens into blocks."""
        B, H, T, D = x.shape
        gs = self.group_size
        n_groups = math.ceil(T / gs)
        pad = n_groups * gs - T
        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))  # pad along T dimension
        else:
            x_padded = x
        # [B, H, n_groups, gs, D]
        x_grouped = x_padded.reshape(B, H, n_groups, gs, D)

        if self.symmetric:
            amax = x_grouped.abs().amax(dim=3, keepdim=True).clamp(min=1e-12)  # [B,H,ng,1,D]
            scale = amax / self.qmax
            zero_point = torch.zeros_like(scale)
        else:
            x_min = x_grouped.amin(dim=3, keepdim=True)
            x_max = x_grouped.amax(dim=3, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-12) / (self.qmax - self.qmin)
            zero_point = torch.round(-x_min / scale).clamp(self.qmin, self.qmax)

        return scale, zero_point

    @torch.no_grad()
    def quantize(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a KV cache tensor.

        Parameters
        ----------
        x : Tensor [B, H, T, D]

        Returns
        -------
        q      : IntTensor — quantized values
        scale  : Tensor — scale factors
        zp     : Tensor — zero points
        """
        if self.granularity == KVQuantGranularity.PER_GROUP:
            return self._quantize_grouped(x)

        scale, zp = self._compute_qparams(x)
        q = torch.round(x / scale + zp).clamp(self.qmin, self.qmax)
        if self.symmetric:
            q = q.to(torch.int8)
        else:
            q = q.to(torch.int16)
        return q, scale, zp

    def _quantize_grouped(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-group quantization."""
        B, H, T, D = x.shape
        gs = self.group_size
        n_groups = math.ceil(T / gs)
        pad = n_groups * gs - T

        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x

        x_grouped = x_padded.reshape(B, H, n_groups, gs, D)
        scale, zp = self._compute_qparams_grouped(x)  # [B,H,ng,1,D]

        q = torch.round(x_grouped / scale + zp).clamp(self.qmin, self.qmax)
        q = q.reshape(B, H, n_groups * gs, D)[:, :, :T, :]
        if self.symmetric:
            q = q.to(torch.int8)
        else:
            q = q.to(torch.int16)

        return q, scale.squeeze(3), zp.squeeze(3)  # scale: [B,H,ng,D]

    @torch.no_grad()
    def dequantize(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        zp: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize a quantized KV cache tensor.

        Parameters
        ----------
        q     : IntTensor [B, H, T, D]
        scale : Tensor — broadcastable scale
        zp    : Tensor — broadcastable zero-point

        Returns
        -------
        x_deq : Tensor [B, H, T, D]
        """
        if self.granularity == KVQuantGranularity.PER_GROUP:
            return self._dequantize_grouped(q, scale, zp)

        return (q.float() - zp) * scale

    def _dequantize_grouped(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        zp: torch.Tensor,
    ) -> torch.Tensor:
        """Per-group dequantization."""
        B, H, T, D = q.shape
        gs = self.group_size
        n_groups = scale.shape[2]  # [B, H, n_groups, D]
        padded_T = n_groups * gs

        if padded_T > T:
            q_padded = F.pad(q.float(), (0, 0, 0, padded_T - T))
        else:
            q_padded = q.float()

        q_grouped = q_padded.reshape(B, H, n_groups, gs, D)
        scale_exp = scale.unsqueeze(3)  # [B, H, ng, 1, D]
        zp_exp = zp.unsqueeze(3)
        x_deq = (q_grouped - zp_exp) * scale_exp
        x_deq = x_deq.reshape(B, H, padded_T, D)[:, :, :T, :]
        return x_deq

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate quantize → dequantize on a KV cache tensor.

        This is differentiable in the sense that the scale computation
        is detached but the shape is preserved.
        """
        q, scale, zp = self.quantize(x)
        return self.dequantize(q, scale, zp)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience subclasses
# ──────────────────────────────────────────────────────────────────────────────

class PerTokenKVCacheQuantizer(KVCacheQuantizer):
    """KV cache quantizer with per-token granularity."""

    def __init__(self, n_bits: int = 8, symmetric: bool = True):
        super().__init__(n_bits=n_bits, symmetric=symmetric,
                         granularity=KVQuantGranularity.PER_TOKEN)


class PerChannelKVCacheQuantizer(KVCacheQuantizer):
    """KV cache quantizer with per-channel granularity."""

    def __init__(self, n_bits: int = 8, symmetric: bool = True):
        super().__init__(n_bits=n_bits, symmetric=symmetric,
                         granularity=KVQuantGranularity.PER_CHANNEL)


class PerGroupKVCacheQuantizer(KVCacheQuantizer):
    """KV cache quantizer with per-group-of-tokens granularity."""

    def __init__(self, n_bits: int = 8, symmetric: bool = True, group_size: int = 32):
        super().__init__(n_bits=n_bits, symmetric=symmetric,
                         granularity=KVQuantGranularity.PER_GROUP,
                         group_size=group_size)


# ──────────────────────────────────────────────────────────────────────────────
# QuantizedKVCache — drop-in replacement for the list-based KV cache
# ──────────────────────────────────────────────────────────────────────────────

class QuantizedKVCache(nn.Module):
    """A KV cache that automatically quantizes on write and dequantizes on read.

    Stores K and V as INT tensors with associated scale/zp metadata.
    Works as a drop-in for the typical ``past_key_values`` list.

    Parameters
    ----------
    num_layers : int — number of transformer layers
    quantizer  : KVCacheQuantizer — the quantizer to use
    """

    def __init__(
        self,
        num_layers: int,
        quantizer: Optional[KVCacheQuantizer] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.quantizer = quantizer or PerTokenKVCacheQuantizer(n_bits=8)

        # Storage: list of (key_q, key_scale, key_zp, val_q, val_scale, val_zp) or None
        self._cache: list[Optional[tuple]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K, V to the cache for the given layer and return full (dequantized) K, V.

        Parameters
        ----------
        layer_idx : int
        key   : Tensor [B, H, S_new, D]  — new key states
        value : Tensor [B, H, S_new, D]  — new value states

        Returns
        -------
        full_key, full_value : Tensor [B, H, S_total, D]
        """
        # Quantize the new tokens
        k_q, k_s, k_zp = self.quantizer.quantize(key)
        v_q, v_s, v_zp = self.quantizer.quantize(value)

        if self._cache[layer_idx] is None:
            self._cache[layer_idx] = (k_q, k_s, k_zp, v_q, v_s, v_zp)
        else:
            old = self._cache[layer_idx]
            # Concatenate quantized along seq_len dimension
            # Note: for per-channel/per-group, scales must also be cat'd
            cat_dim = 2  # seq_len dimension
            new_k_q = torch.cat([old[0], k_q], dim=cat_dim)
            new_v_q = torch.cat([old[3], v_q], dim=cat_dim)

            # Merge scales based on granularity
            new_k_s = self._cat_scales(old[1], k_s, cat_dim)
            new_k_zp = self._cat_scales(old[2], k_zp, cat_dim)
            new_v_s = self._cat_scales(old[4], v_s, cat_dim)
            new_v_zp = self._cat_scales(old[5], v_zp, cat_dim)

            self._cache[layer_idx] = (
                new_k_q, new_k_s, new_k_zp,
                new_v_q, new_v_s, new_v_zp,
            )

        cached = self._cache[layer_idx]
        full_key = self.quantizer.dequantize(cached[0], cached[1], cached[2])
        full_value = self.quantizer.dequantize(cached[3], cached[4], cached[5])
        return full_key, full_value

    def _cat_scales(
        self,
        old_scale: torch.Tensor,
        new_scale: torch.Tensor,
        seq_dim: int,
    ) -> torch.Tensor:
        """Concatenate scale/zp tensors along the appropriate dimension."""
        gran = self.quantizer.granularity
        if gran == KVQuantGranularity.PER_TOKEN:
            # scale [B, H, T, 1] → cat along T (dim 2)
            return torch.cat([old_scale, new_scale], dim=2)
        elif gran == KVQuantGranularity.PER_CHANNEL:
            # scale [B, H, 1, D] → keep the larger one (re-quantize would be ideal,
            # but for simplicity we use element-wise max of absmax)
            if self.quantizer.symmetric:
                return torch.maximum(old_scale, new_scale)
            else:
                # For asymmetric, min/max need recalculation; use max of scales
                return torch.maximum(old_scale, new_scale)
        elif gran == KVQuantGranularity.PER_GROUP:
            # scale [B, H, n_groups, D] → cat along group dim (dim 2)
            return torch.cat([old_scale, new_scale], dim=2)
        return torch.cat([old_scale, new_scale], dim=seq_dim)

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get dequantized K, V for a given layer, or None if empty."""
        if self._cache[layer_idx] is None:
            return None
        cached = self._cache[layer_idx]
        key = self.quantizer.dequantize(cached[0], cached[1], cached[2])
        val = self.quantizer.dequantize(cached[3], cached[4], cached[5])
        return key, val

    def reset(self) -> None:
        """Clear all cached states."""
        self._cache = [None] * self.num_layers

    @property
    def seq_len(self) -> int:
        """Current total sequence length in cache (from layer 0)."""
        if self._cache[0] is None:
            return 0
        return self._cache[0][0].shape[2]


# ──────────────────────────────────────────────────────────────────────────────
# Utility: compute compression ratio
# ──────────────────────────────────────────────────────────────────────────────

def kv_cache_memory_ratio(
    n_bits: int,
    orig_bits: int = 16,
    granularity: KVQuantGranularity = KVQuantGranularity.PER_TOKEN,
    head_dim: int = 128,
    seq_len: int = 2048,
) -> float:
    """Estimate memory compression ratio of quantized vs original KV cache.

    Returns ratio < 1 (lower is better).  Does NOT account for scale overhead.
    """
    base = n_bits / orig_bits

    # Scale overhead estimation
    if granularity == KVQuantGranularity.PER_TOKEN:
        # 1 scale per token per head: 32-bit / (head_dim * n_bits)
        overhead = 32 / (head_dim * n_bits)
    elif granularity == KVQuantGranularity.PER_CHANNEL:
        # 1 scale per channel: 32-bit / (seq_len * n_bits)
        overhead = 32 / (seq_len * n_bits)
    elif granularity == KVQuantGranularity.PER_GROUP:
        overhead = 32 / (32 * n_bits)  # group_size=32 typical
    else:
        overhead = 0

    return base + overhead
