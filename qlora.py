"""
llm_quant.qlora — QLoRA: Efficient Fine-Tuning of Quantized LLMs
                  (Dettmers et al., 2023).

Core idea:
    1. Quantize the pre-trained weight matrix to NF4 (4-bit NormalFloat)
       — a data-type whose quantization levels are optimal for normally-
       distributed weights.
    2. Optionally apply **double quantization**: quantize the FP32 scale
       factors themselves to FP8 to further reduce memory.
    3. Attach trainable LoRA adapters (low-rank A, B matrices) on top of
       the frozen quantized backbone.
    4. During fine-tuning, gradients flow only through the LoRA path.

Key components
--------------
- ``nf4_quantize``:        quantize float weights to NF4 representation
- ``nf4_dequantize``:      dequantize NF4 back to float
- ``double_quantize``:     quantize FP32 scale factors to FP8
- ``double_dequantize``:   restore FP32 scales from FP8
- ``QLoRALinear``:         frozen NF4 base + trainable LoRA adapters
- ``QLoRA``:               high-level API for applying QLoRA to a model

References
----------
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs",
  NeurIPS 2023.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# NF4 lookup table
# ──────────────────────────────────────────────────────────────────────────────

def _build_nf4_table() -> torch.Tensor:
    """Build the 16-level NormalFloat4 lookup table.

    The levels are the quantiles of a standard normal N(0,1) at
    positions 1/(2*16), 3/(2*16), ..., 31/(2*16), rescaled to [-1, 1].
    """
    # 16 quantile points equally spaced in probability
    nf4 = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000,
    ])
    return nf4


NF4_TABLE = _build_nf4_table()


# ──────────────────────────────────────────────────────────────────────────────
# NF4 quantization / dequantization
# ──────────────────────────────────────────────────────────────────────────────

def nf4_quantize(
    x: torch.Tensor,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to NF4 (4-bit NormalFloat).

    Parameters
    ----------
    x : Tensor of any shape (typically 2-D weight [out, in])
    group_size : int — block size for per-group absmax scaling

    Returns
    -------
    codes : LongTensor — NF4 indices in [0, 15], same shape as x
    scales : Tensor — per-group absmax scaling factors
    """
    table = NF4_TABLE.to(x.device)
    orig_shape = x.shape
    x_flat = x.reshape(-1)

    # Pad to multiple of group_size
    n = x_flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        x_flat = F.pad(x_flat, (0, pad))

    # Reshape into groups
    x_grouped = x_flat.reshape(-1, group_size)  # [n_groups, group_size]

    # Per-group absmax scale
    scales = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [n_groups, 1]

    # Normalize to [-1, 1]
    x_norm = x_grouped / scales  # [n_groups, group_size]

    # Find nearest NF4 level
    # x_norm: [n_groups, group_size, 1] vs table: [16]
    diffs = (x_norm.unsqueeze(-1) - table.unsqueeze(0).unsqueeze(0)).abs()
    codes = diffs.argmin(dim=-1)  # [n_groups, group_size]

    # Reshape back and trim padding
    codes = codes.reshape(-1)[:n].reshape(orig_shape)
    scales = scales.squeeze(-1)  # [n_groups]

    return codes.long(), scales


def nf4_dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 64,
    orig_shape: Optional[tuple] = None,
) -> torch.Tensor:
    """Dequantize NF4 codes back to float.

    Parameters
    ----------
    codes  : LongTensor — NF4 indices in [0, 15]
    scales : Tensor [n_groups]
    group_size : int
    orig_shape : optional target shape

    Returns
    -------
    x_deq : Tensor — dequantized values
    """
    table = NF4_TABLE.to(scales.device, dtype=scales.dtype)
    shape = codes.shape
    codes_flat = codes.reshape(-1)
    n = codes_flat.numel()

    # Pad
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        codes_flat = F.pad(codes_flat, (0, pad))

    codes_grouped = codes_flat.reshape(-1, group_size)  # [n_groups, group_size]

    # Lookup
    values = table[codes_grouped]  # [n_groups, group_size]

    # Re-scale
    x_deq = values * scales.unsqueeze(-1)  # [n_groups, group_size]

    # Trim and reshape
    x_deq = x_deq.reshape(-1)[:n]
    if orig_shape is not None:
        x_deq = x_deq.reshape(orig_shape)
    else:
        x_deq = x_deq.reshape(shape)

    return x_deq


# ──────────────────────────────────────────────────────────────────────────────
# Double quantization (quantize the FP32 scales to FP8)
# ──────────────────────────────────────────────────────────────────────────────

def double_quantize(
    scales: torch.Tensor,
    dq_group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 scale factors to 8-bit (simulated FP8).

    Parameters
    ----------
    scales : Tensor [n_groups] — first-level absmax scales
    dq_group_size : int — how many scales per second-level group

    Returns
    -------
    q_scales  : Tensor [n_groups] — quantized scale indices (uint8 range)
    dq_scales : Tensor [n_dq_groups] — second-level scale factors
    """
    n = scales.numel()
    pad = (dq_group_size - n % dq_group_size) % dq_group_size
    if pad > 0:
        s_padded = F.pad(scales, (0, pad))
    else:
        s_padded = scales
    s_grouped = s_padded.reshape(-1, dq_group_size)  # [n_dq_groups, dq_group_size]

    # Per-group absmax
    dq_scales = s_grouped.abs().amax(dim=-1).clamp(min=1e-12)  # [n_dq_groups]

    # Quantize to 8-bit symmetric: range [-127, 127]
    s_norm = s_grouped / dq_scales.unsqueeze(-1)
    q_scales = torch.round(s_norm * 127.0).clamp(-127, 127)

    q_scales = q_scales.reshape(-1)[:n]
    return q_scales, dq_scales


def double_dequantize(
    q_scales: torch.Tensor,
    dq_scales: torch.Tensor,
    dq_group_size: int = 256,
) -> torch.Tensor:
    """Restore FP32 scales from double-quantized representation.

    Parameters
    ----------
    q_scales  : Tensor [n_groups] — quantized 8-bit scale indices
    dq_scales : Tensor [n_dq_groups] — second-level scale factors

    Returns
    -------
    scales : Tensor [n_groups] — restored FP32 scales
    """
    n = q_scales.numel()
    pad = (dq_group_size - n % dq_group_size) % dq_group_size
    if pad > 0:
        q_padded = F.pad(q_scales, (0, pad))
    else:
        q_padded = q_scales
    q_grouped = q_padded.reshape(-1, dq_group_size)
    scales = (q_grouped / 127.0) * dq_scales.unsqueeze(-1)
    return scales.reshape(-1)[:n]


# ──────────────────────────────────────────────────────────────────────────────
# QLoRALinear — frozen NF4 base + trainable LoRA
# ──────────────────────────────────────────────────────────────────────────────

class QLoRALinear(nn.Module):
    """Linear layer with frozen NF4 weights and trainable LoRA adapters.

    forward(x) = NF4_dequant(W_base) @ x + (B @ A) @ x * (alpha / r)

    Parameters
    ----------
    in_features, out_features : int
    rank : int — LoRA rank (r)
    alpha : float — LoRA scaling factor
    group_size : int — NF4 block/group size
    double_quant : bool — whether to double-quantize scales
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        group_size: int = 64,
        double_quant: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.group_size = group_size
        self.double_quant = double_quant
        self.scaling = alpha / rank

        # NF4 base (frozen)
        self.register_buffer("nf4_codes", torch.zeros(out_features, in_features, dtype=torch.long))
        self.register_buffer("nf4_scales", torch.ones(1))  # placeholder
        if double_quant:
            self.register_buffer("dq_q_scales", torch.ones(1))
            self.register_buffer("dq_scales", torch.ones(1))
            self.register_buffer("dq_group_size_val", torch.tensor(256, dtype=torch.long))

        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def pack_base(self, weight: torch.Tensor) -> None:
        """Quantize and store the base weight in NF4 format."""
        codes, scales = nf4_quantize(weight, self.group_size)
        self.nf4_codes = codes
        self.nf4_scales = scales

        if self.double_quant:
            dq_gs = self.dq_group_size_val.item()
            q_scales, dq_scales = double_quantize(scales, dq_group_size=dq_gs)
            self.dq_q_scales = q_scales
            self.dq_scales = dq_scales

    def _dequantize_base(self) -> torch.Tensor:
        """Dequantize the NF4 base weight back to float."""
        if self.double_quant and hasattr(self, "dq_q_scales"):
            dq_gs = self.dq_group_size_val.item()
            scales = double_dequantize(self.dq_q_scales, self.dq_scales, dq_gs)
        else:
            scales = self.nf4_scales
        return nf4_dequantize(
            self.nf4_codes, scales, self.group_size,
            orig_shape=(self.out_features, self.in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen NF4)
        w_base = self._dequantize_base()
        out = F.linear(x, w_base, self.bias)
        # LoRA path (trainable)
        out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out

    def merge_lora(self) -> torch.Tensor:
        """Return the effective merged weight W_eff = W_base + scaling * B @ A."""
        w_base = self._dequantize_base()
        return w_base + self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"group={self.group_size}, double_quant={self.double_quant}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────────────────

class QLoRA:
    """High-level QLoRA pipeline.

    Usage
    -----
        qlora = QLoRA(rank=16, group_size=64)
        ql = qlora.wrap_linear(linear)           # → QLoRALinear
        # Fine-tune: only lora_A, lora_B have requires_grad=True
        optimizer = torch.optim.Adam(ql.parameters(), lr=1e-4)
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        group_size: int = 64,
        double_quant: bool = True,
    ):
        self.rank = rank
        self.alpha = alpha
        self.group_size = group_size
        self.double_quant = double_quant

    def wrap_linear(self, linear: nn.Linear) -> QLoRALinear:
        """Wrap an nn.Linear as a QLoRALinear (NF4 base + LoRA)."""
        ql = QLoRALinear(
            linear.in_features,
            linear.out_features,
            rank=self.rank,
            alpha=self.alpha,
            group_size=self.group_size,
            double_quant=self.double_quant,
            bias=linear.bias is not None,
        )
        ql.pack_base(linear.weight.data)
        if linear.bias is not None:
            ql.bias.data.copy_(linear.bias.data)
        return ql

    def trainable_params(self, module: nn.Module) -> list[nn.Parameter]:
        """Return only the trainable (LoRA) parameters."""
        return [p for p in module.parameters() if p.requires_grad]
