"""
llm_quant.core.utils — Shared observers, calibration helpers, and metrics.

Provides:
- MinMaxObserver:       running min/max tracker for activation ranges
- ActivationCollector:  hook-based collector for calibration data
- quantization_mse:     MSE between original and fake-quantized tensor
- quantization_snr:     signal-to-noise ratio of quantization
- kurtosis:             per-channel excess kurtosis (outlier detection)
- make_dummy_linear:    utility for creating test Linear layers
- make_calibration_data: generate random calibration batches
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional, Dict


# ──────────────────────────────────────────────────────────────────────────────
# Observers
# ──────────────────────────────────────────────────────────────────────────────

class MinMaxObserver:
    """Tracks running min / max across calibration batches.

    Usage:
        obs = MinMaxObserver()
        for batch in calib_data:
            obs.update(batch)
        min_val, max_val = obs.get_range()
    """

    def __init__(self, per_channel: bool = False, channel_dim: int = -1):
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor) -> None:
        if self.per_channel:
            reduce_dims = [i for i in range(x.ndim) if i != self.channel_dim % x.ndim]
            x_min = x.amin(dim=reduce_dims)
            x_max = x.amax(dim=reduce_dims)
        else:
            x_min = x.min()
            x_max = x.max()

        if self.min_val is None:
            self.min_val = x_min.clone()
            self.max_val = x_max.clone()
        else:
            self.min_val = torch.minimum(self.min_val, x_min)
            self.max_val = torch.maximum(self.max_val, x_max)

    def get_range(self):
        """Return (min_val, max_val)."""
        assert self.min_val is not None, "No data observed yet"
        return self.min_val, self.max_val

    def reset(self) -> None:
        self.min_val = None
        self.max_val = None


# ──────────────────────────────────────────────────────────────────────────────
# Activation Collector
# ──────────────────────────────────────────────────────────────────────────────

class ActivationCollector:
    """Collect input/output activations of specified layers using hooks.

    Usage:
        model = ...
        collector = ActivationCollector(model, ['layers.0.ffn', 'layers.1.ffn'])
        for batch in data:
            model(batch)
        acts = collector.get_inputs()   # dict[name] -> list[Tensor]
        collector.remove_hooks()
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.inputs: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        self.outputs: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        self._hooks = []
        for name in layer_names:
            module = dict(model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, inp, out):
            self.inputs[name].append(inp[0].detach().cpu())
            if isinstance(out, torch.Tensor):
                self.outputs[name].append(out.detach().cpu())
        return hook_fn

    def get_inputs(self) -> Dict[str, List[torch.Tensor]]:
        return self.inputs

    def get_outputs(self) -> Dict[str, List[torch.Tensor]]:
        return self.outputs

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def quantization_mse(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Mean squared error between original and quantized tensors."""
    return ((original - quantized) ** 2).mean()


def quantization_snr(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Signal-to-Quantization-Noise Ratio in dB.

    SNR = 10 * log10( ||x||² / ||x - x̂||² )
    """
    signal_power = (original ** 2).sum()
    noise_power = ((original - quantized) ** 2).sum().clamp(min=1e-12)
    return 10.0 * torch.log10(signal_power / noise_power)


def kurtosis(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Excess kurtosis along *dim*.

    High kurtosis → outlier-prone channels → benefit from smoothing.
    """
    mu = x.mean(dim=dim, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=dim, keepdim=True).clamp(min=1e-12)
    m4 = ((x - mu) ** 4).mean(dim=dim, keepdim=True)
    kurt = m4 / (var ** 2) - 3.0
    return kurt.squeeze(dim)


# ──────────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_dummy_linear(
    in_features: int = 768,
    out_features: int = 768,
    bias: bool = True,
) -> nn.Linear:
    """Create a ``nn.Linear`` with deterministic weights for testing."""
    torch.manual_seed(42)
    linear = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(linear.weight, std=0.02)
    if bias:
        nn.init.zeros_(linear.bias)
    return linear


def make_calibration_data(
    batch_size: int = 4,
    seq_len: int = 64,
    hidden: int = 768,
    n_batches: int = 8,
    seed: int = 0,
) -> List[torch.Tensor]:
    """Generate random calibration data (list of activation tensors)."""
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch_size, seq_len, hidden, generator=g)
        for _ in range(n_batches)
    ]
