"""
llm_quant — LLM quantization methods library.

An independent, self-contained library implementing state-of-the-art
LLM post-training quantization (PTQ) algorithms.

Supported methods
-----------------
- **SmoothQuant** (W8A8):  activation smoothing + symmetric quantization
- **QuaRot** (W4):         fixed Hadamard rotation + RTN quantization
- **SpinQuant** (W4):      learnable orthogonal rotation + RTN
- **FlatQuant** (W4):      learnable affine flatten + quantization
- **OmniQuant** (W4):      learnable clipping + smoothing
- **AWQ** (W4):            activation-aware per-channel scaling + group quant
- **SpQR** (W3/W4):        sparse outlier + group-quantized hybrid
- **GPTQ** (W4):           second-order Hessian-based column-wise quantization
- **QLoRA** (NF4+LoRA):    4-bit NormalFloat base + trainable low-rank adapters
- **KV Cache Quant**:      per-token / per-channel / per-group cache quantization

Usage
-----
    import llm_quant

    # SmoothQuant example
    sq = llm_quant.SmoothQuant(n_bits=8)
    sq.calibrate(activations)
    sq.smooth(layer_norm, linear)
    q_linear = sq.quantize_linear(linear)

    # AWQ example
    awq = llm_quant.AWQ(n_bits=4, group_size=128)
    awq.calibrate(activations)
    scales = awq.search_scales(linear.weight)
    awq.apply_scales(prev_layer, linear, scales)
    q_linear = awq.quantize_linear(linear)
"""

# ── Core primitives ────────────────────────────────────────────────────────
from .core.quantizer import (
    QScheme,
    ste_round,
    compute_qparams,
    quantize_tensor,
    dequantize_tensor,
    fake_quantize,
    QuantizedLinear,
)
from .core.rotation import (
    hadamard_matrix,
    random_hadamard_matrix,
    random_orthogonal,
    cayley_transform,
    CayleyRotation,
    rotate_weight_right,
    rotate_weight_left,
    apply_rotation_to_linear,
    fast_hadamard_transform,
)
from .core.utils import (
    MinMaxObserver,
    ActivationCollector,
    quantization_mse,
    quantization_snr,
    kurtosis,
    make_dummy_linear,
    make_calibration_data,
)

# ── Quantization methods ──────────────────────────────────────────────────
from .smooth_quant import SmoothQuant, compute_smooth_scales, collect_act_max
from .quarot import QuaRot
from .spin_quant import SpinQuant, SpinQuantRotation
from .flat_quant import FlatQuant, FlatTransform, FlatQuantSolver
from .omni_quant import OmniQuant, LearnableClipping, LearnableSmoothing
from .awq import AWQ, compute_saliency, awq_grid_search
from .spqr import SpQR, SpQRLinear, compute_weight_sensitivity, detect_outliers
from .gptq import GPTQ, GPTQLinear, compute_hessian, gptq_quantize
from .qlora import (
    QLoRA, QLoRALinear, NF4_TABLE,
    nf4_quantize, nf4_dequantize,
    double_quantize, double_dequantize,
)
from .kv_cache_quant import (
    KVQuantGranularity, KVCacheQuantizer, QuantizedKVCache,
    PerTokenKVCacheQuantizer, PerChannelKVCacheQuantizer,
    PerGroupKVCacheQuantizer, kv_cache_memory_ratio,
)
from .mixed_precision import MixedPrecision, MixedPrecisionLinear, select_protected

__all__ = [
    # Core
    "QScheme", "ste_round", "compute_qparams",
    "quantize_tensor", "dequantize_tensor", "fake_quantize",
    "QuantizedLinear",
    # Rotation
    "hadamard_matrix", "random_hadamard_matrix", "random_orthogonal",
    "cayley_transform", "CayleyRotation",
    "rotate_weight_right", "rotate_weight_left",
    "apply_rotation_to_linear", "fast_hadamard_transform",
    # Utils
    "MinMaxObserver", "ActivationCollector",
    "quantization_mse", "quantization_snr", "kurtosis",
    "make_dummy_linear", "make_calibration_data",
    # Methods
    "SmoothQuant", "compute_smooth_scales", "collect_act_max",
    "QuaRot",
    "SpinQuant", "SpinQuantRotation",
    "FlatQuant", "FlatTransform", "FlatQuantSolver",
    "OmniQuant", "LearnableClipping", "LearnableSmoothing",
    "AWQ", "compute_saliency", "awq_grid_search",
    "SpQR", "SpQRLinear", "compute_weight_sensitivity", "detect_outliers",
    # GPTQ
    "GPTQ", "GPTQLinear", "compute_hessian", "gptq_quantize",
    # QLoRA
    "QLoRA", "QLoRALinear", "NF4_TABLE",
    "nf4_quantize", "nf4_dequantize",
    "double_quantize", "double_dequantize",
    # KV Cache
    "KVQuantGranularity", "KVCacheQuantizer", "QuantizedKVCache",
    "PerTokenKVCacheQuantizer", "PerChannelKVCacheQuantizer",
    "PerGroupKVCacheQuantizer", "kv_cache_memory_ratio",
    # Mixed precision
    "MixedPrecision", "MixedPrecisionLinear", "select_protected",
]
