# LLM Quantization Methods Library

An independent, self-contained library implementing 10 state-of-the-art
LLM post-training quantization (PTQ) and fine-tuning algorithms.

## Supported Methods

| Method | Paper | Bit-width | Key Idea |
|--------|-------|-----------|----------|
| **SmoothQuant** | Xiao et al., 2023 | W8A8 | Per-channel activation smoothing to balance outliers |
| **QuaRot** | Ashkboos et al., 2024 | W4 | Fixed Hadamard rotation to spread outliers + RTN |
| **SpinQuant** | Liu et al., 2024 | W4 | Learnable Cayley rotation optimised for quantization |
| **FlatQuant** | Sun et al., 2024 | W4 | Learnable affine flatten (rotation + scaling) |
| **OmniQuant** | Shao et al., 2024 | W4 | Learnable clipping (LWC) + smoothing (LET) |
| **AWQ** | Lin et al., 2024 | W4 | Activation-aware per-channel scaling via grid search |
| **SpQR** | Dettmers et al., 2024 | W3/W4 | Sparse FP outliers + group-quantized INT hybrid |
| **GPTQ** | Frantar et al., 2023 | W4 | Second-order Hessian-based column-wise OBQ |
| **QLoRA** | Dettmers et al., 2023 | NF4+LoRA | 4-bit NormalFloat base + trainable low-rank adapters |
| **KV Cache Quant** | Hooper et al., 2024 | INT4/INT8 | Per-token / per-channel / per-group KV cache quantization |

## Project Structure

```
llm_quant/
├── __init__.py              # Top-level exports (import llm_quant)
├── requirements.txt         # torch + pytest
├── run_tests.sh             # Run full test suite
├── README.md
│
├── core/                    # Shared primitives
│   ├── __init__.py
│   ├── quantizer.py         # Quantize / dequantize / fake_quantize / QuantizedLinear
│   ├── rotation.py          # Hadamard, Cayley, fast_hadamard_transform
│   └── utils.py             # Observers, metrics (MSE, SNR, kurtosis), test helpers
│
├── smooth_quant.py          # SmoothQuant
├── quarot.py                # QuaRot
├── spin_quant.py            # SpinQuant
├── flat_quant.py            # FlatQuant
├── omni_quant.py            # OmniQuant
├── awq.py                   # AWQ
├── spqr.py                  # SpQR
├── gptq.py                  # GPTQ
├── qlora.py                 # QLoRA (NF4 + LoRA)
├── kv_cache_quant.py        # KV cache quantization
│
└── tests/                   # 13 test files
    ├── test_quantizer.py
    ├── test_rotation.py
    ├── test_utils.py
    ├── test_smooth_quant.py
    ├── test_quarot.py
    ├── test_spin_quant.py
    ├── test_flat_quant.py
    ├── test_omni_quant.py
    ├── test_awq.py
    └── test_spqr.py
    ├── test_gptq.py
    ├── test_qlora.py
    └── test_kv_cache_quant.py
```

## Installation

```bash
pip install torch pytest
```

Only requires **PyTorch ≥ 2.0** and **pytest ≥ 7.0**.  No other dependencies.

## Quick Start

```python
import llm_quant

# ── SmoothQuant (W8A8) ────────────────────────────────
sq = llm_quant.SmoothQuant(n_bits=8, alpha=0.5)
sq.calibrate(activations)               # list[Tensor]
sq.smooth(layer_norm, linear)            # in-place smoothing
q_linear = sq.quantize_linear(linear)    # QuantizedLinear

# ── AWQ (W4, group=128) ───────────────────────────────
awq = llm_quant.AWQ(n_bits=4, group_size=128)
awq.calibrate(activations)
scales = awq.search_scales(linear.weight)
awq.apply_scales(prev_layer, linear, scales)
q_linear = awq.quantize_linear(linear)

# ── QuaRot (W4) ───────────────────────────────────────
qr = llm_quant.QuaRot(n_bits=4, group_size=128)
R = qr.make_rotation(hidden_dim)
qr.rotate_linear_pair(layer_norm, linear, R)
q_linear = qr.quantize_linear(linear)

# ── SpinQuant (W4, learnable rotation) ────────────────
spin = llm_quant.SpinQuant(n_bits=4, group_size=128)
R = spin.optimise_rotation(linear.weight, calib_data, steps=200)
spin.apply_rotation(linear, R)
q_linear = spin.quantize_linear(linear)

# ── FlatQuant (W4, learnable affine) ──────────────────
fq = llm_quant.FlatQuant(n_bits=4, group_size=128)
T_act, T_wt = fq.optimise(linear.weight, calib_data, steps=200)
fq.apply_transforms(linear, T_act, T_wt)
q_linear = fq.quantize_linear(linear)

# ── OmniQuant (W4, learnable clip + smooth) ───────────
oq = llm_quant.OmniQuant(n_bits=4, group_size=128)
results = oq.optimise(linear.weight, calib_data, steps=200)
oq.apply_transforms(linear, results)
q_linear = oq.quantize_linear(linear)

# ── SpQR (W4, sparse outliers) ────────────────────────
spqr = llm_quant.SpQR(n_bits=4, group_size=128, outlier_fraction=0.01)
spqr.calibrate(activations)
spqr_linear = spqr.quantize_linear(linear)  # SpQRLinear

# ── GPTQ (W4, Hessian-based) ──────────────────────────
gptq = llm_quant.GPTQ(n_bits=4, group_size=128)
gptq.calibrate(activations)
q_linear = gptq.quantize_linear(linear)     # GPTQLinear

# ── QLoRA (NF4 + LoRA fine-tuning) ────────────────────
qlora = llm_quant.QLoRA(rank=16, group_size=64)
ql = qlora.wrap_linear(linear)               # QLoRALinear
optimizer = torch.optim.Adam(qlora.trainable_params(ql), lr=1e-4)
# ... fine-tune loop ...

# ── KV Cache Quantization ─────────────────────────────
from llm_quant import PerTokenKVCacheQuantizer, QuantizedKVCache
cache = QuantizedKVCache(num_layers=32,
            quantizer=PerTokenKVCacheQuantizer(n_bits=4))
full_k, full_v = cache.update(layer_idx=0, key=k, value=v)
```

## Core Utilities

```python
from llm_quant import (
    fake_quantize,        # differentiable q→dq simulation
    QuantizedLinear,      # INT weight Linear layer
    hadamard_matrix,      # Sylvester Hadamard construction
    CayleyRotation,       # learnable orthogonal rotation
    fast_hadamard_transform,  # O(d log d) Hadamard
    MinMaxObserver,       # running min/max tracker
    quantization_mse,     # MSE metric
    quantization_snr,     # SNR in dB
    kurtosis,             # outlier detection metric
)
```

## Run Tests

```bash
# Run all tests
bash llm_quant/run_tests.sh

# Run specific method
python -m pytest llm_quant/tests/test_awq.py -v
python -m pytest llm_quant/tests/test_quarot.py -v
```

## Method Comparison Guide

| Criterion | SmoothQuant | QuaRot | SpinQuant | FlatQuant | OmniQuant | AWQ | SpQR | GPTQ | QLoRA | KV Cache |
|-----------|:-----------:|:------:|:---------:|:---------:|:---------:|:---:|:----:|:----:|:-----:|:--------:|
| Calibration needed | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Optimisation steps | 0 | 0 | ~200 | ~200 | ~200 | 0 | 0 | 0 | Fine-tune | 0 |
| Runtime overhead | None | FHT | None | Transform | None | None | Sparse scatter | None | LoRA matmul | Dequant |
| Typical bit-width | W8A8 | W4 | W4 | W4 | W4 | W4 | W3-W4 | W4 | NF4 | INT4/8 |
| Handles act outliers | ✓✓✓ | ✓✓ | ✓✓ | ✓✓ | ✓✓ | ✓✓ | ✓ | ✓✓ | — | — |
| Handles wt outliers | ✓ | ✓✓ | ✓✓✓ | ✓✓✓ | ✓✓ | ✓✓ | ✓✓✓ | ✓✓✓ | — | — |
