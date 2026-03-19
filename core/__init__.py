"""llm_quant.core — Core quantization primitives."""

from .quantizer import (
    ste_round,
    compute_qparams,
    quantize_tensor,
    dequantize_tensor,
    fake_quantize,
    QuantizedLinear,
)
from .rotation import (
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
from .utils import (
    MinMaxObserver,
    ActivationCollector,
    quantization_mse,
    quantization_snr,
    kurtosis,
    make_dummy_linear,
    make_calibration_data,
)
