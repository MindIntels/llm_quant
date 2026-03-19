#!/usr/bin/env bash
# Run all tests for the llm_quant project.
set -e

PYTHON=${PYTHON:-python3}

echo "======================================"
echo "  llm_quant — Quantization Test Suite"
echo "======================================"

echo ""
echo ">>> Core: Quantizer"
$PYTHON -m pytest llm_quant/tests/test_quantizer.py -v

echo ""
echo ">>> Core: Rotation"
$PYTHON -m pytest llm_quant/tests/test_rotation.py -v

echo ""
echo ">>> Core: Utils"
$PYTHON -m pytest llm_quant/tests/test_utils.py -v

echo ""
echo ">>> SmoothQuant"
$PYTHON -m pytest llm_quant/tests/test_smooth_quant.py -v

echo ""
echo ">>> QuaRot"
$PYTHON -m pytest llm_quant/tests/test_quarot.py -v

echo ""
echo ">>> SpinQuant"
$PYTHON -m pytest llm_quant/tests/test_spin_quant.py -v

echo ""
echo ">>> FlatQuant"
$PYTHON -m pytest llm_quant/tests/test_flat_quant.py -v

echo ""
echo ">>> OmniQuant"
$PYTHON -m pytest llm_quant/tests/test_omni_quant.py -v

echo ""
echo ">>> AWQ"
$PYTHON -m pytest llm_quant/tests/test_awq.py -v

echo ""
echo ">>> SpQR"
$PYTHON -m pytest llm_quant/tests/test_spqr.py -v

echo ""
echo "======================================"
echo "  All llm_quant tests passed!"
echo "======================================"
