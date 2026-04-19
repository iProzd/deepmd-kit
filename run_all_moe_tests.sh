#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Run all MoE E2E tests (single-GPU + multi-GPU)

set -e

echo "============================================================"
echo "MoE End-to-End Test Suite"
echo "============================================================"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /mnt/data_nas/zhangd/conda_env/claude-moe

# Single-GPU tests
echo ">>> Running single-GPU regression tests..."
CUDA_VISIBLE_DEVICES=0 python -m pytest \
    source/tests/pt/test_repflow_moe.py \
    source/tests/pt/test_repflows_moe_integration.py \
    source/tests/pt/test_moe_checkpoint.py \
    -v --tb=short
echo ""

echo ">>> Running single-GPU E2E smoke test..."
CUDA_VISIBLE_DEVICES=0 python -m pytest \
    source/tests/pt/test_moe_e2e_training.py \
    -v -s
echo ""

echo ">>> Running single-GPU dp test (inference validation)..."
CUDA_VISIBLE_DEVICES=0 python -m pytest \
    source/tests/pt/test_moe_e2e_dptest.py \
    -v -s
echo ""

# Multi-GPU tests (4 GPU)
echo ">>> Running 4-GPU checkpoint tests..."
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29520 \
    source/tests/pt/test_moe_checkpoint_multigpu.py
echo ""

echo ">>> Running 4-GPU full correctness tests..."
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29521 \
    source/tests/pt/test_moe_full_multigpu.py
echo ""

# Multi-GPU tests (8 GPU)
echo ">>> Running 8-GPU EP+DP training test..."
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29522 \
    source/tests/pt/test_moe_e2e_training_multigpu.py
echo ""

echo ">>> Running 8-GPU full correctness tests..."
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29523 \
    source/tests/pt/test_moe_full_multigpu.py
echo ""

echo ">>> Running 8-GPU speed benchmark..."
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:29524 \
    source/tests/pt/test_moe_e2e_speed.py
echo ""

echo "============================================================"
echo "All MoE E2E tests completed successfully!"
echo "============================================================"
