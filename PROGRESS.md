# SeZM MoE Progress

This file tracks implementation and validation status. `SPEC.md` remains the design contract; update this file when a step is implemented, tested, or blocked.

## Snapshot

- Date: 2026-05-18
- Current phase: Phase A
- Current step: Step 2 (`MoESO2Router`)
- Overall status: Steps 1-2 implementations exist and matching tests pass; Steps 3-10 are not implemented.

## Implemented Files

- `deepmd/pt/model/descriptor/sezm_nn/moe/a2a_ops.py`
- `deepmd/pt/model/descriptor/sezm_nn/moe/router.py`
- `deepmd/pt/model/descriptor/sezm_nn/moe/__init__.py`
- `source/tests/pt/test_sezm_moe_a2a.py`
- `source/tests/pt/test_sezm_moe_a2a_multigpu.py`
- `source/tests/pt/test_sezm_moe_router.py`

## Validation

- Environment smoke test: PASS
  - Python: `/mnt/data_nas/zhangd/conda_env/torch-modern/bin/python`
  - Python version: 3.11.14
  - PyTorch: 2.10.0+cu126
  - CUDA available: true
  - CUDA device count: 8
  - `deepmd.pt` import: PASS
  - `dp --version`: `DeePMD-kit v1.3.3.dev0`
- Single-process Step 1 tests: PASS
  - Command used: `pytest source/tests/pt/test_sezm_moe_a2a.py -q`
  - Result: 5 tests passed, 3 subtests passed
- Multi-process Step 1 8-rank CUDA/NCCL test: PASS
  - Command shape: `torchrun --nproc_per_node=8 ... source/tests/pt/test_sezm_moe_a2a_multigpu.py`
  - Result: 6 tests passed on all 8 ranks, no hang
- Step 1 ruff check: PASS
  - Command used: `/root/miniconda3/bin/ruff check deepmd/pt/model/descriptor/sezm_nn/moe/a2a_ops.py source/tests/pt/test_sezm_moe_a2a.py source/tests/pt/test_sezm_moe_a2a_multigpu.py`
- Single-process Step 2 router tests: PASS
  - Command used: `PATH="/mnt/data_nas/zhangd/conda_env/torch-modern/bin:$PATH" pytest source/tests/pt/test_sezm_moe_router.py -xvs`
  - Result: 7 tests passed, 2 deprecation warnings from `torch.jit.script`
- Step 2 ruff check: PASS
  - Command used: `/root/miniconda3/bin/ruff check deepmd/pt/model/descriptor/sezm_nn/moe/router.py source/tests/pt/test_sezm_moe_router.py`
- DPA3 reference subagent smoke test: PASS
  - Cursor `dpa3-ref-searcher` can read `deepmd-kit-moe` reference files.
- Implementer subagent smoke test: PASS
  - Cursor `sezm-moe-implementer` can read `SPEC.md`, `PROGRESS.md`, and project rules in read-only mode.

## Environment Notes

- `pytest` 9.0.3 is installed in `/mnt/data_nas/zhangd/conda_env/torch-modern`.
- `pytest` is not currently on the shell `PATH`; use `/mnt/data_nas/zhangd/conda_env/torch-modern/bin/python -m pytest ...` when needed.
- `/root/miniconda3/bin/ruff` is available and reports `ruff 0.15.6`.
- Existing Step 1 tests are runnable via `pytest`, `unittest`, and standalone `torchrun`.
- Multi-rank tests use CUDA/NCCL when CUDA is available and fall back to CPU/Gloo only when CUDA is unavailable.

## Not Started

- Step 3: `MoESO2ExpertCollection`
- Step 4: `MoESO2Convolution`
- Step 5: `SO2Convolution` MoE branch and validation
- Step 6: EP/DP groups and gradient sync
- Step 7: `SeZMInteractionBlock` integration
- Step 8: `DescrptSeZM` top-level config
- Step 9: training loop gradient sync
- Step 10: checkpoint resharding
- Phase D end-to-end validation

## Next Recommended Actions

1. Proceed to Step 3 (`MoESO2ExpertCollection`) with `sezm-moe-implementer`.
1. Keep updating this file after each Step's tests and ruff checks.
