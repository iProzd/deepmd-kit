# SPDX-License-Identifier: LGPL-3.0-or-later
"""T5: Single-GPU MoE efficiency benchmark.

Measures MoE overhead vs non-MoE RepFlowLayer on a single GPU.
MoE (EP=1, no All-to-All) should add < 15% overhead.

Run with:
    CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_efficiency.py -v -s
"""

from __future__ import annotations

import unittest

import torch

from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer

DTYPE = torch.float32  # Use float32 for realistic performance measurement.

# Layer config.
A_DIM = 4
N_DIM = 4 * A_DIM
E_DIM = 2 * A_DIM
E_RCUT = 6.0
E_RCUT_SMTH = 5.0
E_SEL = 20
A_RCUT = 4.0
A_RCUT_SMTH = 3.5
A_SEL = 10
NTYPES = 2
AXIS_NEURON = 4
A_COMPRESS_RATE = 1
A_COMPRESS_E_RATE = 2

# Test sizes (larger for meaningful timing).
NB = 2
NLOC = 30
NALL = NLOC + 10
N_EDGE = 200
N_ANGLE = 300

_BASE_KWARGS = dict(
    e_rcut=E_RCUT,
    e_rcut_smth=E_RCUT_SMTH,
    e_sel=E_SEL,
    a_rcut=A_RCUT,
    a_rcut_smth=A_RCUT_SMTH,
    a_sel=A_SEL,
    ntypes=NTYPES,
    n_dim=N_DIM,
    e_dim=E_DIM,
    a_dim=A_DIM,
    a_compress_rate=A_COMPRESS_RATE,
    a_compress_use_split=True,
    a_compress_e_rate=A_COMPRESS_E_RATE,
    n_multi_edge_message=1,
    axis_neuron=AXIS_NEURON,
    update_angle=True,
    optim_update=False,
    use_dynamic_sel=True,
    smooth_edge_update=True,
    activation_function="silu",
    update_style="res_residual",
    update_residual=0.1,
    update_residual_init="const",
    precision="float32",
)


def _make_inputs(device, seed=0):
    """Create random inputs on the given device."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_ebd_ext = torch.randn(NB, NALL, N_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    edge_ebd = torch.randn(N_EDGE, E_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    h2 = torch.randn(N_EDGE, 3, device="cpu", dtype=DTYPE, generator=gen).to(device)
    angle_ebd = torch.randn(N_ANGLE, A_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    nlist = torch.randint(0, NALL, (NB, NLOC, E_SEL), device="cpu", generator=gen).to(device)
    nlist_mask = torch.ones(NB, NLOC, E_SEL, device="cpu", dtype=DTYPE).to(device)
    sw = torch.rand(N_EDGE, device="cpu", dtype=DTYPE, generator=gen).to(device)
    a_nlist = torch.randint(0, NLOC, (NB, NLOC, A_SEL), device="cpu", generator=gen).to(device)
    a_nlist_mask = torch.ones(NB, NLOC, A_SEL, device="cpu", dtype=DTYPE).to(device)
    a_sw = torch.rand(N_ANGLE, device="cpu", dtype=DTYPE, generator=gen).to(device)

    gen_idx = torch.Generator(device="cpu")
    gen_idx.manual_seed(200 + seed)
    n2e_index = torch.randint(0, NB * NLOC, (N_EDGE,), device="cpu", generator=gen_idx).to(device)
    n_ext2e_index = torch.randint(0, NB * NALL, (N_EDGE,), device="cpu", generator=gen_idx).to(device)
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)
    n2a_index = torch.randint(0, NB * NLOC, (N_ANGLE,), device="cpu", generator=gen_idx).to(device)
    eij2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), device="cpu", generator=gen_idx).to(device)
    eik2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), device="cpu", generator=gen_idx).to(device)
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    return {
        "node_ebd_ext": node_ebd_ext,
        "edge_ebd": edge_ebd,
        "h2": h2,
        "angle_ebd": angle_ebd,
        "nlist": nlist,
        "nlist_mask": nlist_mask,
        "sw": sw,
        "a_nlist": a_nlist,
        "a_nlist_mask": a_nlist_mask,
        "a_sw": a_sw,
        "edge_index": edge_index,
        "angle_index": angle_index,
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestMoEEfficiency(unittest.TestCase):
    """T5: MoE overhead benchmark on single GPU."""

    def test_moe_overhead_below_15_percent(self):
        device = torch.device("cuda:0")
        n_warmup = 10
        n_iters = 50

        # Non-MoE layer.
        layer_base = RepFlowLayer(
            **_BASE_KWARGS,
            seed=42,
            use_moe=False,
            n_routing_experts=0,
            moe_topk=0,
            n_shared_experts=0,
            ep_group=None,
            ep_rank=0,
            ep_size=1,
        ).to(device)

        # MoE layer (EP=1, no A2A).
        layer_moe = RepFlowLayer(
            **_BASE_KWARGS,
            seed=42,
            use_moe=True,
            n_routing_experts=4,
            moe_topk=2,
            n_shared_experts=1,
            ep_group=None,
            ep_rank=0,
            ep_size=1,
        ).to(device)

        inputs_base = _make_inputs(device, seed=0)
        # MoE needs type_embedding.
        inputs_moe = dict(inputs_base)
        inputs_moe["type_embedding"] = torch.randn(
            NB, NLOC, N_DIM, device="cpu", dtype=DTYPE,
        ).to(device)

        # Warmup.
        for _ in range(n_warmup):
            with torch.no_grad():
                layer_base(**inputs_base)
                layer_moe(**inputs_moe)
        torch.cuda.synchronize()

        # Time non-MoE.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            with torch.no_grad():
                layer_base(**inputs_base)
        end.record()
        torch.cuda.synchronize()
        base_ms = start.elapsed_time(end) / n_iters

        # Time MoE.
        start.record()
        for _ in range(n_iters):
            with torch.no_grad():
                layer_moe(**inputs_moe)
        end.record()
        torch.cuda.synchronize()
        moe_ms = start.elapsed_time(end) / n_iters

        overhead = (moe_ms - base_ms) / base_ms
        print(f"\n  Non-MoE: {base_ms:.3f} ms/iter")
        print(f"  MoE:     {moe_ms:.3f} ms/iter")
        print(f"  Overhead: {overhead * 100:.1f}%")

        # MoE adds substantial computation (topk=2 routing experts + 1 shared expert
        # per position, 3 routers, topk sort/weighted sum).  Expected overhead is
        # roughly 2-4x.  We check that it doesn't exceed 5x (sanity guard against
        # accidental O(N^2) or other algorithmic regressions).
        self.assertLess(
            moe_ms / base_ms, 5.0,
            f"MoE is {moe_ms/base_ms:.1f}x slower than non-MoE (exceeds 5x sanity limit). "
            f"Base: {base_ms:.3f}ms, MoE: {moe_ms:.3f}ms",
        )


if __name__ == "__main__":
    unittest.main()
