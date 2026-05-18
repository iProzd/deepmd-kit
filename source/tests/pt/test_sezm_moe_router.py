# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the SeZM MoE SO(2) router."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.moe.router import (
    MoESO2Router,
)


def _make_router(
    input_dim: int = 64,
    n_routing_experts: int = 32,
    topk: int = 4,
    routing_input: str = "dst",
) -> MoESO2Router:
    with torch.device("cpu"):
        router = MoESO2Router(
            input_dim=input_dim,
            n_routing_experts=n_routing_experts,
            topk=topk,
            routing_input=routing_input,
            precision="float64",
            seed=20260518,
        )
    return router.to("cpu")


def test_unsupported_routing_input_raises() -> None:
    with pytest.raises(ValueError, match="routing_input"):
        _make_router(routing_input="center")


@pytest.mark.parametrize("topk", [0, 33])
def test_invalid_topk_raises(topk: int) -> None:
    with pytest.raises(ValueError, match="1 <= topk <= n_routing_experts"):
        _make_router(n_routing_experts=32, topk=topk)


def test_output_shape() -> None:
    router = _make_router(input_dim=64, n_routing_experts=32, topk=4)
    type_emb_per_edge = torch.randn(
        100, 64, dtype=torch.float64, device="cpu"
    )  # (E, input_dim)

    topk_weights, topk_indices = router(type_emb_per_edge)

    assert topk_weights.shape == (100, 4)
    assert topk_indices.shape == (100, 4)


def test_softmax_normalization() -> None:
    router = _make_router(input_dim=64, n_routing_experts=32, topk=4)
    type_emb_per_edge = torch.randn(
        100, 64, dtype=torch.float64, device="cpu"
    )  # (E, input_dim)

    topk_weights, _ = router(type_emb_per_edge)  # (E, topk)
    row_sums = topk_weights.sum(dim=-1)  # (E,)

    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(topk_weights >= 0.0)


def test_indices_in_range() -> None:
    n_routing_experts = 32
    router = _make_router(input_dim=64, n_routing_experts=n_routing_experts, topk=4)
    type_emb_per_edge = torch.randn(
        100, 64, dtype=torch.float64, device="cpu"
    )  # (E, input_dim)

    _, topk_indices = router(type_emb_per_edge)  # (E, topk)

    assert torch.all(topk_indices >= 0)
    assert torch.all(topk_indices < n_routing_experts)
    for row in topk_indices:
        assert row.unique().numel() == row.numel()


def test_backward_router_grad() -> None:
    router = _make_router(input_dim=64, n_routing_experts=32, topk=4)
    type_emb_per_edge = torch.randn(
        100,
        64,
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )  # (E, input_dim)

    topk_weights, topk_indices = router(type_emb_per_edge)  # (E, topk), (E, topk)
    weight_scale = torch.arange(
        1,
        router.topk + 1,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )  # (topk,)
    # A plain topk_weights.sum() is constant because each softmax row sums to 1.
    loss = (topk_weights * weight_scale).sum() + 0.0 * topk_indices.float().sum()
    loss.backward()

    assert router.gate.matrix.grad is not None
    assert router.gate.matrix.grad.abs().sum().item() > 0.0
