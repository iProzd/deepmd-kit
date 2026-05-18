# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZM MoE SO(2) expert collection."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.moe.experts import (
    MoESO2ExpertCollection,
)


def _device_of(collection: MoESO2ExpertCollection) -> torch.device:
    return collection.routing_stack.layers[0].routing_matrix_m0.device


def _make_collection(
    *,
    lmax: int = 3,
    mmax: int = 1,
    focus_dim: int = 8,
    n_experts_per_gpu: int = 4,
    n_shared_experts: int = 2,
    so2_layers: int = 4,
    mlp_bias: bool = False,
    use_layer_scale: bool = False,
    seed: int = 20260518,
) -> MoESO2ExpertCollection:
    return MoESO2ExpertCollection(
        lmax=lmax,
        mmax=mmax,
        focus_dim=focus_dim,
        n_experts_per_gpu=n_experts_per_gpu,
        n_shared_experts=n_shared_experts,
        so2_layers=so2_layers,
        activation_function="silu",
        mlp_bias=mlp_bias,
        use_layer_scale=use_layer_scale,
        precision="float64",
        seed=seed,
    )


def _routing_inputs(
    collection: MoESO2ExpertCollection,
    split_sizes: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if split_sizes is None:
        split_sizes = [5, 5, 5, 5]
    device = _device_of(collection)
    n_tokens = sum(split_sizes)
    tokens = torch.randn(
        n_tokens,
        collection.reduced_dim,
        collection.focus_dim,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    eids = torch.cat(
        [
            torch.full((count,), eid, dtype=torch.long, device=device)
            for eid, count in enumerate(split_sizes)
            if count > 0
        ]
    )
    return tokens, eids, split_sizes


def _shared_inputs(
    collection: MoESO2ExpertCollection,
    n_edge: int = 20,
    *,
    same_slots: bool = False,
) -> torch.Tensor:
    device = _device_of(collection)
    if same_slots:
        base = torch.randn(
            n_edge,
            1,
            collection.reduced_dim,
            collection.focus_dim,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        return base.repeat(1, collection.n_shared_experts, 1, 1)
    return torch.randn(
        n_edge,
        collection.n_shared_experts,
        collection.reduced_dim,
        collection.focus_dim,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )


def test_invalid_config_raises() -> None:
    cases = [
        ({"lmax": -1}, "`lmax` must be non-negative"),
        ({"mmax": -1}, "`mmax` must be non-negative"),
        ({"lmax": 1, "mmax": 2}, "`mmax` must be <= `lmax`"),
        ({"focus_dim": 0}, "`focus_dim` must be positive"),
        ({"n_experts_per_gpu": 0}, "`n_experts_per_gpu` must be >= 1"),
        ({"so2_layers": 0}, "`so2_layers` must be >= 1"),
        ({"n_shared_experts": -1}, "`n_shared_experts` must be >= 0"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            _make_collection(**kwargs)


def test_param_naming_routing() -> None:
    collection = _make_collection(mlp_bias=True)
    names = dict(collection.named_parameters())

    routing_matrix_count = sum(1 for name in names if ".routing_matrix" in name)
    routing_bias_count = sum(1 for name in names if ".routing_bias" in name)

    assert routing_matrix_count >= collection.so2_layers
    assert routing_bias_count >= 1


def test_param_naming_shared() -> None:
    collection = _make_collection(mlp_bias=True)
    names = dict(collection.named_parameters())
    shared_names = [name for name in names if "shared_stack" in name]

    assert any(".shared_matrix" in name for name in shared_names)
    for name in shared_names:
        assert ".routing_matrix" not in name
        assert ".routing_bias" not in name


def test_routing_forward_shape() -> None:
    collection = _make_collection(n_shared_experts=0)
    tokens, eids, split_sizes = _routing_inputs(collection)

    out = collection.forward_routing(tokens, eids, split_sizes)

    assert out.shape == (20, collection.reduced_dim, collection.focus_dim)
    assert out.dtype == tokens.dtype
    assert torch.isfinite(out).all()


def test_routing_empty_expert_skip() -> None:
    collection = _make_collection(n_shared_experts=0)
    tokens, eids, split_sizes = _routing_inputs(collection, [3, 0, 5, 2])

    out = collection.forward_routing(tokens, eids, split_sizes)

    assert out.shape == (10, collection.reduced_dim, collection.focus_dim)
    assert torch.isfinite(out).all()


def test_shared_forward_shape() -> None:
    collection = _make_collection(n_shared_experts=2)
    x_shared = _shared_inputs(collection)

    out = collection.forward_shared(x_shared)

    assert out.shape == (20, 2, collection.reduced_dim, collection.focus_dim)
    assert torch.isfinite(out).all()

    no_shared = _make_collection(n_shared_experts=0)
    device = _device_of(no_shared)
    x_empty = torch.randn(
        20,
        0,
        no_shared.reduced_dim,
        no_shared.focus_dim,
        dtype=torch.float64,
        device=device,
    )
    out_empty = no_shared.forward_shared(x_empty)
    assert out_empty.shape == (20, 0, no_shared.reduced_dim, no_shared.focus_dim)


def test_routing_independence() -> None:
    collection = _make_collection(n_shared_experts=0, seed=1234)
    device = _device_of(collection)
    token = torch.randn(
        1,
        collection.reduced_dim,
        collection.focus_dim,
        dtype=torch.float64,
        device=device,
    )

    out0 = collection.forward_routing(
        token,
        torch.tensor([0], dtype=torch.long, device=device),
        [1, 0, 0, 0],
    )
    out1 = collection.forward_routing(
        token,
        torch.tensor([1], dtype=torch.long, device=device),
        [0, 1, 0, 0],
    )

    assert not torch.allclose(out0, out1)


def test_shared_independence() -> None:
    collection = _make_collection(
        n_experts_per_gpu=2,
        n_shared_experts=3,
        seed=5678,
    )
    x_shared = _shared_inputs(collection, n_edge=1, same_slots=True)

    out = collection.forward_shared(x_shared)

    assert not torch.allclose(out[:, 0], out[:, 1])
    assert not torch.allclose(out[:, 0], out[:, 2])
    assert not torch.allclose(out[:, 1], out[:, 2])


def test_backward_grads_present() -> None:
    collection = _make_collection(n_shared_experts=2)
    tokens, eids, split_sizes = _routing_inputs(collection)
    x_shared = _shared_inputs(collection)

    loss = collection.forward_routing(tokens, eids, split_sizes).sum()
    loss = loss + collection.forward_shared(x_shared).sum()
    loss.backward()

    for name, param in collection.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name
            assert param.grad.abs().sum().item() > 0.0, name


def test_create_graph_second_backward() -> None:
    collection = _make_collection(n_shared_experts=2)
    tokens, eids, split_sizes = _routing_inputs(collection)
    x_shared = _shared_inputs(collection)

    loss = collection.forward_routing(tokens, eids, split_sizes).sum()
    loss = loss + collection.forward_shared(x_shared).sum()
    grad_tokens, grad_shared = torch.autograd.grad(
        loss,
        (tokens, x_shared),
        create_graph=True,
    )
    (grad_tokens.sum() + grad_shared.sum()).backward()

    for name, param in collection.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name
            assert param.grad.abs().sum().item() > 0.0, name


def test_shared_parallel_correctness() -> None:
    collection = _make_collection(
        n_experts_per_gpu=2,
        n_shared_experts=3,
        seed=91011,
    )
    x_shared = _shared_inputs(collection, n_edge=5)

    y_par = collection.forward_shared(x_shared)
    y_naive_list = []
    device = x_shared.device
    for shared_idx in range(3):
        y_s = collection.shared_stack.forward_routing(
            sorted_tokens=x_shared[:, shared_idx, :, :],
            local_eids_sorted=torch.full(
                (5,), shared_idx, dtype=torch.long, device=device
            ),
            split_sizes=[0] * shared_idx + [5] + [0] * (3 - shared_idx - 1),
        )
        y_naive_list.append(y_s)
    y_naive = torch.stack(y_naive_list, dim=1)

    torch.testing.assert_close(y_par, y_naive, atol=1e-12, rtol=1e-12)


def test_mlp_bias_true_smoke() -> None:
    collection = _make_collection(n_shared_experts=2, mlp_bias=True)
    tokens, eids, split_sizes = _routing_inputs(collection, [3, 0, 5, 2])
    x_shared = _shared_inputs(collection)
    device = _device_of(collection)
    radial_factor = torch.randn(
        tokens.shape[0],
        collection.focus_dim,
        dtype=torch.float64,
        device=device,
    )
    radial_shared = torch.randn(
        x_shared.shape[0],
        collection.n_shared_experts,
        collection.focus_dim,
        dtype=torch.float64,
        device=device,
    )

    out_routing = collection.forward_routing(
        tokens,
        eids,
        split_sizes,
        radial_factor,
    )
    out_shared = collection.forward_shared(x_shared, radial_shared)
    (out_routing.sum() + out_shared.sum()).backward()

    assert out_routing.shape == (10, collection.reduced_dim, collection.focus_dim)
    assert out_shared.shape == (20, 2, collection.reduced_dim, collection.focus_dim)
    assert torch.isfinite(out_routing).all()
    assert torch.isfinite(out_shared).all()
