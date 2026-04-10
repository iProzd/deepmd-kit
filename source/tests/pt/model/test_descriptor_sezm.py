# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

# NOTE: avoid torch thread reconfiguration errors during import.
import torch

torch_set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
torch_set_num_threads = getattr(torch, "set_num_threads", None)
if torch_set_num_interop_threads is not None:
    torch.set_num_interop_threads = lambda *args, **kwargs: None  # type: ignore[assignment]
if torch_set_num_threads is not None:
    torch.set_num_threads = lambda *args, **kwargs: None  # type: ignore[assignment]

from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    InnerClamp,
    SO2Linear,
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)
from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)


def _random_quaternion(
    n_batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample normalized quaternions in ``(w, x, y, z)`` order."""
    sample_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    q = torch.randn(n_batch, 4, device=device, dtype=sample_dtype)
    q = q / torch.sqrt(
        torch.sum(q * q, dim=-1, keepdim=True).clamp_min(torch.finfo(sample_dtype).eps)
    )
    return q.to(dtype=dtype)


def _tiny_two_atom_system(
    device: torch.device,
    dtype: torch.dtype,
    *,
    padded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a minimal two-atom system for descriptor tests."""
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    ).view(1, -1, 3)
    atype = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    nlist_values = [[[1, -1], [0, -1]]] if padded else [[[1, 1], [0, 0]]]
    nlist = torch.tensor(nlist_values, dtype=torch.int64, device=device)
    return coord, atype, nlist


def _descriptor_kwargs(**overrides) -> dict:
    """Build a compact SeZM descriptor config for tests."""
    kwargs = {
        "rcut": 3.0,
        "sel": [1, 1],
        "ntypes": 2,
        "l_schedule": [1, 0],
        "channels": 4,
        "n_radial": 3,
        "radial_mlp": [6],
        "ffn_neurons": 8,
        "ffn_blocks": 1,
        "random_gamma": False,
        "n_atten_head": 0,
        "mlp_bias": True,
        "precision": "float32",
        "trainable": True,
    }
    kwargs.update(overrides)
    return kwargs


def _attention_descriptor_kwargs(
    *,
    precision: str = "float32",
    seed: int | None = None,
    **overrides,
) -> dict:
    """Build a richer attention-enabled SeZM descriptor config for tests."""
    kwargs = _descriptor_kwargs(
        l_schedule=[1, 1, 0],
        channels=8,
        n_focus=2,
        focus_dim=0,
        n_radial=4,
        radial_mlp=[8],
        so2_layers=2,
        full_attn_res="dependent",
        so2_attn_res="dependent",
        ffn_neurons=16,
        ffn_blocks=2,
        layer_scale=False,
        precision=precision,
        seed=seed,
    )
    kwargs.update(overrides)
    return kwargs


def _forward_tols(dtype: torch.dtype) -> tuple[float, float]:
    """Return output comparison tolerances for one dtype."""
    if dtype == torch.float64:
        return 1e-10, 1e-10
    if dtype == torch.float32:
        return 5e-5, 5e-5
    return 5e-3, 5e-3


def _parameter_tols(dtype: torch.dtype) -> tuple[float, float]:
    """Return parameter comparison tolerances for one dtype."""
    if dtype == torch.float64:
        return 1e-10, 1e-10
    if dtype == torch.float32:
        return 1e-6, 1e-6
    return 1e-3, 1e-3


class _SeZMTestCase(unittest.TestCase):
    """Base test case with the shared device setup."""

    def setUp(self) -> None:
        self.device = env.DEVICE


class TestDescrptSeZM(_SeZMTestCase):
    """Test the SeZM descriptor."""

    def _assert_forward_backward_smoke(self, **model_kwargs) -> DescrptSeZM:
        """Run a compact forward/backward smoke test and return the model."""
        coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=torch.float32)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
        model = DescrptSeZM(**model_kwargs)
        desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
        self.assertEqual(desc.shape, (1, 2, model_kwargs["channels"]))
        self.assertEqual(desc.dtype, env.GLOBAL_PT_FLOAT_PRECISION)
        desc.sum().backward()
        self.assertIsNotNone(extended_coord.grad)
        self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))
        return model

    def test_forward_with_focus_dim_variants(self) -> None:
        """Test forward/backward smoke paths for focus_dim=0 and explicit hidden width."""
        cases = {
            "focus_dim_zero": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
            ),
            "focus_dim_explicit": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=3,
                so2_layers=2,
            ),
            "focus_dim_zero_s2": _descriptor_kwargs(
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                s2_activation=True,
                s2_grid_resolution=[8, 12],
            ),
        }
        for name, model_kwargs in cases.items():
            with self.subTest(mode=name):
                self._assert_forward_backward_smoke(**model_kwargs)

    def test_forward_with_attention_variants(self) -> None:
        """Test forward/backward smoke paths for attention-based variants."""
        cases = {
            "full_attention": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
            ),
            "block_attention": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
                full_attn_res="none",
                block_attn_res="dependent",
            ),
            "full_attention_s2": _attention_descriptor_kwargs(
                precision="float32",
                seed=123,
                s2_activation=True,
                s2_grid_resolution=[8, 12],
            ),
        }
        for name, model_kwargs in cases.items():
            with self.subTest(mode=name):
                self._assert_forward_backward_smoke(**model_kwargs)

    def test_forward_backward_second_order_fixed_edges(self) -> None:
        """Test fixed-shape edge path matches nlist for fwd/bwd/2nd order."""
        dtype = torch.float32
        coord = torch.tensor(
            [[0.1, 0.2, 0.3], [1.1, 0.7, 0.2]],
            dtype=dtype,
            device=self.device,
        ).view(1, -1, 3)
        atype = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        nlist = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.int64, device=self.device)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)

        model = DescrptSeZM(
            **_attention_descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=1,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
            )
        )

        desc_nlist, _, _, _, sw_nlist = model(
            extended_coord, atype, nlist, mapping=None, comm_dict=None
        )

        # Fixed-shape edge list for n_node=2, nsel=2
        edge_index = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 1]],
            dtype=torch.long,
            device=self.device,
        )
        coord_view = extended_coord.view(1, 2, 3)
        valid_nlist = nlist >= 0
        gather_index = torch.where(valid_nlist, nlist, torch.zeros_like(nlist))
        index = gather_index.view(1, 4, 1).expand(-1, -1, 3)
        nei_pos = torch.gather(coord_view, 1, index).view(1, 2, 2, 3)
        atom_pos = coord_view[:, :2].unsqueeze(2)
        diff = nei_pos - atom_pos
        edge_vec = diff.reshape(4, 3)
        edge_mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool, device=self.device)

        desc_edge, _, _, _, sw_edge = model(
            extended_coord,
            atype,
            nlist,
            mapping=None,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            comm_dict=None,
        )

        torch.testing.assert_close(desc_nlist, desc_edge, atol=1e-6, rtol=1e-6)

        loss_nlist = desc_nlist.sum()
        loss_edge = desc_edge.sum()

        (grad_nlist,) = torch.autograd.grad(
            loss_nlist, extended_coord, create_graph=True
        )
        (grad_edge,) = torch.autograd.grad(loss_edge, extended_coord, create_graph=True)
        torch.testing.assert_close(grad_nlist, grad_edge, atol=1e-6, rtol=1e-6)

        (grad2_nlist,) = torch.autograd.grad(
            grad_nlist.sum(), extended_coord, create_graph=False
        )
        (grad2_edge,) = torch.autograd.grad(
            grad_edge.sum(), extended_coord, create_graph=False
        )
        torch.testing.assert_close(grad2_nlist, grad2_edge, atol=1e-6, rtol=1e-6)

    def test_backward_gradient(self) -> None:
        """Test backward gradient through coordinates."""
        for prec in ["float64", "float32", "bfloat16"]:
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
            extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
            model = DescrptSeZM(
                **_descriptor_kwargs(
                    ffn_blocks=2,
                    layer_scale=False,
                    precision=prec,
                )
            )
            desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
            loss = desc.sum()
            loss.backward()
            self.assertIsNotNone(extended_coord.grad)
            self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))

    def test_serialization_deserialization(self) -> None:
        """Test serialization and deserialization preserves model state."""
        cases = {
            "focus_dim_zero": _attention_descriptor_kwargs(
                precision="float32",
                focus_dim=0,
            ),
            "focus_dim_explicit": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=3,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
            ),
            "focus_dim_zero_s2": _descriptor_kwargs(
                precision="float32",
                channels=4,
                n_focus=2,
                focus_dim=0,
                so2_layers=2,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                s2_activation=True,
                s2_grid_resolution=[8, 12],
            ),
        }
        dtype = PRECISION_DICT["float32"]
        for case_name, model_kwargs in cases.items():
            coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
            extended_coord = coord.reshape(1, -1)
            with self.subTest(mode=case_name):
                model = DescrptSeZM(**model_kwargs)

                desc1, _, _, _, sw1 = model(extended_coord, atype, nlist)
                data = model.serialize()
                model_restored = DescrptSeZM.deserialize(data)
                desc2, _, _, _, sw2 = model_restored(extended_coord, atype, nlist)
                atol, rtol = _forward_tols(dtype)

                torch.testing.assert_close(
                    desc1,
                    desc2,
                    atol=atol,
                    rtol=rtol,
                    msg="Descriptor mismatch after deserialization",
                )
                torch.testing.assert_close(
                    sw1,
                    sw2,
                    atol=atol,
                    rtol=rtol,
                    msg="Smooth weight mismatch after deserialization",
                )

    def test_seed_reproducibility(self) -> None:
        """Test that fixed seed produces identical model initialization."""
        for prec in ["float64", "float32", "bfloat16"]:
            dtype = PRECISION_DICT[prec]
            seed = 12345

            model_kwargs = _attention_descriptor_kwargs(precision=prec, seed=seed)
            model1 = DescrptSeZM(**model_kwargs)
            model2 = DescrptSeZM(**model_kwargs)
            param_atol, param_rtol = _parameter_tols(dtype)

            for (n1, p1), (n2, p2) in zip(
                model1.named_parameters(), model2.named_parameters(), strict=False
            ):
                self.assertEqual(n1, n2, msg="Parameter name mismatch")
                torch.testing.assert_close(
                    p1,
                    p2,
                    atol=param_atol,
                    rtol=param_rtol,
                    msg=f"Parameter {n1} differs between models with same seed",
                )

            coord, atype, nlist = _tiny_two_atom_system(self.device, dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            desc1, _, _, _, sw1 = model1(extended_coord, atype, nlist)
            desc2, _, _, _, sw2 = model2(extended_coord, atype, nlist)
            forward_atol, forward_rtol = _forward_tols(dtype)

            torch.testing.assert_close(
                desc1,
                desc2,
                atol=forward_atol,
                rtol=forward_rtol,
                msg="Forward output differs for models with same seed",
            )
            torch.testing.assert_close(
                sw1,
                sw2,
                atol=forward_atol,
                rtol=forward_rtol,
                msg="Smooth weight differs for models with same seed",
            )


class TestBuildEdgeQuaternion(_SeZMTestCase):
    """Test the stable edge-quaternion chart used by SeZM."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 1e-4, 1e-4
        return 5e-3, 5e-3

    def _safe_norm(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True).clamp(min=eps))

    def _assert_quaternion_invariants(
        self, edge_quat: torch.Tensor, edge_vec: torch.Tensor
    ) -> None:
        atol, rtol = self._get_tols(edge_vec.dtype)
        rot_mat = quaternion_to_rotation_matrix(edge_quat)
        n_edge = int(edge_vec.shape[0])
        eye = torch.eye(3, device=self.device, dtype=edge_vec.dtype).expand(
            n_edge, 3, 3
        )
        torch.testing.assert_close(
            rot_mat @ rot_mat.transpose(-1, -2),
            eye,
            atol=atol,
            rtol=rtol,
        )

        edge_unit = edge_vec / self._safe_norm(edge_vec)
        ez = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=edge_vec.dtype
        ).expand(n_edge, 3)
        rotated = (rot_mat @ edge_unit.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(rotated, ez, atol=atol, rtol=rtol)

        det_mat = rot_mat.float() if rot_mat.dtype == torch.bfloat16 else rot_mat
        det = torch.linalg.det(det_mat)
        torch.testing.assert_close(
            det,
            torch.ones_like(det),
            atol=atol,
            rtol=rtol,
        )

    def test_invariants_random_edges(self) -> None:
        for dtype in [torch.float64, torch.float32]:
            edge_vec = torch.randn(512, 3, device=self.device, dtype=dtype)
            edge_quat = build_edge_quaternion(edge_vec)
            self._assert_quaternion_invariants(edge_quat, edge_vec)

    def test_invariants_near_poles(self) -> None:
        for dtype in [torch.float64, torch.float32]:
            delta = torch.tensor(
                [-1.0e-3, -1.0e-4, 0.0, 1.0e-4, 1.0e-3],
                device=self.device,
                dtype=dtype,
            )
            for sign in [1.0, -1.0]:
                edge_vec = torch.stack(
                    [delta, torch.zeros_like(delta), torch.full_like(delta, sign)],
                    dim=-1,
                )
                edge_quat = build_edge_quaternion(edge_vec)
                self._assert_quaternion_invariants(edge_quat, edge_vec)


class TestWignerDCalculator(_SeZMTestCase):
    """Test the quaternion-driven Wigner-D calculator."""

    def setUp(self) -> None:
        super().setUp()
        self.batch = 8
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 5e-5, 5e-5
        return 5e-3, 5e-3

    def _extract_l_block(
        self,
        D_full: torch.Tensor,
        l: int,
    ) -> torch.Tensor:
        """Extract the l-block from D_full."""
        s, e = l * l, (l + 1) * (l + 1)
        return D_full[:, s:e, s:e]

    def test_orthogonality(self) -> None:
        """Test D @ D^T = I for random quaternions."""
        for dtype, lmax in itertools.product([torch.float64, torch.float32], [1, 3, 6]):
            atol, rtol = self._get_tols(dtype)
            wigner = WignerDCalculator(lmax=lmax, dtype=dtype)
            edge_quat = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            D_full, Dt_full = wigner(edge_quat)

            for l in range(lmax + 1):
                dim = 2 * l + 1
                eye = torch.eye(dim, device=self.device, dtype=dtype).expand(
                    self.batch, dim, dim
                )
                D_l = self._extract_l_block(D_full, l)
                Dt_l = self._extract_l_block(Dt_full, l)
                torch.testing.assert_close(
                    D_l @ Dt_l,
                    eye,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Orthogonality failed for WignerDCalculator, dtype={dtype}, lmax={lmax}, l={l}"
                    ),
                )

    def test_group_property(self) -> None:
        """Test group property in quaternion composition order."""
        for dtype, lmax in itertools.product([torch.float64, torch.float32], [1, 3, 6]):
            atol = 1e-10 if dtype == torch.float64 else 5e-4
            rtol = 1e-10 if dtype == torch.float64 else 5e-4
            wigner = WignerDCalculator(lmax=lmax, dtype=dtype)

            q1 = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            q2 = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            q12 = quaternion_multiply(q1, q2)

            D1_full, _ = wigner(q1)
            D2_full, _ = wigner(q2)
            D12_full, _ = wigner(q12)

            for l in range(lmax + 1):
                D1_l = self._extract_l_block(D1_full, l)
                D2_l = self._extract_l_block(D2_full, l)
                D12_l = self._extract_l_block(D12_full, l)
                torch.testing.assert_close(
                    D12_l,
                    D1_l @ D2_l,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Group property failed for WignerDCalculator, dtype={dtype}, lmax={lmax}, l={l}"
                    ),
                )

    def test_l1_matches_vector_representation(self) -> None:
        """Test that the l=1 block matches the Cartesian vector representation."""
        for dtype in [torch.float64, torch.float32]:
            atol, rtol = self._get_tols(dtype)
            S = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                device=self.device,
                dtype=dtype,
            )
            S_batch = S.unsqueeze(0).expand(self.batch, 3, 3)

            edge_quat = _random_quaternion(self.batch, device=self.device, dtype=dtype)
            rot = quaternion_to_rotation_matrix(edge_quat)
            wigner = WignerDCalculator(lmax=1, dtype=dtype)
            D_full, Dt_full = wigner(edge_quat)
            D1 = self._extract_l_block(D_full, 1)
            Dt1 = self._extract_l_block(Dt_full, 1)

            expected = S_batch @ rot @ S_batch.transpose(-1, -2)
            torch.testing.assert_close(
                D1,
                expected,
                atol=atol,
                rtol=rtol,
                msg=f"l=1 block mismatch for WignerDCalculator, dtype={dtype}",
            )
            torch.testing.assert_close(
                Dt1,
                expected.transpose(-1, -2),
                atol=atol,
                rtol=rtol,
                msg=f"l=1 transpose block mismatch for WignerDCalculator, dtype={dtype}",
            )

    def test_pole_path_gradient_matches_finite_difference(self) -> None:
        """Check one pole-crossing Wigner probe against finite differences."""
        for dtype in [torch.float64, torch.float32]:
            wigner = WignerDCalculator(lmax=6, dtype=dtype)
            atol = 5.0e-8 if dtype == torch.float64 else 2.0e-6
            rtol = 1.0e-6 if dtype == torch.float64 else 2.0e-4
            for sign in [1.0, -1.0]:
                delta = torch.linspace(
                    -0.02,
                    0.02,
                    257,
                    device=self.device,
                    dtype=dtype,
                    requires_grad=True,
                )
                edge_vec = torch.stack(
                    [delta, torch.zeros_like(delta), torch.full_like(delta, sign)],
                    dim=-1,
                )
                edge_quat = build_edge_quaternion(edge_vec)
                D_full, _ = wigner(edge_quat)
                probe = D_full[:, 5, 7] + D_full[:, 17, 19]
                grad = torch.autograd.grad(probe.sum(), delta)[0]
                delta_detached = delta.detach()
                probe_detached = probe.detach()
                numerical_grad = (probe_detached[2:] - probe_detached[:-2]) / (
                    2.0 * (delta_detached[1] - delta_detached[0])
                )
                torch.testing.assert_close(
                    grad[1:-1].detach(),
                    numerical_grad,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Pole-path Wigner gradient mismatch for dtype={dtype}, sign={sign}"
                    ),
                )

    def test_y_crossing_overlap_has_no_large_wigner_jump(self) -> None:
        """Check chart-overlap continuity for a path that crosses y=0."""
        for dtype in [torch.float64, torch.float32]:
            wigner = WignerDCalculator(lmax=4, dtype=dtype)
            max_allowed = 1.0e-2 if dtype == torch.float64 else 1.5e-2
            y_vals = torch.tensor(
                [-1.0e-3, -5.0e-4, -1.0e-4, 0.0, 1.0e-4, 5.0e-4, 1.0e-3],
                device=self.device,
                dtype=dtype,
            )
            edge_vec = torch.stack(
                [
                    torch.full_like(y_vals, 0.35),
                    y_vals,
                    torch.full_like(y_vals, 0.25),
                ],
                dim=-1,
            )
            edge_quat = build_edge_quaternion(edge_vec)
            D_full, _ = wigner(edge_quat)
            step = (D_full[1:] - D_full[:-1]).abs().amax(dim=(1, 2))
            self.assertLess(
                step.max().item(),
                max_allowed,
                msg=f"Large Wigner jump across y=0 for dtype={dtype}",
            )


class TestSO2LinearEquivariance(_SeZMTestCase):
    """Test SO2Linear z-rotation equivariance: SO2Linear(Z @ x) = Z @ SO2Linear(x)."""

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 1e-5, 1e-5
        # bf16 has only 7-bit mantissa; use looser tolerance for equivariance tests.
        return 2e-2, 2e-2

    def _build_m_major_z_rotation(
        self, angles: torch.Tensor, lmax: int, mmax: int
    ) -> torch.Tensor:
        """
        Build block z-rotation matrix for the m-major truncated layout.

        Parameters
        ----------
        angles
            Rotation angles with shape (batch,).
        lmax
            Maximum degree.
        mmax
            Maximum order (|m|). Must satisfy mmax <= lmax.

        Returns
        -------
        torch.Tensor
            Z matrix with shape (batch, dim_red, dim_red).
        """
        batch = angles.shape[0]
        m0_size = lmax + 1
        dim_red = m0_size
        for m in range(1, mmax + 1):
            num_l = lmax - m + 1
            dim_red += 2 * num_l

        Z = angles.new_zeros(batch, dim_red, dim_red)
        eye0 = torch.eye(m0_size, device=self.device, dtype=angles.dtype).expand(
            batch, m0_size, m0_size
        )
        Z[:, :m0_size, :m0_size] = eye0

        offset = m0_size
        for m in range(1, mmax + 1):
            num_l = lmax - m + 1
            eye = torch.eye(num_l, device=self.device, dtype=angles.dtype).expand(
                batch, num_l, num_l
            )
            cos_m = torch.cos(m * angles).view(batch, 1, 1)
            sin_m = torch.sin(m * angles).view(batch, 1, 1)

            # In m-major layout, each m group is stored as [neg(l), pos(l)] with two halves.
            # Rotation is [[cos I, -sin I], [sin I, cos I]] for the (neg, pos) pair.
            Z[:, offset : offset + num_l, offset : offset + num_l] = cos_m * eye
            Z[
                :,
                offset : offset + num_l,
                offset + num_l : offset + 2 * num_l,
            ] = -sin_m * eye
            Z[
                :,
                offset + num_l : offset + 2 * num_l,
                offset : offset + num_l,
            ] = sin_m * eye
            Z[
                :,
                offset + num_l : offset + 2 * num_l,
                offset + num_l : offset + 2 * num_l,
            ] = cos_m * eye
            offset += 2 * num_l

        return Z

    def test_equivariance_random_angles(self) -> None:
        """Test SO2Linear(Z @ x) = Z @ SO2Linear(x) for random z-rotations."""
        for dtype, lmax, mmax in itertools.product(
            [torch.float64, torch.float32, torch.bfloat16],
            [1, 2, 3],
            [1, 2, 3],
        ):
            if mmax > lmax:
                continue
            atol, rtol = self._get_tols(dtype)
            batch = 16
            channels_in = 8
            channels_out = 12

            so2_linear = SO2Linear(
                lmax=lmax,
                mmax=mmax,
                in_channels=channels_in,
                out_channels=channels_out,
                dtype=dtype,
                seed=123,
                trainable=True,
            )

            dim_red = so2_linear.reduced_dim
            x = torch.randn(
                batch, 1, dim_red, channels_in, device=self.device, dtype=dtype
            )

            angles = torch.rand(batch, device=self.device, dtype=dtype) * 2 * 3.14159
            Z = self._build_m_major_z_rotation(angles, lmax, mmax)

            x_rotated = torch.einsum("bij,bfjc->bfic", Z, x)
            lhs = so2_linear(x_rotated)
            rhs = torch.einsum("bij,bfjc->bfic", Z, so2_linear(x))

            torch.testing.assert_close(
                lhs,
                rhs,
                atol=atol,
                rtol=rtol,
                msg=f"SO2Linear equivariance failed for dtype={dtype}, lmax={lmax}, mmax={mmax}",
            )


class TestInnerClamp(_SeZMTestCase):
    """Test InnerClamp C3-continuous septic Hermite clamping."""

    def setUp(self) -> None:
        super().setUp()
        self.r_inner = 1.0
        self.r_outer = 1.5
        self.clamp = InnerClamp(self.r_inner, self.r_outer)

    def test_monotonicity(self) -> None:
        """Test that r̃ is monotonically non-decreasing."""
        r = torch.linspace(0.0, 3.0, 1000, dtype=torch.float64, device=self.device)
        out = self.clamp(r)
        diff = out[1:] - out[:-1]
        self.assertTrue((diff >= -1e-14).all(), "InnerClamp is not monotonic")

    def test_frozen_zone_zero_gradient(self) -> None:
        """Test that dr̃/dr = 0 for r < r_inner (frozen zone)."""
        r = torch.tensor(
            [0.3, 0.5, 0.8, 0.99],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        out = self.clamp(r)
        grads = torch.autograd.grad(out.sum(), r)[0]
        torch.testing.assert_close(
            grads,
            torch.zeros_like(grads),
            atol=1e-12,
            rtol=0,
            msg="Gradient should be zero in the frozen zone",
        )

    def test_identity_zone_unit_gradient(self) -> None:
        """Test that dr̃/dr = 1 for r > r_outer (identity zone)."""
        r = torch.tensor(
            [1.6, 2.0, 3.0, 5.0],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        out = self.clamp(r)
        grads = torch.autograd.grad(out.sum(), r)[0]
        torch.testing.assert_close(
            grads,
            torch.ones_like(grads),
            atol=1e-12,
            rtol=0,
            msg="Gradient should be 1 in the identity zone",
        )

    def test_c3_continuity_at_boundaries(self) -> None:
        """Test C3 continuity at r_inner and r_outer via autograd derivatives."""
        eps = 1e-6
        for boundary in [self.r_inner, self.r_outer]:
            r = torch.tensor(
                [boundary - eps, boundary, boundary + eps],
                dtype=torch.float64,
                device=self.device,
                requires_grad=True,
            )
            out = self.clamp(r)

            # First derivative via autograd
            grads = torch.autograd.grad(out.sum(), r, create_graph=True)[0]
            # dr̃/dr should be continuous (left ≈ center ≈ right)
            self.assertAlmostEqual(
                grads[0].item(),
                grads[1].item(),
                places=4,
                msg=f"First derivative discontinuous at {boundary}",
            )
            self.assertAlmostEqual(
                grads[1].item(),
                grads[2].item(),
                places=4,
                msg=f"First derivative discontinuous at {boundary}",
            )

            # Second derivative via autograd
            grads2 = torch.autograd.grad(grads.sum(), r, create_graph=True)[0]
            self.assertAlmostEqual(
                grads2[0].item(),
                grads2[1].item(),
                places=3,
                msg=f"Second derivative discontinuous at {boundary}",
            )
            self.assertAlmostEqual(
                grads2[1].item(),
                grads2[2].item(),
                places=3,
                msg=f"Second derivative discontinuous at {boundary}",
            )

            # Third derivative via autograd
            grads3 = torch.autograd.grad(grads2.sum(), r)[0]
            self.assertAlmostEqual(
                grads3[0].item(),
                grads3[1].item(),
                places=2,
                msg=f"Third derivative discontinuous at {boundary}",
            )
            self.assertAlmostEqual(
                grads3[1].item(),
                grads3[2].item(),
                places=2,
                msg=f"Third derivative discontinuous at {boundary}",
            )

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            InnerClamp(1.5, 1.0)
        with self.assertRaises(ValueError):
            InnerClamp(-1.0, 1.0)
        with self.assertRaises(ValueError):
            InnerClamp(1.0, 1.0)


class TestDescriptorInnerClamp(_SeZMTestCase):
    """Test DescrptSeZM with inner clamping enabled vs disabled."""

    def test_clamp_no_effect_beyond_r_outer(self) -> None:
        """Test that inner clamp has no effect when all distances > r_outer."""
        dtype = torch.float32
        coord = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=dtype,
            device=self.device,
        ).view(1, -1, 3)
        atype = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        nlist = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.int64, device=self.device)
        extended_coord = coord.reshape(1, -1)

        base_kwargs = _descriptor_kwargs(
            precision="float32",
            seed=42,
        )

        model_no_clamp = DescrptSeZM(**base_kwargs)
        model_clamp = DescrptSeZM(
            inner_clamp_r_inner=1.0,
            inner_clamp_r_outer=1.5,
            **base_kwargs,
        )
        model_clamp.load_state_dict(model_no_clamp.state_dict())

        desc_no, *_ = model_no_clamp(extended_coord, atype, nlist)
        desc_yes, *_ = model_clamp(extended_coord, atype, nlist)

        # At r=2.0 > r_outer=1.5, outputs should be identical
        torch.testing.assert_close(
            desc_no,
            desc_yes,
            atol=1e-6,
            rtol=1e-6,
            msg="Descriptor should be identical when all distances > r_outer",
        )


class TestDescriptorEnergyCurveSmoothness(_SeZMTestCase):
    """Test PES smoothness from scaled symmetric eight-atom probes."""

    RANDOM_WEIGHT_BASE_SEED = 184
    RANDOM_WEIGHT_STD = 0.1
    N_DISPLACEMENT_POINTS = 201
    MAX_DISPLACEMENT = 0.1
    RCUT_NEAR_DISTANCE = 4.95
    BRIDGING_R_INNER = 0.9
    BRIDGING_R_OUTER = 1.3
    ENERGY_SPAN_MARGIN = 1.0e-7
    FIRST_DERIVATIVE_MARGIN = 5.0e-7
    SECOND_DERIVATIVE_MARGIN = 5.0e-4
    EXTREMUM_MARGIN = 1.0e-9

    def setUp(self) -> None:
        super().setUp()
        self.dtype = torch.float64
        self.symmetry_frac_coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=self.dtype,
            device=self.device,
        ).view(1, 8, 3)
        self.symmetry_atype = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1]],
            dtype=torch.int32,
            device=self.device,
        )

    def _build_model_params(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int = 1,
        bridging_method: str = "none",
        bridging_r_inner: float = 0.9,
        bridging_r_outer: float = 1.3,
    ) -> dict:
        """Build the SeZM probe model used for one PES scan."""
        params = {
            "type": "SeZM",
            "type_map": ["Na", "Cl"],
            "descriptor": {
                "type": "SeZM",
                "sel": [16, 16],
                "rcut": 5.0,
                "channels": 16,
                "n_focus": n_focus,
                "focus_dim": 0,
                "focus_compete": True,
                "n_radial": 6,
                "radial_mlp": [16],
                "use_env_seed": True,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": n_atten_head,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 16,
                "ffn_blocks": 1,
                "mlp_bias": True,
                "layer_scale": False,
                "use_amp": use_amp,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [],
                "activation_function": "silu",
                "precision": "float64",
                "seed": 7,
            },
        }
        if bridging_method.lower() != "none":
            params["bridging_method"] = bridging_method
            params["bridging_r_inner"] = bridging_r_inner
            params["bridging_r_outer"] = bridging_r_outer
        return params

    def _build_scaled_symmetric_structure(
        self,
        nearest_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale one symmetric eight-atom template to a target nearest-neighbor distance."""
        lattice = 2.0 * nearest_distance
        coord = self.symmetry_frac_coord * lattice
        box = torch.tensor(
            [[lattice, 0.0, 0.0, 0.0, lattice, 0.0, 0.0, 0.0, lattice]],
            dtype=self.dtype,
            device=self.device,
        )
        return coord, box

    def _stable_parameter_seed(self, name: str) -> int:
        """Return an order-independent seed derived from one parameter name."""
        seed_value = self.RANDOM_WEIGHT_BASE_SEED
        for idx, char in enumerate(name):
            seed_value += (idx + 1) * ord(char)
        return seed_value

    def _build_random_weight_model(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int = 1,
        bridging_method: str = "none",
        bridging_r_inner: float = 0.9,
        bridging_r_outer: float = 1.3,
    ) -> torch.nn.Module:
        """Build a probe model and randomize all floating-point parameters."""
        model = get_sezm_model(
            self._build_model_params(
                n_atten_head,
                use_amp=use_amp,
                n_focus=n_focus,
                bridging_method=bridging_method,
                bridging_r_inner=bridging_r_inner,
                bridging_r_outer=bridging_r_outer,
            )
        ).to(
            device=self.device,
            dtype=self.dtype,
        )
        randomized = 0

        # Assign deterministic random weights by parameter name.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.is_floating_point():
                    continue
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self._stable_parameter_seed(name))
                param.normal_(
                    mean=0.0,
                    std=self.RANDOM_WEIGHT_STD,
                    generator=generator,
                )
                randomized += 1

        self.assertGreater(
            randomized, 0, "No floating-point parameters were randomized"
        )
        model.eval()
        return model

    def _scan_total_energy_curve(
        self,
        model: torch.nn.Module,
        *,
        nearest_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan total energies while displacing atom 0 along x."""
        displacements = torch.linspace(
            -self.MAX_DISPLACEMENT,
            self.MAX_DISPLACEMENT,
            self.N_DISPLACEMENT_POINTS,
            dtype=self.dtype,
            device=self.device,
        )
        coord0, box = self._build_scaled_symmetric_structure(nearest_distance)
        coord = coord0.repeat(displacements.shape[0], 1, 1)
        coord[:, 0, 0] += displacements
        result = model(
            coord,
            self.symmetry_atype.expand(displacements.shape[0], -1),
            box=box.expand(displacements.shape[0], -1),
        )
        return displacements, result["energy"][:, 0].detach()

    def _collect_curve_statistics(
        self,
        energies: torch.Tensor,
        displacements: torch.Tensor,
    ) -> dict[str, float | str]:
        """Compute derivative and extremum statistics for one energy curve."""
        step = displacements[1] - displacements[0]
        first = (energies[2:] - energies[:-2]) / (2.0 * step)
        second = (energies[2:] - 2.0 * energies[1:-1] + energies[:-2]) / (step * step)
        center_idx = energies.shape[0] // 2
        deriv_center_idx = first.shape[0] // 2
        left_first = first[:deriv_center_idx]
        right_first = first[deriv_center_idx + 1 :]
        center_energy = energies[center_idx]
        other_energies = torch.cat((energies[:center_idx], energies[center_idx + 1 :]))

        bowl_up = bool(
            torch.all(left_first < -self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(right_first > self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(second > self.SECOND_DERIVATIVE_MARGIN)
            and torch.all(center_energy <= other_energies + self.EXTREMUM_MARGIN)
        )
        bowl_down = bool(
            torch.all(left_first > self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(right_first < -self.FIRST_DERIVATIVE_MARGIN)
            and torch.all(second < -self.SECOND_DERIVATIVE_MARGIN)
            and torch.all(center_energy >= other_energies - self.EXTREMUM_MARGIN)
        )

        if bowl_up:
            curve_kind = "minimum"
        elif bowl_down:
            curve_kind = "maximum"
        else:
            curve_kind = "invalid"

        return {
            "curve_kind": curve_kind,
            "energy_span": float((energies.max() - energies.min()).abs()),
            "left_abs_min": float(left_first.abs().min()),
            "right_abs_min": float(right_first.abs().min()),
            "curvature_abs_min": float(second.abs().min()),
        }

    def _assert_curve_has_usable_signal(
        self,
        stats: dict[str, float | str],
        *,
        label: str,
        n_atten_head: int,
    ) -> None:
        """Check that the scanned curve has enough signal above numerical noise."""
        self.assertGreater(
            stats["energy_span"],
            self.ENERGY_SPAN_MARGIN,
            f"{label} energy curve became nearly flat for n_atten_head={n_atten_head}: {stats}",
        )
        self.assertGreater(
            stats["left_abs_min"],
            self.FIRST_DERIVATIVE_MARGIN,
            f"{label} left-branch slope is too small for n_atten_head={n_atten_head}: {stats}",
        )
        self.assertGreater(
            stats["right_abs_min"],
            self.FIRST_DERIVATIVE_MARGIN,
            f"{label} right-branch slope is too small for n_atten_head={n_atten_head}: {stats}",
        )
        self.assertGreater(
            stats["curvature_abs_min"],
            self.SECOND_DERIVATIVE_MARGIN,
            f"{label} curvature is too small for n_atten_head={n_atten_head}: {stats}",
        )

    def _assert_cutoff_near_energy_curve_is_smooth(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int,
    ) -> None:
        """Check that the non-bridged near-cutoff probe keeps one smooth extremum."""
        model = self._build_random_weight_model(
            n_atten_head,
            use_amp=use_amp,
            n_focus=n_focus,
        )
        displacements, energies = self._scan_total_energy_curve(
            model,
            nearest_distance=self.RCUT_NEAR_DISTANCE,
        )
        self.assertTrue(torch.isfinite(energies).all().item())

        stats = self._collect_curve_statistics(energies, displacements)
        self._assert_curve_has_usable_signal(
            stats,
            label=f"Near-cutoff (use_amp={use_amp}, n_focus={n_focus})",
            n_atten_head=n_atten_head,
        )
        self.assertIn(
            stats["curve_kind"],
            {"minimum", "maximum"},
            (
                "Near-cutoff energy curve is not a single smooth bowl "
                f"for n_atten_head={n_atten_head}, use_amp={use_amp}, n_focus={n_focus}: {stats}"
            ),
        )

    def _assert_bridged_boundary_energy_curve_is_smooth(
        self,
        n_atten_head: int,
        *,
        use_amp: bool,
        n_focus: int,
        nearest_distance: float,
        boundary_label: str,
    ) -> None:
        """Check that one bridged boundary probe keeps one smooth minimum."""
        model = self._build_random_weight_model(
            n_atten_head,
            use_amp=use_amp,
            n_focus=n_focus,
            bridging_method="ZBL",
            bridging_r_inner=self.BRIDGING_R_INNER,
            bridging_r_outer=self.BRIDGING_R_OUTER,
        )
        displacements, energies = self._scan_total_energy_curve(
            model,
            nearest_distance=nearest_distance,
        )
        self.assertTrue(torch.isfinite(energies).all().item())

        stats = self._collect_curve_statistics(energies, displacements)
        self._assert_curve_has_usable_signal(
            stats,
            label=f"Bridged {boundary_label} (use_amp={use_amp}, n_focus={n_focus})",
            n_atten_head=n_atten_head,
        )
        self.assertEqual(
            stats["curve_kind"],
            "minimum",
            (
                f"Bridged {boundary_label} probe should form one symmetric repulsive bowl "
                f"for n_atten_head={n_atten_head}, use_amp={use_amp}, n_focus={n_focus}: {stats}"
            ),
        )

    def test_scaled_cutoff_near_energy_curve_is_smooth_across_attention_modes(
        self,
    ) -> None:
        """Check the non-bridged near-cutoff PES shape across attention and AMP modes."""
        for use_amp in (False, True):
            for n_atten_head in (0, 2):
                for n_focus in (1, 2):
                    with self.subTest(
                        n_atten_head=n_atten_head, use_amp=use_amp, n_focus=n_focus
                    ):
                        self._assert_cutoff_near_energy_curve_is_smooth(
                            n_atten_head,
                            use_amp=use_amp,
                            n_focus=n_focus,
                        )

    def test_scaled_bridging_inner_energy_curve_is_smooth_across_attention_modes(
        self,
    ) -> None:
        """Check the bridged near-r_inner PES shape across attention and AMP modes."""
        for use_amp in (False, True):
            for n_atten_head in (0, 2):
                for n_focus in (1, 2):
                    with self.subTest(
                        n_atten_head=n_atten_head, use_amp=use_amp, n_focus=n_focus
                    ):
                        self._assert_bridged_boundary_energy_curve_is_smooth(
                            n_atten_head,
                            use_amp=use_amp,
                            n_focus=n_focus,
                            nearest_distance=self.BRIDGING_R_INNER,
                            boundary_label="r_inner",
                        )

    def test_scaled_bridging_outer_energy_curve_is_smooth_across_attention_modes(
        self,
    ) -> None:
        """Check the bridged near-r_outer PES shape across attention and AMP modes."""
        for use_amp in (False, True):
            for n_atten_head in (0, 2):
                for n_focus in (1, 2):
                    with self.subTest(
                        n_atten_head=n_atten_head, use_amp=use_amp, n_focus=n_focus
                    ):
                        self._assert_bridged_boundary_energy_curve_is_smooth(
                            n_atten_head,
                            use_amp=use_amp,
                            n_focus=n_focus,
                            nearest_distance=self.BRIDGING_R_OUTER,
                            boundary_label="r_outer",
                        )


if __name__ == "__main__":
    unittest.main()
