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

from deepmd.pt.model.descriptor.se_zm_block import (
    PerDegreeLinear,
    PerDegreeLinearV2,
    SO2Linear,
    _so3_dim_of_lmax,
)
from deepmd.pt.model.descriptor.se_zm_net import (
    DescrptSeZMNet,
    init_edge_rot_mat_frisvad,
)
from deepmd.pt.model.descriptor.wigner_d import (
    WignerDCalc,
    WignerDCalcParallel,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)


def _zyz_euler_to_matrix(
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """
    Build rotation matrix from ZYZ Euler angles.

    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    R = alpha.new_zeros((*alpha.shape, 3, 3))
    R[..., 0, 0] = ca * cb * cg - sa * sg
    R[..., 0, 1] = -ca * cb * sg - sa * cg
    R[..., 0, 2] = ca * sb
    R[..., 1, 0] = sa * cb * cg + ca * sg
    R[..., 1, 1] = -sa * cb * sg + ca * cg
    R[..., 1, 2] = sa * sb
    R[..., 2, 0] = -sb * cg
    R[..., 2, 1] = sb * sg
    R[..., 2, 2] = cb
    return R


class TestDescrptSeZMNet(unittest.TestCase):
    """Test the SeZM-Net descriptor."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def _tiny_system(
        self, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a minimal two-atom system for testing."""
        coord = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=dtype,
            device=self.device,
        ).view(1, -1, 3)
        atype = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        nlist = torch.tensor(
            [[[1, -1], [0, -1]]], dtype=torch.int64, device=self.device
        )
        return coord, atype, nlist

    def test_forward_shape_and_dtype(self) -> None:
        """Test that forward produces correct shape and dtype."""
        for prec, use_parallel in itertools.product(
            ["float64", "float32"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            model = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                ffn_neuron=[16],
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
            )
            self.assertEqual(model.dtype, dtype)
            if use_parallel:
                self.assertIsInstance(model.wigner_calc, WignerDCalcParallel)
            else:
                self.assertIsInstance(model.wigner_calc, WignerDCalc)

            desc, _, _, _, sw = model(
                extended_coord, atype, nlist, mapping=None, comm_dict=None
            )
            self.assertEqual(desc.shape, (1, 2, 8))
            self.assertEqual(desc.dtype, env.GLOBAL_PT_FLOAT_PRECISION)
            self.assertEqual(sw.shape, (1, 2, 2, 1))
            self.assertEqual(sw.dtype, env.GLOBAL_PT_FLOAT_PRECISION)

    def test_backward_gradient(self) -> None:
        """Test backward gradient through coordinates."""
        for prec, use_parallel in itertools.product(
            ["float64", "float32"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
            model = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 0],
                channels=4,
                n_radial=3,
                radial_mlp=[6],
                ffn_neuron=[8],
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
            )
            desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
            loss = desc.sum()
            loss.backward()
            self.assertIsNotNone(extended_coord.grad)
            self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))

    def test_edge_cache_type_features(self) -> None:
        """Test edge cache contains per-edge type embeddings for src/dst."""
        for prec in ["float64", "float32"]:
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)
            nf, nloc, nnei = nlist.shape

            model = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                ffn_neuron=[16],
                use_parallel=False,
                precision=prec,
                trainable=True,
            )
            self.assertEqual(model.dtype, dtype)

            atype_loc = atype[:, :nloc]
            x0 = model.type_embedding(atype_loc).reshape(nf * nloc, model.channels)
            pair_keep_mask = torch.ones_like(
                nlist, dtype=torch.bool, device=self.device
            )

            edge_cache, sw = model._build_edge_cache(
                x0=x0,
                extended_coord=extended_coord.to(dtype=model.dtype),
                extended_atype=atype,
                nlist=nlist,
                mapping=None,
                pair_keep_mask=pair_keep_mask,
            )

            self.assertEqual(sw.shape, (nf, nloc, nnei, 1))
            self.assertEqual(edge_cache.node_type_feat.dtype, model.dtype)
            self.assertEqual(edge_cache.node_type_feat.shape, x0.shape)
            torch.testing.assert_close(edge_cache.node_type_feat, x0)
            self.assertEqual(
                edge_cache.src_type_feat.shape, (edge_cache.num_edges, model.channels)
            )
            self.assertEqual(
                edge_cache.dst_type_feat.shape, (edge_cache.num_edges, model.channels)
            )
            torch.testing.assert_close(edge_cache.src_type_feat, x0[edge_cache.src])
            torch.testing.assert_close(edge_cache.dst_type_feat, x0[edge_cache.dst])
            self.assertFalse(hasattr(edge_cache, "edge_len"))
            self.assertFalse(hasattr(edge_cache, "edge_unit"))
            self.assertFalse(hasattr(edge_cache, "sw"))

    def test_serialization_deserialization(self) -> None:
        """Test serialization and deserialization preserves model state."""
        for prec, use_parallel in itertools.product(
            ["float64", "float32"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            # Create model
            model = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                so2_layers=2,
                ffn_neuron=[16],
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
            )

            # Forward before serialization
            desc1, _, _, _, sw1 = model(extended_coord, atype, nlist)

            # Serialize
            data = model.serialize()

            # Deserialize
            model_restored = DescrptSeZMNet.deserialize(data)

            # Forward after deserialization
            desc2, _, _, _, sw2 = model_restored(extended_coord, atype, nlist)

            if dtype == torch.float32:
                atol = 5e-5
                rtol = 5e-5
            else:
                atol = 1e-10
                rtol = 1e-10

            # Check outputs match
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
        for prec, use_parallel in itertools.product(
            ["float64", "float32"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            seed = 12345

            # Create two models with the same seed
            model1 = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                so2_layers=2,
                ffn_neuron=[16],
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
                seed=seed,
            )

            model2 = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                so2_layers=2,
                ffn_neuron=[16],
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
                seed=seed,
            )

            # Compare parameters
            for (n1, p1), (n2, p2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
                self.assertEqual(n1, n2, msg="Parameter name mismatch")
                torch.testing.assert_close(
                    p1,
                    p2,
                    atol=1e-10 if dtype == torch.float64 else 1e-6,
                    rtol=1e-10 if dtype == torch.float64 else 1e-6,
                    msg=f"Parameter {n1} differs between models with same seed",
                )

            # Compare forward outputs
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            desc1, _, _, _, sw1 = model1(extended_coord, atype, nlist)
            desc2, _, _, _, sw2 = model2(extended_coord, atype, nlist)

            if dtype == torch.float32:
                atol = 5e-5
                rtol = 5e-5
            else:
                atol = 1e-10
                rtol = 1e-10

            torch.testing.assert_close(
                desc1,
                desc2,
                atol=atol,
                rtol=rtol,
                msg="Forward output differs for models with same seed",
            )
            torch.testing.assert_close(
                sw1,
                sw2,
                atol=atol,
                rtol=rtol,
                msg="Smooth weight differs for models with same seed",
            )


class TestInitEdgeRotMatFrisvad(unittest.TestCase):
    """Test the Frisvad edge rotation matrix builder."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float32:
            return 1e-4, 1e-4
        return 1e-10, 1e-10

    def _safe_norm(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)).clamp(min=eps)

    def _assert_rotation_invariants(
        self, rot_mat: torch.Tensor, edge_vec: torch.Tensor
    ) -> None:
        atol, rtol = self._get_tols(edge_vec.dtype)

        # === Step 1. Orthonormality ===
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

        # === Step 2. Alignment (R @ z_hat = e_z) ===
        edge_unit = edge_vec / self._safe_norm(edge_vec)
        ez = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=edge_vec.dtype
        ).expand(n_edge, 3)
        rotated = (rot_mat @ edge_unit.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(rotated, ez, atol=atol, rtol=rtol)

        # === Step 3. Proper rotation (det = +1) ===
        det = torch.linalg.det(rot_mat)
        torch.testing.assert_close(
            det,
            torch.ones_like(det),
            atol=atol,
            rtol=rtol,
        )

    def test_invariants_random_edges(self) -> None:
        for dtype in [torch.float64, torch.float32]:
            edge_vec = torch.randn(512, 3, device=self.device, dtype=dtype)
            rot_mat = init_edge_rot_mat_frisvad(edge_vec)
            self._assert_rotation_invariants(rot_mat, edge_vec)

    def test_invariants_near_minus_z(self) -> None:
        for dtype in [torch.float64, torch.float32]:
            thetas = torch.tensor(
                [0.0, 1.0e-3, 2.0e-3, 1.0e-2], device=self.device, dtype=dtype
            )
            edge_vec = torch.stack(
                [torch.sin(thetas), torch.zeros_like(thetas), -torch.cos(thetas)],
                dim=-1,
            )
            rot_mat = init_edge_rot_mat_frisvad(edge_vec)
            self._assert_rotation_invariants(rot_mat, edge_vec)


class TestWignerDCalc(unittest.TestCase):
    """Test the Wigner-D matrix calculator."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        self.batch = 8
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float32:
            return 5e-5, 5e-5
        return 1e-10, 1e-10

    def test_orthogonality(self) -> None:
        """Test D @ D^T = I for random rotations."""
        for dtype, wigner_cls, lmax in itertools.product(
            [torch.float64, torch.float32],
            [WignerDCalc, WignerDCalcParallel],
            [1, 2, 3],
        ):
            atol, rtol = self._get_tols(dtype)
            wigner = wigner_cls(lmax=lmax, dtype=dtype)
            alpha = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            beta = torch.rand(self.batch, device=self.device, dtype=dtype) * 3.14159
            gamma = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            rot = _zyz_euler_to_matrix(alpha, beta, gamma)
            D_list, Dt_list = wigner(rot)

            for l in range(lmax + 1):
                dim = 2 * l + 1
                eye = torch.eye(dim, device=self.device, dtype=dtype).expand(
                    self.batch, dim, dim
                )
                prod = D_list[l] @ Dt_list[l]
                torch.testing.assert_close(
                    prod,
                    eye,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Orthogonality failed for {wigner_cls.__name__}, dtype={dtype}, lmax={lmax}, l={l}"
                    ),
                )

    def test_group_property(self) -> None:
        """Test group property: D(R1 @ R2) ~= D(R1) @ D(R2)."""
        for dtype, wigner_cls, lmax in itertools.product(
            [torch.float64, torch.float32],
            [WignerDCalc, WignerDCalcParallel],
            [1, 2, 3],
        ):
            if dtype == torch.float32:
                atol = 5e-4
                rtol = 5e-4
            else:
                atol = 1e-10
                rtol = 1e-10

            wigner = wigner_cls(lmax=lmax, dtype=dtype)

            # Avoid gimbal lock by keeping beta away from 0 and pi.
            alpha1 = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            beta1 = 0.2 + torch.rand(self.batch, device=self.device, dtype=dtype) * (
                3.14159 - 0.4
            )
            gamma1 = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )

            alpha2 = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            beta2 = 0.2 + torch.rand(self.batch, device=self.device, dtype=dtype) * (
                3.14159 - 0.4
            )
            gamma2 = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )

            rot1 = _zyz_euler_to_matrix(alpha1, beta1, gamma1)
            rot2 = _zyz_euler_to_matrix(alpha2, beta2, gamma2)
            rot12 = rot1 @ rot2

            D1, _ = wigner(rot1)
            D2, _ = wigner(rot2)
            D12, _ = wigner(rot12)

            for l in range(lmax + 1):
                torch.testing.assert_close(
                    D12[l],
                    D1[l] @ D2[l],
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"Group property failed for {wigner_cls.__name__}, dtype={dtype}, lmax={lmax}, l={l}"
                    ),
                )

    def test_parallel_matches_per_l(self) -> None:
        """Test that the parallel implementation matches the per-l implementation."""
        for dtype, lmax in itertools.product([torch.float64, torch.float32], [1, 2, 3]):
            atol, rtol = self._get_tols(dtype)
            wigner_ref = WignerDCalc(lmax=lmax, dtype=dtype)
            wigner_par = WignerDCalcParallel(lmax=lmax, dtype=dtype)

            alpha = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            beta = 0.1 + torch.rand(self.batch, device=self.device, dtype=dtype) * (
                3.14159 - 0.2
            )
            gamma = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            rot = _zyz_euler_to_matrix(alpha, beta, gamma)

            D_ref, Dt_ref = wigner_ref(rot)
            D_par, Dt_par = wigner_par(rot)

            for l in range(lmax + 1):
                torch.testing.assert_close(
                    D_par[l],
                    D_ref[l],
                    atol=atol,
                    rtol=rtol,
                    msg=f"D mismatch for dtype={dtype}, lmax={lmax}, l={l}",
                )
                torch.testing.assert_close(
                    Dt_par[l],
                    Dt_ref[l],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Dt mismatch for dtype={dtype}, lmax={lmax}, l={l}",
                )

    def test_z_rotation_m0_invariance(self) -> None:
        """Test that m=0 component is invariant under z-rotation."""
        for dtype in [torch.float64, torch.float32]:
            atol, rtol = self._get_tols(dtype)
            lmax = 3
            wigner = WignerDCalc(lmax=lmax, dtype=dtype)
            n_edges = 5
            angles = torch.rand(n_edges, device=self.device, dtype=dtype) * 2 * 3.14159

            # Test each l separately
            for l in range(lmax + 1):
                Z = wigner._build_z_rotation(angles, l)
                # Shape: (n_edges, 2l+1, 2l+1)
                dim = 2 * l + 1
                self.assertEqual(Z.shape, (n_edges, dim, dim))

                # m=0 index within the block is at position l (center)
                m0_idx = l

                # Check diagonal element is 1
                diag_vals = Z[:, m0_idx, m0_idx]
                torch.testing.assert_close(
                    diag_vals,
                    torch.ones(n_edges, device=self.device, dtype=dtype),
                    atol=atol,
                    rtol=rtol,
                    msg=f"m=0 diagonal should be 1 for dtype={dtype}, l={l}",
                )

                # Check off-diagonal elements in m=0 row/col are 0
                for j in range(dim):
                    if j != m0_idx:
                        row_vals = Z[:, m0_idx, j]
                        col_vals = Z[:, j, m0_idx]
                        torch.testing.assert_close(
                            row_vals,
                            torch.zeros(n_edges, device=self.device, dtype=dtype),
                            atol=atol,
                            rtol=rtol,
                            msg=f"m=0 row off-diagonal should be 0 for dtype={dtype}, l={l}, j={j}",
                        )
                        torch.testing.assert_close(
                            col_vals,
                            torch.zeros(n_edges, device=self.device, dtype=dtype),
                            atol=atol,
                            rtol=rtol,
                            msg=f"m=0 col off-diagonal should be 0 for dtype={dtype}, l={l}, j={j}",
                        )

    def test_l1_matches_vector_representation(self) -> None:
        """
        Test that the l=1 real-basis block matches the 3D rotation matrix.

        For l=1, the irrep is equivalent to the vector representation, up to a
        fixed basis transform between Cartesian (x, y, z) and the chosen real SH
        ordering (m=-1,0,+1) with the implementation's phase conventions.
        """
        # === Step 1. Define the fixed Cartesian <-> real-SH basis map (l=1) ===
        # With the real SH conventions used in WignerDCalc, the mapping is a signed
        # permutation:
        #   x_sh = S @ v_cart
        #   v_cart = S^T @ x_sh
        for dtype, wigner_cls in itertools.product(
            [torch.float64, torch.float32],
            [WignerDCalc, WignerDCalcParallel],
        ):
            atol, rtol = self._get_tols(dtype)
            S = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                device=self.device,
                dtype=dtype,
            )
            S_batch = S.unsqueeze(0).expand(self.batch, 3, 3)

            wigner = wigner_cls(lmax=1, dtype=dtype)

            alpha = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            beta = 0.1 + torch.rand(self.batch, device=self.device, dtype=dtype) * (
                3.14159 - 0.2
            )
            gamma = (
                torch.rand(self.batch, device=self.device, dtype=dtype) * 2 * 3.14159
            )
            rot = _zyz_euler_to_matrix(alpha, beta, gamma)

            D_list, Dt_list = wigner(rot)
            D1 = D_list[1]
            Dt1 = Dt_list[1]

            # === Step 2. Compare against the vector representation ===
            expected = S_batch @ rot @ S_batch.transpose(-1, -2)
            torch.testing.assert_close(
                D1,
                expected,
                atol=atol,
                rtol=rtol,
                msg=f"l=1 block mismatch for {wigner_cls.__name__}, dtype={dtype}",
            )
            torch.testing.assert_close(
                Dt1,
                expected.transpose(-1, -2),
                atol=atol,
                rtol=rtol,
                msg=f"l=1 transpose block mismatch for {wigner_cls.__name__}, dtype={dtype}",
            )

    def test_edge_frame_m0_column_matches_edge_direction(self) -> None:
        """
        Test that the local m=0 basis rotated to global matches the edge direction.

        The edge frame is built so that ``rot_mat @ edge_unit = (0,0,1)`` (global->local).
        Therefore, the local->global transform is ``rot_mat^T`` and should map the local
        m=0 (l=1) axis to the (signed) edge direction in Cartesian space.
        """
        # === Step 1. Fixed l=1 real-SH -> Cartesian map ===
        for dtype in [torch.float64, torch.float32]:
            atol, rtol = self._get_tols(dtype)
            S = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                device=self.device,
                dtype=dtype,
            )
            St = S.transpose(0, 1)

            # === Step 2. Build edge-aligned frames and Wigner-D blocks ===
            n_edges = 128
            edge_vec = torch.randn(n_edges, 3, device=self.device, dtype=dtype)
            edge_unit = edge_vec / torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
            rot_mat = init_edge_rot_mat_frisvad(edge_vec)

            wigner = WignerDCalc(lmax=1, dtype=dtype)
            _, Dt_list = wigner(rot_mat)

            # === Step 3. Column m=0 of Dt^{(1)} equals the rotated local m=0 axis ===
            # In the implementation's real-SH convention, the Cartesian axis corresponding to
            # (l=1, m=0) is -z. Therefore the rotated vector is -edge_unit.
            m0 = 1
            col = Dt_list[1][:, :, m0]  # (E, 3) in real-SH basis
            vec_cart = torch.einsum("ij,ej->ei", St, col)
            torch.testing.assert_close(
                vec_cart,
                -edge_unit,
                atol=atol,
                rtol=rtol,
                msg=f"Dt_list[1][:,:,m0] does not match -edge_unit in Cartesian space (dtype={dtype})",
            )


class TestSO2LinearEquivariance(unittest.TestCase):
    """Test SO2Linear z-rotation equivariance: SO2Linear(Z @ x) = Z @ SO2Linear(x)."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(42)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float32:
            return 1e-5, 1e-5
        return 1e-10, 1e-10

    def _build_full_z_rotation(self, angles: torch.Tensor, lmax: int) -> torch.Tensor:
        """
        Build block-diagonal z-rotation matrix for full SO(3) space.

        Parameters
        ----------
        angles
            Rotation angles with shape (batch,).
        lmax
            Maximum order.

        Returns
        -------
        torch.Tensor
            Block-diagonal Z matrix with shape (batch, ebed_dim, ebed_dim).
        """
        batch = angles.shape[0]
        ebed_dim = (lmax + 1) ** 2
        Z_full = angles.new_zeros(batch, ebed_dim, ebed_dim)

        offset = 0
        for l in range(lmax + 1):
            dim = 2 * l + 1
            # Build per-l z-rotation block
            Z_l = angles.new_zeros(batch, dim, dim)

            # m=0: diagonal element is 1
            m0_idx = l
            Z_l[:, m0_idx, m0_idx] = 1.0

            # m != 0: 2x2 rotation blocks
            for m in range(1, l + 1):
                cos_m = torch.cos(m * angles)
                sin_m = torch.sin(m * angles)
                pos_idx = l + m  # +m index within block
                neg_idx = l - m  # -m index within block

                # 2x2 block: [[cos, -sin], [sin, cos]] for (neg, pos) = (Re, Im)
                Z_l[:, neg_idx, neg_idx] = cos_m
                Z_l[:, neg_idx, pos_idx] = -sin_m
                Z_l[:, pos_idx, neg_idx] = sin_m
                Z_l[:, pos_idx, pos_idx] = cos_m

            Z_full[:, offset : offset + dim, offset : offset + dim] = Z_l
            offset += dim

        return Z_full

    def test_equivariance_random_angles(self) -> None:
        """Test SO2Linear(Z @ x) = Z @ SO2Linear(x) for random z-rotations."""
        for dtype, lmax in itertools.product([torch.float64, torch.float32], [1, 2, 3]):
            atol, rtol = self._get_tols(dtype)
            batch = 16
            channels_in = 8
            channels_out = 12
            ebed_dim = (lmax + 1) ** 2

            so2_linear = SO2Linear(
                lmax=lmax,
                in_channels=channels_in,
                out_channels=channels_out,
                dtype=dtype,
                seed=None,
                trainable=True,
            )

            # Random input features
            x = torch.randn(
                batch, ebed_dim, channels_in, device=self.device, dtype=dtype
            )

            # Random z-rotation angles
            angles = torch.rand(batch, device=self.device, dtype=dtype) * 2 * 3.14159
            Z = self._build_full_z_rotation(angles, lmax)

            # Rotate input: Z @ x (apply rotation to each channel)
            # Z has shape (batch, ebed_dim, ebed_dim), x has shape (batch, ebed_dim, C)
            x_rotated = torch.einsum("bij,bjc->bic", Z, x)

            # Compute both sides of equivariance relation
            lhs = so2_linear(x_rotated)  # SO2Linear(Z @ x)
            rhs = torch.einsum("bij,bjc->bic", Z, so2_linear(x))  # Z @ SO2Linear(x)

            torch.testing.assert_close(
                lhs,
                rhs,
                atol=atol,
                rtol=rtol,
                msg=f"SO2Linear equivariance failed for dtype={dtype}, lmax={lmax}",
            )


class TestPerDegreeLinearV2(unittest.TestCase):
    """Test PerDegreeLinearV2 correctness and consistency with PerDegreeLinear."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(42)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float32:
            return 1e-5, 1e-5
        return 1e-10, 1e-10

    def test_bias_only_on_l0(self) -> None:
        """Test that bias is only applied to l=0 (scalar) components."""
        for dtype in [torch.float64, torch.float32]:
            atol, rtol = self._get_tols(dtype)
            lmax = 2
            channels = 4
            ebed_dim = _so3_dim_of_lmax(lmax)
            batch = 8

            # === Step 1. Create layer with non-zero bias ===
            layer = PerDegreeLinearV2(
                lmax=lmax,
                channels=channels,
                dtype=dtype,
                trainable=True,
            )
            layer.bias.data.fill_(1.5)
            layer.weight.data.zero_()

            # === Step 2. Input with zeros ===
            x = torch.zeros(batch, ebed_dim, channels, device=self.device, dtype=dtype)

            # === Step 3. Output should have bias only at l=0 ===
            out = layer(x)

            # l=0 component (index 0) should have bias
            torch.testing.assert_close(
                out[:, 0, :],
                torch.full((batch, channels), 1.5, device=self.device, dtype=dtype),
                atol=atol,
                rtol=rtol,
            )

            # l>0 components should be zero (no bias)
            torch.testing.assert_close(
                out[:, 1:, :],
                torch.zeros(
                    batch, ebed_dim - 1, channels, device=self.device, dtype=dtype
                ),
                atol=atol,
                rtol=rtol,
            )

    def test_consistency_with_per_order_linear(self) -> None:
        """Test that PerDegreeLinearV2 produces the same output as PerDegreeLinear."""
        for dtype, lmax in itertools.product([torch.float64, torch.float32], [1, 2, 3]):
            atol, rtol = self._get_tols(dtype)
            channels = 8
            ebed_dim = _so3_dim_of_lmax(lmax)
            batch = 16

            # Create both layers
            layer_v1 = PerDegreeLinear(
                lmax=lmax,
                channels=channels,
                dtype=dtype,
                trainable=True,
            )
            layer_v2 = PerDegreeLinearV2(
                lmax=lmax,
                channels=channels,
                dtype=dtype,
                trainable=True,
            )

            # Copy weights from v1 to v2
            # Note: MLPLayer.matrix has shape [in, out], while PerDegreeLinearV2.weight[l] has shape [out, in]
            for l in range(lmax + 1):
                layer_v2.weight.data[l] = layer_v1.linears[l].matrix.data.t().clone()
            if layer_v1.linears[0].bias is not None:
                layer_v2.bias.data = layer_v1.linears[0].bias.data.clone()

            # Random input
            x = torch.randn(batch, ebed_dim, channels, device=self.device, dtype=dtype)

            # Compare outputs
            out_v1 = layer_v1(x)
            out_v2 = layer_v2(x)

            torch.testing.assert_close(
                out_v1,
                out_v2,
                atol=atol,
                rtol=rtol,
                msg=f"PerDegreeLinearV2 output mismatch for dtype={dtype}, lmax={lmax}",
            )


if __name__ == "__main__":
    unittest.main()
