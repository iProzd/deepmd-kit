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
    SO2Convolution,
    SO2Linear,
)
from deepmd.pt.model.descriptor.se_zm_helper import (
    EdgeFeatureCache,
    WignerDCalc,
    WignerDCalcParallel,
    build_m_major_index,
    edge_cache_to_dtype,
)
from deepmd.pt.model.descriptor.se_zm_net import (
    DescrptSeZMNet,
    init_edge_rot_mat_frisvad,
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
            ["float64", "float32", "bfloat16"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            model = DescrptSeZMNet(
                rcut=3.0,
                sel=[1, 1],
                ntypes=2,
                l_schedule=[1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                ffn_neurons=16,
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
            ["float64", "float32", "bfloat16"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)
            model = DescrptSeZMNet(
                rcut=3.0,
                sel=[1, 1],
                ntypes=2,
                l_schedule=[1, 0],
                channels=4,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
            )
            desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
            loss = desc.sum()
            loss.backward()
            self.assertIsNotNone(extended_coord.grad)
            self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))

    def test_serialization_deserialization(self) -> None:
        """Test serialization and deserialization preserves model state."""
        for prec, use_parallel in itertools.product(
            ["float64", "float32", "bfloat16"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            # Create model
            model = DescrptSeZMNet(
                rcut=3.0,
                sel=[1, 1],
                ntypes=2,
                l_schedule=[1, 1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                so2_layers=2,
                ffn_neurons=16,
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

            if dtype == torch.float64:
                atol = 1e-10
                rtol = 1e-10
            elif dtype == torch.float32:
                atol = 5e-5
                rtol = 5e-5
            else:
                atol = 5e-3
                rtol = 5e-3

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
            ["float64", "float32", "bfloat16"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            seed = 12345

            # Create two models with the same seed
            model1 = DescrptSeZMNet(
                rcut=3.0,
                sel=[1, 1],
                ntypes=2,
                l_schedule=[1, 1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                so2_layers=2,
                ffn_neurons=16,
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
                seed=seed,
            )

            model2 = DescrptSeZMNet(
                rcut=3.0,
                sel=[1, 1],
                ntypes=2,
                l_schedule=[1, 1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                so2_layers=2,
                ffn_neurons=16,
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
                if dtype == torch.float64:
                    atol = 1e-10
                    rtol = 1e-10
                elif dtype == torch.float32:
                    atol = 1e-6
                    rtol = 1e-6
                else:
                    atol = 1e-3
                    rtol = 1e-3
                torch.testing.assert_close(
                    p1,
                    p2,
                    atol=atol,
                    rtol=rtol,
                    msg=f"Parameter {n1} differs between models with same seed",
                )

            # Compare forward outputs
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            desc1, _, _, _, sw1 = model1(extended_coord, atype, nlist)
            desc2, _, _, _, sw2 = model2(extended_coord, atype, nlist)

            if dtype == torch.float64:
                atol = 1e-10
                rtol = 1e-10
            elif dtype == torch.float32:
                atol = 5e-5
                rtol = 5e-5
            else:
                atol = 5e-3
                rtol = 5e-3

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

    def _set_identity_so2(self, conv: SO2Convolution) -> None:
        linear_m0 = conv.so2_linears[0].linear_m0
        eye = torch.eye(conv.channels, device=self.device, dtype=conv.dtype)
        linear_m0.matrix.data.copy_(eye)
        if linear_m0.bias is not None:
            linear_m0.bias.data.zero_()
        conv.so2_linears[0].bias0.data.zero_()
        weight = conv.so3_linear.weight
        weight.data.zero_()
        weight.data[0].copy_(
            torch.eye(conv.channels, device=self.device, dtype=conv.dtype)
        )
        conv.so3_linear.bias.data.zero_()

    def test_so2_conv_step5_no_attention_matches_scatter(self) -> None:
        dtype = torch.float32
        conv = SO2Convolution(
            lmax=0,
            mmax=0,
            channels=4,
            so2_layers=1,
            n_atten_head=0,
            use_parallel=False,
            dtype=dtype,
            seed=None,
            trainable=True,
        )
        self._set_identity_so2(conv)

        x = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
            device=self.device,
            dtype=dtype,
        )
        radial_feat = torch.ones(2, 1, 4, device=self.device, dtype=dtype)
        src = torch.tensor([0, 1], device=self.device, dtype=torch.long)
        dst = torch.tensor([1, 0], device=self.device, dtype=torch.long)
        edge_env = torch.tensor([[0.5], [0.25]], device=self.device, dtype=dtype)
        inv_sqrt_deg = torch.tensor([[[1.0]], [[0.5]]], device=self.device, dtype=dtype)
        D_list = [torch.ones(2, 1, 1, device=self.device, dtype=dtype)]
        Dt_list = [torch.ones(2, 1, 1, device=self.device, dtype=dtype)]
        edge_cache = EdgeFeatureCache(
            src=src,
            dst=dst,
            edge_type_feat=torch.zeros(2, 4, device=self.device, dtype=dtype),
            edge_vec=torch.zeros(2, 3, device=self.device, dtype=dtype),
            edge_rbf=torch.zeros(2, 1, device=self.device, dtype=dtype),
            edge_env=edge_env,
            D_list=D_list,
            Dt_list=Dt_list,
            inv_sqrt_deg=inv_sqrt_deg,
            D_full=None,
            Dt_full=None,
            D_to_m_cache={},
            Dt_from_m_cache={},
        )

        out = conv(x, edge_cache, radial_feat)

        x_global = x[src] * edge_env.view(-1, 1, 1)
        expected = torch.zeros_like(x)
        expected.index_add_(0, dst, x_global)
        expected = expected * inv_sqrt_deg
        torch.testing.assert_close(out, expected)

    def test_so2_conv_step5_head_attention(self) -> None:
        dtype = torch.float32
        conv = SO2Convolution(
            lmax=0,
            mmax=0,
            channels=4,
            so2_layers=1,
            n_atten_head=2,
            use_parallel=False,
            dtype=dtype,
            seed=None,
            trainable=True,
        )
        self._set_identity_so2(conv)
        assert conv.proj_dst is not None
        assert conv.proj_rad is not None
        assert conv.proj_head is not None
        assert conv.proj_msg is not None
        assert conv.norm_dst_for_gate is not None
        assert conv.norm_msg_for_gate is not None
        conv.proj_dst.matrix.data.fill_(1.0)
        conv.proj_rad.matrix.data.fill_(1.0)
        conv.proj_head.matrix.data.fill_(1.0)
        conv.proj_msg.matrix.data.fill_(1.0)
        if conv.proj_dst.bias is not None:
            conv.proj_dst.bias.data.zero_()
        if conv.proj_rad.bias is not None:
            conv.proj_rad.bias.data.zero_()
        if conv.proj_head.bias is not None:
            conv.proj_head.bias.data.zero_()
        if conv.proj_msg.bias is not None:
            conv.proj_msg.bias.data.zero_()
        if conv.msg_gate_scale_raw is not None:
            conv.msg_gate_scale_raw.data.fill_(0.0)

        x = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
            device=self.device,
            dtype=dtype,
        )
        radial_feat = torch.ones(2, 1, 4, device=self.device, dtype=dtype)
        src = torch.tensor([0, 1], device=self.device, dtype=torch.long)
        dst = torch.tensor([1, 0], device=self.device, dtype=torch.long)
        edge_env = torch.tensor([[0.5], [0.25]], device=self.device, dtype=dtype)
        inv_sqrt_deg = torch.tensor([[[1.0]], [[0.5]]], device=self.device, dtype=dtype)
        D_list = [torch.ones(2, 1, 1, device=self.device, dtype=dtype)]
        Dt_list = [torch.ones(2, 1, 1, device=self.device, dtype=dtype)]
        edge_cache = EdgeFeatureCache(
            src=src,
            dst=dst,
            edge_type_feat=torch.zeros(2, 4, device=self.device, dtype=dtype),
            edge_vec=torch.zeros(2, 3, device=self.device, dtype=dtype),
            edge_rbf=torch.zeros(2, 1, device=self.device, dtype=dtype),
            edge_env=edge_env,
            D_list=D_list,
            Dt_list=Dt_list,
            inv_sqrt_deg=inv_sqrt_deg,
            D_full=None,
            Dt_full=None,
        )

        out = conv(x, edge_cache, radial_feat)

        x_global = x[src]
        x_l0 = x[:, 0, :].contiguous()  # (N, C) in self.dtype
        radial_l0 = radial_feat[:, 0, :].contiguous()  # (E, C) in self.dtype
        msg_l0 = x_global[:, 0, :].contiguous()
        # Edge gate: dst normalized, radial unchanged, msg normalized with bounded feedback
        b = conv.proj_dst(conv.norm_dst_for_gate(x_l0))
        s = conv.proj_rad(radial_l0)
        m = conv.proj_msg(conv.norm_msg_for_gate(msg_l0))
        msg_gate_scale = conv.msg_gate_scale_max * torch.sigmoid(
            conv.msg_gate_scale_raw
        )
        edge_logits = b.index_select(0, dst) + s + msg_gate_scale.to(dtype) * m
        g = 2.0 * torch.sigmoid(edge_logits.to(torch.float32).clamp(-6.0, 6.0)).to(
            dtype
        )
        w = edge_env.view(-1, 1) * g
        head_dim = conv.head_dim
        assert head_dim is not None
        msg = x_global.view(2, 1, conv.n_atten_head, head_dim) * w.view(
            -1, 1, conv.n_atten_head, 1
        )
        expected = torch.zeros(
            2, 1, conv.n_atten_head, head_dim, device=self.device, dtype=torch.float32
        )
        expected.index_add_(0, dst, msg.to(torch.float32))
        # Head gate: 2 * sigmoid with normalization, fp32 sigmoid + clamp
        head_logits = conv.proj_head(conv.norm_dst_for_head(x_l0))
        alpha = 2.0 * torch.sigmoid(head_logits.to(torch.float32).clamp(-6.0, 6.0))
        gamma_eff = 2.0 * torch.sigmoid(conv.gamma_head_raw)
        alpha = alpha * gamma_eff.view(1, -1)
        expected = expected * alpha.view(-1, 1, conv.n_atten_head, 1)
        # Apply degree normalization (consistent with baseline path)
        expected = expected.view(2, 1, conv.channels) * inv_sqrt_deg.to(torch.float32)
        expected = expected.to(dtype)

        torch.testing.assert_close(out, expected)

    def test_so2_conv_head_dim_validation(self) -> None:
        with self.assertRaises(ValueError):
            SO2Convolution(
                lmax=0,
                mmax=0,
                channels=3,
                so2_layers=1,
                n_atten_head=2,
                use_parallel=False,
                dtype=torch.float32,
                seed=None,
                trainable=True,
            )


class TestInitEdgeRotMatFrisvad(unittest.TestCase):
    """Test the Frisvad edge rotation matrix builder."""

    def setUp(self) -> None:
        self.device = env.DEVICE
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
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 5e-5, 5e-5
        return 5e-3, 5e-3

    def _extract_l_block(
        self,
        D_list: list[torch.Tensor],
        D_full: torch.Tensor | None,
        l: int,
    ) -> torch.Tensor:
        """Extract the l-block from either D_list or D_full."""
        if D_list:
            return D_list[l]
        s, e = l * l, (l + 1) * (l + 1)
        return D_full[:, s:e, s:e]

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
            D_list, Dt_list, D_full, Dt_full = wigner(rot)

            for l in range(lmax + 1):
                dim = 2 * l + 1
                eye = torch.eye(dim, device=self.device, dtype=dtype).expand(
                    self.batch, dim, dim
                )
                D_l = self._extract_l_block(D_list, D_full, l)
                Dt_l = self._extract_l_block(Dt_list, Dt_full, l)
                prod = D_l @ Dt_l
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
            if dtype == torch.float64:
                atol = 1e-10
                rtol = 1e-10
            else:
                atol = 5e-4
                rtol = 5e-4

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

            D1, _, D1_full, _ = wigner(rot1)
            D2, _, D2_full, _ = wigner(rot2)
            D12, _, D12_full, _ = wigner(rot12)

            for l in range(lmax + 1):
                D1_l = self._extract_l_block(D1, D1_full, l)
                D2_l = self._extract_l_block(D2, D2_full, l)
                D12_l = self._extract_l_block(D12, D12_full, l)
                torch.testing.assert_close(
                    D12_l,
                    D1_l @ D2_l,
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

            D_ref, Dt_ref, _, _ = wigner_ref(rot)
            _, _, D_full_par, Dt_full_par = wigner_par(rot)

            for l in range(lmax + 1):
                D_par_l = self._extract_l_block([], D_full_par, l)
                Dt_par_l = self._extract_l_block([], Dt_full_par, l)
                torch.testing.assert_close(
                    D_par_l,
                    D_ref[l],
                    atol=atol,
                    rtol=rtol,
                    msg=f"D mismatch for dtype={dtype}, lmax={lmax}, l={l}",
                )
                torch.testing.assert_close(
                    Dt_par_l,
                    Dt_ref[l],
                    atol=atol,
                    rtol=rtol,
                    msg=f"Dt mismatch for dtype={dtype}, lmax={lmax}, l={l}",
                )

    def test_backward_no_nan_at_gimbal_lock(self) -> None:
        """Test backward stability for rotations with beta=0 or pi."""
        for dtype, wigner_cls, lmax in itertools.product(
            [torch.float64, torch.float32],
            [WignerDCalc, WignerDCalcParallel],
            [1, 2, 3],
        ):
            wigner = wigner_cls(lmax=lmax, dtype=dtype)
            edge_vec = torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            rot_mat = init_edge_rot_mat_frisvad(edge_vec)
            D_list, _, D_full, _ = wigner(rot_mat)
            # Use D_full for parallel mode, D_list for per-l mode.
            if D_list:
                loss = sum((D**2).sum() for D in D_list)
            else:
                loss = (D_full**2).sum()
            loss.backward()

            self.assertIsNotNone(edge_vec.grad)
            self.assertTrue(
                torch.isfinite(edge_vec.grad).all().item(),
                msg=(
                    f"Non-finite gradients at gimbal lock for {wigner_cls.__name__}, dtype={dtype}, lmax={lmax}"
                ),
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

            D_list, Dt_list, D_full, Dt_full = wigner(rot)
            D1 = self._extract_l_block(D_list, D_full, 1)
            Dt1 = self._extract_l_block(Dt_list, Dt_full, 1)

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
            edge_norm = torch.sqrt(
                torch.sum(edge_vec * edge_vec, dim=-1, keepdim=True).clamp_min(
                    torch.finfo(edge_vec.dtype).eps
                )
            )
            edge_unit = edge_vec / edge_norm
            rot_mat = init_edge_rot_mat_frisvad(edge_vec)

            wigner = WignerDCalc(lmax=1, dtype=dtype)
            _, Dt_list, _, _ = wigner(rot_mat)

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


class TestEdgeFeatureCacheProjection(unittest.TestCase):
    """Test EdgeFeatureCache caches projected D matrices."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_parallel_projection_cached(self) -> None:
        lmax = 2
        mmax = 1
        dtype = torch.float32
        channels = 3
        n_edges = 4
        D_full_dim = (lmax + 1) ** 2
        D_full = torch.randn(
            n_edges, D_full_dim, D_full_dim, device=self.device, dtype=dtype
        )
        Dt_full = D_full.transpose(1, 2)
        coeff_index_m = build_m_major_index(lmax, mmax, device=self.device)
        cache = EdgeFeatureCache(
            src=torch.arange(n_edges, device=self.device),
            dst=torch.zeros(n_edges, dtype=torch.long, device=self.device),
            edge_type_feat=torch.zeros(
                n_edges, channels, device=self.device, dtype=dtype
            ),
            edge_vec=torch.zeros(n_edges, 3, device=self.device, dtype=dtype),
            edge_rbf=torch.zeros(n_edges, 1, device=self.device, dtype=dtype),
            edge_env=torch.ones(n_edges, 1, device=self.device, dtype=dtype),
            D_list=[
                torch.zeros(
                    n_edges, 2 * l + 1, 2 * l + 1, device=self.device, dtype=dtype
                )
                for l in range(lmax + 1)
            ],
            Dt_list=[
                torch.zeros(
                    n_edges, 2 * l + 1, 2 * l + 1, device=self.device, dtype=dtype
                )
                for l in range(lmax + 1)
            ],
            inv_sqrt_deg=torch.ones(1, 1, 1, device=self.device, dtype=dtype),
            D_full=D_full,
            Dt_full=Dt_full,
            D_to_m_cache={},
            Dt_from_m_cache={},
        )

        D_to_m_first = cache.get_D_to_m(
            ebed_dim_full=D_full_dim,
            coeff_index_m=coeff_index_m,
            key_lmax=lmax,
            key_mmax=mmax,
        )
        D_to_m_second = cache.get_D_to_m(
            ebed_dim_full=D_full_dim,
            coeff_index_m=coeff_index_m,
            key_lmax=lmax,
            key_mmax=mmax,
        )
        Dt_from_m_first = cache.get_Dt_from_m(
            ebed_dim_full=D_full_dim,
            coeff_index_m=coeff_index_m,
            key_lmax=lmax,
            key_mmax=mmax,
        )
        Dt_from_m_second = cache.get_Dt_from_m(
            ebed_dim_full=D_full_dim,
            coeff_index_m=coeff_index_m,
            key_lmax=lmax,
            key_mmax=mmax,
        )

        expected_D_to_m = D_full[:, :D_full_dim, :D_full_dim].index_select(
            1, coeff_index_m
        )
        expected_Dt_from_m = Dt_full[:, :D_full_dim, :D_full_dim].index_select(
            2, coeff_index_m
        )

        self.assertIs(D_to_m_first, D_to_m_second)
        self.assertIs(Dt_from_m_first, Dt_from_m_second)
        torch.testing.assert_close(D_to_m_first, expected_D_to_m)
        torch.testing.assert_close(Dt_from_m_first, expected_Dt_from_m)


class TestSO2LinearEquivariance(unittest.TestCase):
    """Test SO2Linear z-rotation equivariance: SO2Linear(Z @ x) = Z @ SO2Linear(x)."""

    def setUp(self) -> None:
        self.device = env.DEVICE
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
                batch, dim_red, channels_in, device=self.device, dtype=dtype
            )

            angles = torch.rand(batch, device=self.device, dtype=dtype) * 2 * 3.14159
            Z = self._build_m_major_z_rotation(angles, lmax, mmax)

            x_rotated = torch.einsum("bij,bjc->bic", Z, x)
            lhs = so2_linear(x_rotated)
            rhs = torch.einsum("bij,bjc->bic", Z, so2_linear(x))

            torch.testing.assert_close(
                lhs,
                rhs,
                atol=atol,
                rtol=rtol,
                msg=f"SO2Linear equivariance failed for dtype={dtype}, lmax={lmax}, mmax={mmax}",
            )


class TestSO2ConvolutionReducedRotation(unittest.TestCase):
    """Test that reduced rotate/back matches full rotate/back with zero-filled truncation."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(0)

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 5e-5, 5e-5
        # bf16 has only 7-bit mantissa; use looser tolerance.
        return 2e-2, 2e-2

    def test_reduced_rotation_matches_full(self) -> None:
        lmax = 3
        mmax = 1
        channels = 6
        so2_layers = 2
        n_nodes = 12
        n_edges = 32

        for dtype, use_parallel in itertools.product(
            [torch.float64, torch.float32, torch.bfloat16],
            [False, True],
        ):
            atol, rtol = self._get_tols(dtype)
            D_full = (lmax + 1) ** 2

            # === Step 1. Build random rotations and Wigner-D blocks ===
            alpha = torch.rand(n_edges, device=self.device, dtype=dtype) * 2 * 3.14159
            beta = 0.2 + torch.rand(n_edges, device=self.device, dtype=dtype) * (
                3.14159 - 0.4
            )
            gamma = torch.rand(n_edges, device=self.device, dtype=dtype) * 2 * 3.14159
            rot = _zyz_euler_to_matrix(alpha, beta, gamma)

            if use_parallel:
                wigner = WignerDCalcParallel(lmax=lmax, dtype=dtype)
                D_list, Dt_list, D_full_mat, Dt_full_mat = wigner(rot)
                assert D_full_mat is not None
                assert Dt_full_mat is not None
            else:
                wigner = WignerDCalc(lmax=lmax, dtype=dtype)
                D_list, Dt_list, D_full_mat, Dt_full_mat = wigner(rot)
                self.assertIsNone(D_full_mat)
                self.assertIsNone(Dt_full_mat)

            # === Step 2. Synthetic graph and cached invariants ===
            src = torch.randint(0, n_nodes, (n_edges,), device=self.device)
            dst = torch.randint(0, n_nodes, (n_edges,), device=self.device)

            node_type_feat = torch.randn(
                n_nodes, channels, device=self.device, dtype=dtype
            )
            edge_type_feat = node_type_feat.index_select(
                0, src
            ) + node_type_feat.index_select(0, dst)
            edge_env = torch.rand(n_edges, 1, device=self.device, dtype=dtype)
            inv_sqrt_deg = torch.ones(n_nodes, 1, 1, device=self.device, dtype=dtype)
            edge_cache = EdgeFeatureCache(
                src=src,
                dst=dst,
                edge_type_feat=edge_type_feat,
                edge_vec=torch.zeros(n_edges, 3, device=self.device, dtype=dtype),
                edge_rbf=torch.zeros(n_edges, 1, device=self.device, dtype=dtype),
                edge_env=edge_env,
                D_list=D_list,
                Dt_list=Dt_list,
                inv_sqrt_deg=inv_sqrt_deg,
                D_full=D_full_mat,
                Dt_full=Dt_full_mat,
                D_to_m_cache={},
                Dt_from_m_cache={},
            )
            # WignerDCalc forces fp32+, convert cache to target dtype
            edge_cache = edge_cache_to_dtype(edge_cache, dtype)
            # Update references from converted cache for Step 4
            D_list = edge_cache.D_list
            Dt_list = edge_cache.Dt_list
            D_full_mat = edge_cache.D_full
            Dt_full_mat = edge_cache.Dt_full

            radial_feat = torch.randn(
                n_edges, lmax + 1, channels, device=self.device, dtype=dtype
            )
            radial_feat = radial_feat + edge_type_feat.unsqueeze(1)

            # Node features (global packed layout)
            x = torch.randn(n_nodes, D_full, channels, device=self.device, dtype=dtype)

            so2_conv = SO2Convolution(
                lmax=lmax,
                mmax=mmax,
                channels=channels,
                so2_layers=so2_layers,
                use_parallel=use_parallel,
                dtype=dtype,
                seed=123,
                trainable=True,
            )

            # === Step 3. Optimized path (reduced rotate/back inside SO2Convolution) ===
            out_opt = so2_conv(x, edge_cache, radial_feat)

            # === Step 4. Reference path: full rotate/back + zero-fill truncation ===
            x_src = x.index_select(0, src)

            if use_parallel:
                assert D_full_mat is not None
                assert Dt_full_mat is not None
                D_block = D_full_mat[:, :D_full, :D_full]
                Dt_block = Dt_full_mat[:, :D_full, :D_full]
                x_local_full = torch.bmm(D_block, x_src)
            else:
                x_local_full = x_src.new_empty(x_src.shape)
                for l in range(lmax + 1):
                    start, end = l * l, (l + 1) * (l + 1)
                    x_local_full[:, start:end, :] = torch.einsum(
                        "eij,ejc->eic", D_list[l], x_src[:, start:end, :]
                    )

            m_idx = build_m_major_index(lmax, mmax, device=self.device)
            x_local_red = x_local_full.index_select(1, m_idx)

            rad_feat_red = radial_feat[:, so2_conv.degree_index_m, :]
            x_local_red = x_local_red * rad_feat_red

            for layer_idx, (so2_linear, non_linear) in enumerate(
                zip(so2_conv.so2_linears, so2_conv.non_linearities)
            ):
                x_local_red = so2_linear(x_local_red)
                if layer_idx == 0:
                    bias_correction = so2_linear.bias0 * (
                        rad_feat_red[:, 0, :] * edge_env - 1.0
                    )
                    x_local_red[:, 0, :].add_(bias_correction)
                x_local_red = non_linear(x_local_red)

            x_local_full_zero = x_local_full.new_zeros(x_local_full.shape)
            x_local_full_zero.index_copy_(1, m_idx, x_local_red)

            if use_parallel:
                x_global_ref = torch.bmm(Dt_block, x_local_full_zero)
            else:
                x_global_ref = x_local_full_zero.new_empty(x_local_full_zero.shape)
                for l in range(lmax + 1):
                    start, end = l * l, (l + 1) * (l + 1)
                    x_global_ref[:, start:end, :] = torch.einsum(
                        "eij,ejc->eic", Dt_list[l], x_local_full_zero[:, start:end, :]
                    )

            x_global_ref = x_global_ref * edge_env.view(-1, 1, 1)
            out_ref = x.new_zeros(x.shape)
            out_ref.index_add_(0, dst, x_global_ref)
            out_ref = out_ref * inv_sqrt_deg
            out_ref = so2_conv.so3_linear(out_ref)

            torch.testing.assert_close(out_opt, out_ref, atol=atol, rtol=rtol)


class TestJITScript(unittest.TestCase):
    """Test torch.jit.script compatibility for SeZM-Net descriptor."""

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

    def _get_tols(self, dtype: torch.dtype) -> tuple[float, float]:
        if dtype == torch.float64:
            return 1e-10, 1e-10
        if dtype == torch.float32:
            return 5e-5, 5e-5
        return 5e-3, 5e-3

    def test_jit_script_basic(self) -> None:
        """Test that DescrptSeZMNet can be scripted with torch.jit.script."""
        for prec, use_parallel in itertools.product(
            ["float64", "float32", "bfloat16"], [False, True]
        ):
            dtype = PRECISION_DICT[prec]
            atol, rtol = self._get_tols(dtype)
            coord, atype, nlist = self._tiny_system(dtype=dtype)
            extended_coord = coord.reshape(1, -1)

            # Create and script the model
            model = DescrptSeZMNet(
                rcut=3.0,
                sel=[1, 1],
                ntypes=2,
                l_schedule=[1, 0],
                channels=4,
                n_radial=3,
                radial_mlp=[6],
                ffn_neurons=8,
                use_parallel=use_parallel,
                precision=prec,
                trainable=True,
            )

            # Get reference output before scripting
            model.eval()
            with torch.no_grad():
                desc_ref, _, _, _, sw_ref = model(
                    extended_coord, atype, nlist, mapping=None, comm_dict=None
                )

            # Script the model
            try:
                scripted_model = torch.jit.script(model)
            except Exception as e:
                self.fail(
                    f"torch.jit.script failed for {prec}, use_parallel={use_parallel}: {e}"
                )

            # Get output from scripted model
            with torch.no_grad():
                desc_script, _, _, _, sw_script = scripted_model(
                    extended_coord, atype, nlist, mapping=None, comm_dict=None
                )

            # Verify outputs match
            torch.testing.assert_close(
                desc_script,
                desc_ref,
                atol=atol,
                rtol=rtol,
                msg=f"Descriptor output mismatch after scripting (prec={prec}, parallel={use_parallel})",
            )
            torch.testing.assert_close(
                sw_script,
                sw_ref,
                atol=atol,
                rtol=rtol,
                msg=f"Smooth weight mismatch after scripting (prec={prec}, parallel={use_parallel})",
            )

    def test_jit_script_forward_backward(self) -> None:
        """Test that scripted model can execute forward and backward pass correctly."""
        dtype = torch.float64
        coord, atype, nlist = self._tiny_system(dtype=dtype)
        extended_coord = coord.reshape(1, -1).detach().requires_grad_(True)

        model = DescrptSeZMNet(
            rcut=3.0,
            sel=[1, 1],
            ntypes=2,
            l_schedule=[1, 1, 0],
            channels=8,
            n_radial=4,
            radial_mlp=[8],
            so2_layers=2,
            ffn_neurons=16,
            use_parallel=False,
            precision="float64",
            trainable=True,
        )

        # Script the model
        scripted_model = torch.jit.script(model)

        # Execute forward pass
        desc, rot_mat, g2, h2, sw = scripted_model(
            extended_coord, atype, nlist, mapping=None, comm_dict=None
        )

        # Verify output shapes and types
        self.assertEqual(desc.shape, (1, 2, 8))
        self.assertEqual(desc.dtype, env.GLOBAL_PT_FLOAT_PRECISION)
        self.assertIsNone(rot_mat)
        self.assertIsNone(g2)
        self.assertIsNone(h2)
        self.assertEqual(sw.shape, (1, 2, 2, 1))
        self.assertEqual(sw.dtype, env.GLOBAL_PT_FLOAT_PRECISION)

        # Verify backward pass
        loss = desc.sum()
        loss.backward()
        self.assertIsNotNone(extended_coord.grad)
        self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))


if __name__ == "__main__":
    unittest.main()
