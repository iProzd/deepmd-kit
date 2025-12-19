# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

# NOTE: avoid torch thread reconfiguration errors during import.
import torch

torch_set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
torch_set_num_threads = getattr(torch, "set_num_threads", None)
if torch_set_num_interop_threads is not None:
    torch.set_num_interop_threads = lambda *args, **kwargs: None  # type: ignore[assignment]
if torch_set_num_threads is not None:
    torch.set_num_threads = lambda *args, **kwargs: None  # type: ignore[assignment]

from deepmd.pt.model.descriptor.se_zm_net import (
    DescrptSeZMNet,
)
from deepmd.pt.model.descriptor.wigner_d import (
    WignerDCalc,
    WignerDCalcParallel,
)
from deepmd.pt.utils import (
    env,
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
        self.dtype = env.GLOBAL_PT_FLOAT_PRECISION

    def _tiny_system(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a minimal two-atom system for testing."""
        coord = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=self.dtype,
            device=self.device,
        ).view(1, -1, 3)
        atype = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        nlist = torch.tensor(
            [[[1, -1], [0, -1]]], dtype=torch.int64, device=self.device
        )
        return coord, atype, nlist

    def test_forward_shape_and_dtype(self) -> None:
        """Test that forward produces correct shape and dtype."""
        coord, atype, nlist = self._tiny_system()
        extended_coord = coord.reshape(1, -1)
        for wigner_parallel in [False, True]:
            model = DescrptSeZMNet(
                rcut=3.0,
                rcut_smth=2.5,
                sel=[1, 1],
                l_schedule=[1, 0],
                channels=8,
                n_radial=4,
                radial_mlp=[8],
                ffn_neuron=[16],
                neighbor_norm=False,
                wigner_parallel=wigner_parallel,
                trainable=True,
            )
            if wigner_parallel:
                self.assertIsInstance(model.wigner_calc, WignerDCalcParallel)
            else:
                self.assertIsInstance(model.wigner_calc, WignerDCalc)

            desc, _, _, _, sw = model(
                extended_coord, atype, nlist, mapping=None, comm_dict=None
            )
            self.assertEqual(desc.shape, (1, 2, 8))
            self.assertEqual(desc.dtype, self.dtype)
            self.assertEqual(sw.shape, (1, 2, 2, 1))

    def test_backward_gradient(self) -> None:
        """Test backward gradient through coordinates."""
        coord, atype, nlist = self._tiny_system()
        for wigner_parallel in [False, True]:
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
                neighbor_norm=False,
                wigner_parallel=wigner_parallel,
                trainable=True,
            )
            desc, *_ = model(extended_coord, atype, nlist, mapping=None, comm_dict=None)
            loss = desc.sum()
            loss.backward()
            self.assertIsNotNone(extended_coord.grad)
            self.assertTrue(torch.all(torch.isfinite(extended_coord.grad)))


class TestWignerDCalc(unittest.TestCase):
    """Test the Wigner-D matrix calculator."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        self.dtype = env.GLOBAL_PT_FLOAT_PRECISION
        self.atol = 5e-5 if self.dtype == torch.float32 else 1e-10
        self.rtol = 5e-5 if self.dtype == torch.float32 else 1e-10
        self.batch = 8
        torch.manual_seed(0)

    def test_orthogonality(self) -> None:
        """Test D @ D^T = I for random rotations."""
        for wigner_cls in [WignerDCalc, WignerDCalcParallel]:
            for lmax in [1, 2, 3]:
                wigner = wigner_cls(lmax=lmax, dtype=self.dtype)
                alpha = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 2
                    * 3.14159
                )
                beta = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 3.14159
                )
                gamma = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 2
                    * 3.14159
                )
                rot = _zyz_euler_to_matrix(alpha, beta, gamma)
                D_list, Dt_list = wigner(rot)

                for l in range(lmax + 1):
                    dim = 2 * l + 1
                    eye = torch.eye(dim, device=self.device, dtype=self.dtype)
                    eye = eye.expand(self.batch, dim, dim)
                    prod = D_list[l] @ Dt_list[l]
                    torch.testing.assert_close(
                        prod,
                        eye,
                        atol=self.atol,
                        rtol=self.rtol,
                        msg=(
                            f"Orthogonality failed for {wigner_cls.__name__}, lmax={lmax}, l={l}"
                        ),
                    )

    def test_group_property(self) -> None:
        """Test D(R1 @ R2) = D(R1) @ D(R2)."""
        for wigner_cls in [WignerDCalc, WignerDCalcParallel]:
            for lmax in [1, 2, 3]:
                wigner = wigner_cls(lmax=lmax, dtype=self.dtype)

                # Avoid gimbal lock by keeping beta away from 0 and pi
                alpha1 = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 2
                    * 3.14159
                )
                beta1 = 0.1 + torch.rand(
                    self.batch, device=self.device, dtype=self.dtype
                ) * (3.14159 - 0.2)
                gamma1 = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 2
                    * 3.14159
                )

                alpha2 = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 2
                    * 3.14159
                )
                beta2 = 0.1 + torch.rand(
                    self.batch, device=self.device, dtype=self.dtype
                ) * (3.14159 - 0.2)
                gamma2 = (
                    torch.rand(self.batch, device=self.device, dtype=self.dtype)
                    * 2
                    * 3.14159
                )

                rot1 = _zyz_euler_to_matrix(alpha1, beta1, gamma1)
                rot2 = _zyz_euler_to_matrix(alpha2, beta2, gamma2)
                rot12 = rot1 @ rot2

                D1, _ = wigner(rot1)
                D2, _ = wigner(rot2)
                D12, _ = wigner(rot12)

                for l in range(lmax + 1):
                    left = D12[l]
                    right = D1[l] @ D2[l]
                    torch.testing.assert_close(
                        left,
                        right,
                        atol=2e-10,
                        rtol=2e-10,
                        msg=(
                            f"Group property failed for {wigner_cls.__name__}, lmax={lmax}, l={l}"
                        ),
                    )

    def test_parallel_matches_per_l(self) -> None:
        """Test that the parallel implementation matches the per-l implementation."""
        for lmax in [1, 2, 3]:
            wigner_ref = WignerDCalc(lmax=lmax, dtype=self.dtype)
            wigner_par = WignerDCalcParallel(lmax=lmax, dtype=self.dtype)

            alpha = (
                torch.rand(self.batch, device=self.device, dtype=self.dtype)
                * 2
                * 3.14159
            )
            beta = 0.1 + torch.rand(
                self.batch, device=self.device, dtype=self.dtype
            ) * (3.14159 - 0.2)
            gamma = (
                torch.rand(self.batch, device=self.device, dtype=self.dtype)
                * 2
                * 3.14159
            )
            rot = _zyz_euler_to_matrix(alpha, beta, gamma)

            D_ref, Dt_ref = wigner_ref(rot)
            D_par, Dt_par = wigner_par(rot)

            for l in range(lmax + 1):
                torch.testing.assert_close(
                    D_par[l],
                    D_ref[l],
                    atol=self.atol,
                    rtol=self.rtol,
                    msg=f"D mismatch for lmax={lmax}, l={l}",
                )
                torch.testing.assert_close(
                    Dt_par[l],
                    Dt_ref[l],
                    atol=self.atol,
                    rtol=self.rtol,
                    msg=f"Dt mismatch for lmax={lmax}, l={l}",
                )

    def test_z_rotation_m0_invariance(self) -> None:
        """Test that m=0 component is invariant under z-rotation."""
        lmax = 3
        wigner = WignerDCalc(lmax=lmax, dtype=self.dtype)
        n_edge = 5
        angles = torch.rand(n_edge, device=self.device, dtype=self.dtype) * 2 * 3.14159

        # Test each l separately
        for l in range(lmax + 1):
            Z = wigner._build_z_rotation(angles, l)
            # Shape: (n_edge, 2l+1, 2l+1)
            dim = 2 * l + 1
            self.assertEqual(Z.shape, (n_edge, dim, dim))

            # m=0 index within the block is at position l (center)
            m0_idx = l

            # Check diagonal element is 1
            diag_vals = Z[:, m0_idx, m0_idx]
            torch.testing.assert_close(
                diag_vals,
                torch.ones(n_edge, device=self.device, dtype=self.dtype),
                atol=1e-10,
                rtol=1e-10,
                msg=f"m=0 diagonal should be 1 for l={l}",
            )

            # Check off-diagonal elements in m=0 row/col are 0
            for j in range(dim):
                if j != m0_idx:
                    row_vals = Z[:, m0_idx, j]
                    col_vals = Z[:, j, m0_idx]
                    torch.testing.assert_close(
                        row_vals,
                        torch.zeros(n_edge, device=self.device, dtype=self.dtype),
                        atol=1e-10,
                        rtol=1e-10,
                        msg=f"m=0 row off-diagonal should be 0 for l={l}, j={j}",
                    )
                    torch.testing.assert_close(
                        col_vals,
                        torch.zeros(n_edge, device=self.device, dtype=self.dtype),
                        atol=1e-10,
                        rtol=1e-10,
                        msg=f"m=0 col off-diagonal should be 0 for l={l}, j={j}",
                    )


if __name__ == "__main__":
    unittest.main()
