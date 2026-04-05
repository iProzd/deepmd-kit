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

from deepmd.pt.model.descriptor.sezm_nn import (
    build_m_major_index,
    project_D_to_m,
    project_Dt_from_m,
)
from deepmd.pt.model.descriptor.sezm_nn.triton import (
    SEZM_TRITON_AVAILABLE,
    TritonRotationMode,
    resolve_triton_rotation_mode,
    rotate_back_triton,
    rotate_to_local_triton,
)

TRITON_CUDA_AVAILABLE = SEZM_TRITON_AVAILABLE and torch.cuda.is_available()


class TestSeZMTritonDispatch(unittest.TestCase):
    """Validate the SeZM Triton dispatch policy."""

    def test_resolve_rotation_mode_covers_small_generic_and_fallback(self) -> None:
        """Dispatch policy should cover small kernels, generic kernels, and fallback."""
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=1, reduced_dim=1),
            TritonRotationMode.SMALL_LE1,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=4, reduced_dim=4),
            TritonRotationMode.SMALL_LE1,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=9, reduced_dim=7),
            TritonRotationMode.SMALL_L2,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=16, reduced_dim=10),
            TritonRotationMode.SMALL_L3,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=25, reduced_dim=15),
            TritonRotationMode.EAGER_REFERENCE,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=25, reduced_dim=16),
            TritonRotationMode.GENERIC_TILED,
        )


@unittest.skipUnless(
    TRITON_CUDA_AVAILABLE,
    "SeZM Triton rotation tests require CUDA and Triton.",
)
class TestSeZMTritonSO2(unittest.TestCase):
    """Validate Triton SO(2) rotation kernels against the eager reference path."""

    def _require_cuda_bfloat16(self) -> None:
        """Skip the mixed-precision Triton tests when CUDA bf16 is unavailable."""
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA bfloat16 is required for mixed-precision Triton tests.")

    def test_rotate_to_local_matches_reference_forward_backward(self) -> None:
        """Compare fused Triton rotate-to-local with projected eager matmul."""
        device = torch.device("cuda")
        dtype = torch.float32
        n_node = 7
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
            x_ref = torch.randn(
                n_node,
                dim_full,
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_triton = x_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_D_to_m(
                    D_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_ref.index_select(0, src),
            )
            out_triton = rotate_to_local_triton(
                x=x_triton,
                src=src,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

    def test_rotate_back_matches_reference_forward_backward(self) -> None:
        """Compare fused Triton rotate-back with projected eager matmul."""
        device = torch.device("cuda")
        dtype = torch.float32
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            reduced_dim = int(coeff_index.numel())
            x_local_ref = torch.randn(
                n_edge,
                reduced_dim,
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_Dt_from_m(
                    Dt_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_local_ref,
            )
            out_triton = rotate_back_triton(
                x_local=x_local_triton,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_local_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_local_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

    def test_rotate_to_local_matches_mixed_precision_reference(self) -> None:
        """Compare Triton rotate-to-local with bf16 activations and fp32 Wigner."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        x_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_node = 7
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
            x_ref = torch.randn(
                n_node,
                dim_full,
                channels,
                device=device,
                dtype=x_dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=wigner_dtype,
                requires_grad=True,
            )
            x_triton = x_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_D_to_m(
                    D_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ).to(dtype=x_dtype),
                x_ref.index_select(0, src),
            )
            out_triton = rotate_to_local_triton(
                x=x_triton,
                src=src,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=3.0e-2, rtol=3.0e-2)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )

    def test_rotate_back_matches_mixed_precision_reference(self) -> None:
        """Compare Triton rotate-back with bf16 activations and fp32 Wigner."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        x_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            reduced_dim = int(coeff_index.numel())
            x_local_ref = torch.randn(
                n_edge,
                reduced_dim,
                channels,
                device=device,
                dtype=x_dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=wigner_dtype,
                requires_grad=True,
            )
            x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_Dt_from_m(
                    Dt_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ).to(dtype=x_dtype),
                x_local_ref,
            )
            out_triton = rotate_back_triton(
                x_local=x_local_triton,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=3.0e-2, rtol=3.0e-2)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_local_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_local_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )

    def test_rotate_to_local_matches_bfloat16_autocast_semantics(self) -> None:
        """Use the activation dtype selected by AMP for Triton rotate-to-local."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        act_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_node = 7
        n_edge = 11
        dim_full = 16
        channels = 8
        coeff_index = build_m_major_index(3, 1, device=device)
        src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
        x_ref = torch.randn(
            n_node,
            dim_full,
            channels,
            device=device,
            dtype=act_dtype,
            requires_grad=True,
        )
        wigner_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=wigner_dtype,
            requires_grad=True,
        )
        x_triton = x_ref.detach().clone().requires_grad_(True)
        wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

        D_m_prime = project_D_to_m(
            D_full=wigner_ref,
            coeff_index_m=coeff_index,
            ebed_dim_full=dim_full,
            cache=None,
            key_lmax=3,
            key_mmax=1,
        ).to(dtype=act_dtype)
        out_ref = torch.bmm(D_m_prime, x_ref.index_select(0, src))
        out_triton = rotate_to_local_triton(
            x=x_triton,
            src=src,
            wigner=wigner_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(out_triton, out_ref, atol=5.0e-2, rtol=5.0e-2)

        grad_out = torch.randn_like(out_ref)
        grad_x_ref, grad_wigner_ref = torch.autograd.grad(
            out_ref,
            (x_ref, wigner_ref),
            grad_outputs=grad_out,
        )
        grad_x_triton, grad_wigner_triton = torch.autograd.grad(
            out_triton,
            (x_triton, wigner_triton),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_x_triton,
            grad_x_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )
        torch.testing.assert_close(
            grad_wigner_triton,
            grad_wigner_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )

    def test_rotate_back_matches_bfloat16_autocast_semantics(self) -> None:
        """Use the activation dtype selected by AMP for Triton rotate-back."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        act_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_edge = 11
        dim_full = 16
        channels = 8
        coeff_index = build_m_major_index(3, 1, device=device)
        reduced_dim = int(coeff_index.numel())
        x_local_ref = torch.randn(
            n_edge,
            reduced_dim,
            channels,
            device=device,
            dtype=act_dtype,
            requires_grad=True,
        )
        wigner_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=wigner_dtype,
            requires_grad=True,
        )
        x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
        wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

        Dt_from_m = project_Dt_from_m(
            Dt_full=wigner_ref,
            coeff_index_m=coeff_index,
            ebed_dim_full=dim_full,
            cache=None,
            key_lmax=3,
            key_mmax=1,
        ).to(dtype=act_dtype)
        out_ref = torch.bmm(Dt_from_m, x_local_ref)
        out_triton = rotate_back_triton(
            x_local=x_local_triton,
            wigner=wigner_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(out_triton, out_ref, atol=5.0e-2, rtol=5.0e-2)

        grad_out = torch.randn_like(out_ref)
        grad_x_ref, grad_wigner_ref = torch.autograd.grad(
            out_ref,
            (x_local_ref, wigner_ref),
            grad_outputs=grad_out,
        )
        grad_x_triton, grad_wigner_triton = torch.autograd.grad(
            out_triton,
            (x_local_triton, wigner_triton),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_x_triton,
            grad_x_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )
        torch.testing.assert_close(
            grad_wigner_triton,
            grad_wigner_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )

    def test_generic_small_k_falls_back_to_reference_forward_backward(self) -> None:
        """Fallback to eager bmm when generic Triton tiles would have K < 16."""
        device = torch.device("cuda")
        dtype = torch.float32
        lmax, mmax = 4, 0
        dim_full = (lmax + 1) ** 2
        n_node = 7
        n_edge = 11
        channels = 8
        coeff_index = build_m_major_index(lmax, mmax, device=device)
        self.assertLess(int(coeff_index.numel()), 16)

        src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
        x_ref = torch.randn(
            n_node,
            dim_full,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        wigner_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        x_triton = x_ref.detach().clone().requires_grad_(True)
        wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

        out_ref = torch.bmm(
            project_D_to_m(
                D_full=wigner_ref,
                coeff_index_m=coeff_index,
                ebed_dim_full=dim_full,
                cache=None,
                key_lmax=lmax,
                key_mmax=mmax,
            ),
            x_ref.index_select(0, src),
        )
        out_triton = rotate_to_local_triton(
            x=x_triton,
            src=src,
            wigner=wigner_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

        grad_out = torch.randn_like(out_ref)
        grad_x_ref, grad_wigner_ref = torch.autograd.grad(
            out_ref,
            (x_ref, wigner_ref),
            grad_outputs=grad_out,
        )
        grad_x_triton, grad_wigner_triton = torch.autograd.grad(
            out_triton,
            (x_triton, wigner_triton),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_x_triton,
            grad_x_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            grad_wigner_triton,
            grad_wigner_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

        x_local_ref = torch.randn(
            n_edge,
            int(coeff_index.numel()),
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        wigner_back_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
        wigner_back_triton = wigner_back_ref.detach().clone().requires_grad_(True)

        out_back_ref = torch.bmm(
            project_Dt_from_m(
                Dt_full=wigner_back_ref,
                coeff_index_m=coeff_index,
                ebed_dim_full=dim_full,
                cache=None,
                key_lmax=lmax,
                key_mmax=mmax,
            ),
            x_local_ref,
        )
        out_back_triton = rotate_back_triton(
            x_local=x_local_triton,
            wigner=wigner_back_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(
            out_back_triton,
            out_back_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

        grad_back = torch.randn_like(out_back_ref)
        grad_x_local_ref, grad_wigner_back_ref = torch.autograd.grad(
            out_back_ref,
            (x_local_ref, wigner_back_ref),
            grad_outputs=grad_back,
        )
        grad_x_local_triton, grad_wigner_back_triton = torch.autograd.grad(
            out_back_triton,
            (x_local_triton, wigner_back_triton),
            grad_outputs=grad_back,
        )
        torch.testing.assert_close(
            grad_x_local_triton,
            grad_x_local_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            grad_wigner_back_triton,
            grad_wigner_back_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    def test_generic_large_k_matches_reference_forward_backward(self) -> None:
        """Exercise the true generic Triton path when reduced_dim >= 16."""
        device = torch.device("cuda")
        dtype = torch.float32
        n_node = 7
        n_edge = 11
        channels = 8
        for lmax, mmax in ((4, 2), (4, 4), (5, 2)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            self.assertGreaterEqual(int(coeff_index.numel()), 16)

            src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
            x_ref = torch.randn(
                n_node,
                dim_full,
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_triton = x_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_D_to_m(
                    D_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_ref.index_select(0, src),
            )
            out_triton = rotate_to_local_triton(
                x=x_triton,
                src=src,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

            x_local_ref = torch.randn(
                n_edge,
                int(coeff_index.numel()),
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_back_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
            wigner_back_triton = wigner_back_ref.detach().clone().requires_grad_(True)

            out_back_ref = torch.bmm(
                project_Dt_from_m(
                    Dt_full=wigner_back_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_local_ref,
            )
            out_back_triton = rotate_back_triton(
                x_local=x_local_triton,
                wigner=wigner_back_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(
                out_back_triton,
                out_back_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

            grad_back = torch.randn_like(out_back_ref)
            grad_x_local_ref, grad_wigner_back_ref = torch.autograd.grad(
                out_back_ref,
                (x_local_ref, wigner_back_ref),
                grad_outputs=grad_back,
            )
            grad_x_local_triton, grad_wigner_back_triton = torch.autograd.grad(
                out_back_triton,
                (x_local_triton, wigner_back_triton),
                grad_outputs=grad_back,
            )
            torch.testing.assert_close(
                grad_x_local_triton,
                grad_x_local_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_back_triton,
                grad_wigner_back_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
