# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZM's AOTInductor ``.pt2`` freeze pipeline.

Layout mirrors ``source/tests/pt_expt/model/test_export_pipeline.py``:
a tiny fp64 SeZM model is built on the fly, so the tests are fully
self-contained and have no external-artefact dependency.
"""

from __future__ import (
    annotations,
)

import contextlib
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import torch

from deepmd.pt.entrypoints.freeze_pt2 import (
    _build_dynamic_shapes,
    _resolve_nframes,
    freeze_sezm_to_pt2,
    is_sezm_checkpoint,
)
from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

# Tracing and numerical parity always run on CPU — see module docstring
# of deepmd/pt/entrypoints/freeze_pt2.py for why.
_CPU = torch.device("cpu")

_REQUIRED_OUTPUT_KEYS = {
    "energy",
    "energy_redu",
    "energy_derv_r",
    "energy_derv_c",
    "energy_derv_c_redu",
}


def _tiny_sezm_model_params() -> dict:
    """Minimal fp64 SeZM config for self-contained export tests.

    ``precision="float64"`` is what unlocks the ``rtol=1e-10, atol=1e-10``
    parity pt_expt enforces; fp32 accumulation alone drifts in the 1e-6
    range.  All other knobs are tuned to keep ``make_fx`` tracing time
    in the low-single-digit seconds.
    """
    return {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "SeZM",
            "sel": [2, 2],
            "rcut": 3.0,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": True,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 1,
            "sandwich_norm": [True, False, True, False],
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "s2_activation": [False, True],
            "mlp_bias": False,
            "layer_scale": False,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float64",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float64",
            "seed": 7,
        },
        "use_compile": False,
    }


def _build_tiny_sezm_model() -> torch.nn.Module:
    """Fresh tiny SeZM model on CPU, in eval mode."""
    model = get_sezm_model(_tiny_sezm_model_params())
    model.eval()
    model.to(_CPU)
    return model


def _write_tiny_sezm_checkpoint(tmp_path: Path, params: dict) -> Path:
    """Serialise a tiny SeZM model to a ``.pt`` in the trainer's layout.

    ``ModelWrapper`` populates ``state_dict["_extra_state"]`` from its
    ``get_extra_state`` hook, which is exactly the shape
    :func:`freeze_sezm_to_pt2` expects.
    """
    model = get_sezm_model(params)
    model.eval()
    model.to(_CPU)
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(params))
    ckpt_path = tmp_path / "tiny_sezm.pt"
    torch.save({"model": wrapper.state_dict()}, ckpt_path)
    return ckpt_path


def _make_sample(model: torch.nn.Module, *, nloc: int, start: int) -> tuple:
    """Build a forward_common_lower sample on CPU via the freeze helper."""
    _, sample = _resolve_nframes(model, nloc=nloc, device=_CPU, start=start)
    return sample


@contextlib.contextmanager
def _clear_default_device() -> Iterator[None]:
    """Clear the pt-test ``cuda:9999999`` sentinel default device.

    ``source/tests/pt/__init__.py`` sets the default device to an
    invalid ``"cuda:9999999"`` so that tests relying on implicit
    placement fail loudly.  The AOTI / export pipeline in PyTorch 2.11
    allocates unnamed tensors (e.g. inside ``PhiloxStateTracker``)
    without an explicit device and would trip the guard.  Matches the
    pattern used by ``pt_expt/test_change_bias.py``.
    """
    saved = torch.get_default_device()
    torch.set_default_device(None)
    try:
        yield
    finally:
        torch.set_default_device(saved)


def _eager_forward(
    model: torch.nn.Module,
    sample_inputs: tuple,
) -> dict[str, torch.Tensor]:
    """Mirror the trace closure: fresh leaf coord + ``requires_grad=True``."""
    ext_coord, ext_atype, nlist, mapping, fparam, aparam = sample_inputs
    eager_coord = ext_coord.detach().clone().requires_grad_(True)
    return model.forward_common_lower(
        eager_coord,
        ext_atype,
        nlist,
        mapping=mapping,
        fparam=fparam,
        aparam=aparam,
        do_atomic_virial=True,
        extra_nlist_sort=model.need_sorted_nlist_for_lower(),
    )


class TestSeZMExportPipeline(unittest.TestCase):
    """Bitwise trace / export / ``.pte`` round-trip parity (``rtol=1e-10``).

    The ExportedProgram is a pure FX graph (no Inductor codegen), so
    it must reproduce the eager result exactly.  Drift here implies a
    bug in ``forward_common_lower_exportable`` or the dynamic-shape
    spec, not in AOTI.  The pipeline is built once per class because
    ``make_fx`` and ``.pte`` round-trip dominate wall time.
    """

    @classmethod
    def setUpClass(cls) -> None:
        with _clear_default_device():
            cls.model = _build_tiny_sezm_model()
            cls.sample_inputs = _make_sample(cls.model, nloc=7, start=2)
            cls.traced, cls.loaded, cls._pte_tmp = cls._build_pipeline(
                cls.model, cls.sample_inputs
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._pte_tmp.close()

    def setUp(self) -> None:
        self._device_ctx = _clear_default_device()
        self._device_ctx.__enter__()

    def tearDown(self) -> None:
        self._device_ctx.__exit__(None, None, None)

    @staticmethod
    def _build_pipeline(
        model: torch.nn.Module,
        sample_inputs: tuple,
    ) -> tuple[
        torch.fx.GraphModule,
        torch.nn.Module,
        tempfile._TemporaryFileWrapper,
    ]:
        traced = model.forward_common_lower_exportable(
            *sample_inputs,
            do_atomic_virial=True,
        )
        exported = torch.export.export(
            traced,
            sample_inputs,
            dynamic_shapes=_build_dynamic_shapes(sample_inputs),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        # Keep the tempfile alive for the class lifetime so the loaded
        # module can lazily reference its backing bytes.
        pte_tmp = tempfile.NamedTemporaryFile(suffix=".pte", delete=True)
        torch.export.save(exported, pte_tmp.name)
        loaded = torch.export.load(pte_tmp.name).module()
        return traced, loaded, pte_tmp

    def _assert_dict_allclose(
        self,
        ref: dict[str, torch.Tensor],
        test_dict: dict[str, torch.Tensor] | object,
        *,
        context: str,
    ) -> None:
        test_pairs = (
            list(test_dict.items())
            if hasattr(test_dict, "items")
            else list(zip(ref.keys(), test_dict, strict=True))
        )
        for key, test_val in test_pairs:
            self.assertIn(key, ref, msg=f"{context}: unexpected output key {key!r}")
            ref_val = ref[key]
            self.assertEqual(
                tuple(ref_val.shape),
                tuple(test_val.shape),
                msg=(
                    f"{context} ({key}): shape mismatch "
                    f"ref={tuple(ref_val.shape)} vs test={tuple(test_val.shape)}"
                ),
            )
            np.testing.assert_allclose(
                ref_val.detach().cpu().numpy(),
                test_val.detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"{context}: {key}",
            )

    def test_traced_matches_eager_same_shape(self) -> None:
        eager_out = _eager_forward(self.model, self.sample_inputs)
        traced_out = self.traced(*self.sample_inputs)
        self._assert_dict_allclose(
            eager_out, traced_out, context="traced vs eager (trace shape)"
        )

    def test_loaded_pte_matches_eager_same_shape(self) -> None:
        eager_out = _eager_forward(self.model, self.sample_inputs)
        loaded_out = self.loaded(*self.sample_inputs)
        self._assert_dict_allclose(
            eager_out, loaded_out, context="loaded (.pte) vs eager (trace shape)"
        )

    def test_loaded_pte_matches_eager_different_shape(self) -> None:
        # start=3 retargets the nframes symbol away from the trace
        # value of 2; nloc=11 exercises the nloc symbol.
        infer_inputs = _make_sample(self.model, nloc=11, start=3)
        eager_out = _eager_forward(self.model, infer_inputs)
        loaded_out = self.loaded(*infer_inputs)
        self._assert_dict_allclose(
            eager_out, loaded_out, context="loaded (.pte) vs eager (infer shape)"
        )


class TestSeZMExportArchive(unittest.TestCase):
    """AOTI ``.pt2`` archive structure + load-and-run smoke.

    Numerical parity of the compiled ``.pt2`` is covered by the
    pipeline class through the ``.pte`` round-trip; here we only
    verify the archive layout and the C++ consumer contract.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_root = Path(cls._tmpdir.name)
        cls.params = _tiny_sezm_model_params()
        with _clear_default_device():
            cls.ckpt_path = _write_tiny_sezm_checkpoint(tmp_root, cls.params)
            cls.out_path = tmp_root / "frozen_sezm.pt2"
            freeze_sezm_to_pt2(str(cls.ckpt_path), str(cls.out_path), device=_CPU)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self._device_ctx = _clear_default_device()
        self._device_ctx.__enter__()

    def tearDown(self) -> None:
        self._device_ctx.__exit__(None, None, None)

    def test_detector_recognises_sezm(self) -> None:
        self.assertTrue(is_sezm_checkpoint(str(self.ckpt_path)))

    def test_archive_metadata(self) -> None:
        """ZIP layout + metadata fields match the DeepPotPTExpt contract."""
        self.assertTrue(zipfile.is_zipfile(str(self.out_path)))
        with zipfile.ZipFile(str(self.out_path), "r") as zf:
            names = zf.namelist()
            self.assertIn("extra/metadata.json", names)
            self.assertIn("extra/model_def_script.json", names)
            metadata = json.loads(zf.read("extra/metadata.json").decode("utf-8"))
            mds = json.loads(zf.read("extra/model_def_script.json").decode("utf-8"))

        for key in (
            "type_map",
            "rcut",
            "sel",
            "dim_fparam",
            "dim_aparam",
            "mixed_types",
            "has_default_fparam",
            "output_keys",
            "fitting_output_defs",
            "is_spin",
        ):
            self.assertIn(key, metadata)

        self.assertEqual(metadata["type_map"], self.params["type_map"])
        self.assertEqual(metadata["rcut"], self.params["descriptor"]["rcut"])
        self.assertEqual(list(metadata["sel"]), list(self.params["descriptor"]["sel"]))
        self.assertTrue(metadata["mixed_types"])
        self.assertFalse(metadata["is_spin"])
        self.assertEqual(metadata["dim_fparam"], 0)
        self.assertEqual(metadata["dim_aparam"], 0)
        self.assertTrue(_REQUIRED_OUTPUT_KEYS.issubset(set(metadata["output_keys"])))

        # model_def_script preserves the training params verbatim.
        self.assertEqual(str(mds.get("type", "")).lower(), "sezm")
        self.assertEqual(mds.get("use_compile"), self.params["use_compile"])

    def test_aoti_load_and_run_returns_finite_outputs(self) -> None:
        from torch._inductor import (
            aoti_load_package,
        )

        loader = aoti_load_package(str(self.out_path))
        probe = _build_tiny_sezm_model()
        sample_inputs = _make_sample(probe, nloc=5, start=2)
        outs = loader(*sample_inputs)

        # AOTICompiledModel returns an immutable_dict on PyTorch ≥2.11
        # and a flat tuple on older versions; normalise both.
        with zipfile.ZipFile(str(self.out_path), "r") as zf:
            output_keys = json.loads(zf.read("extra/metadata.json").decode("utf-8"))[
                "output_keys"
            ]
        if hasattr(outs, "items"):
            out_map = dict(outs.items())
            self.assertEqual(list(out_map.keys()), output_keys)
        else:
            self.assertEqual(len(outs), len(output_keys))
            out_map = dict(zip(output_keys, outs, strict=True))

        for key in ("energy_redu", "energy_derv_r", "energy_derv_c_redu"):
            self.assertIn(key, out_map)
            self.assertTrue(torch.isfinite(out_map[key]).all().item())


class TestSeZMFreezeGuards(unittest.TestCase):
    """Error paths: detector rejections and CLI-level ``NotImplementedError``s."""

    def test_is_sezm_checkpoint_rejects_non_sezm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ener.pt"
            torch.save(
                {"model": {"_extra_state": {"model_params": {"type": "ener"}}}},
                ckpt_path,
            )
            self.assertFalse(is_sezm_checkpoint(str(ckpt_path)))

    def test_freeze_rejects_head_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "fake.pt"
            torch.save(
                {"model": {"_extra_state": {"model_params": {"type": "SeZM"}}}},
                ckpt_path,
            )
            out = Path(tmp) / "out.pt2"
            with self.assertRaises(NotImplementedError):
                freeze_sezm_to_pt2(str(ckpt_path), str(out), head="branch")

    def test_freeze_rejects_multi_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "multi.pt"
            torch.save(
                {
                    "model": {
                        "_extra_state": {
                            "model_params": {
                                "type": "SeZM",
                                "model_dict": {"branch": {}},
                            }
                        }
                    }
                },
                ckpt_path,
            )
            out = Path(tmp) / "out.pt2"
            with self.assertRaises(NotImplementedError):
                freeze_sezm_to_pt2(str(ckpt_path), str(out))


if __name__ == "__main__":
    unittest.main()
