"""Tests for versatil.post_training_compression.compressor module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.post_training_compression.pruning.structured import StructuredPruner
from versatil.post_training_compression.pruning.unstructured import UnstructuredPruner
from versatil.quantization.backends.x86_inductor import X86InductorBackend
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy

COMPRESSOR_MODULE = "versatil.post_training_compression.compressor"


@pytest.fixture
def policy_with_submodules() -> nn.Module:
    """Real small policy for validate() tests."""

    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
            self.decoder = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.backbone(x))

    return Policy()


@pytest.mark.unit
class TestPostTrainingCompressorValidate:
    @pytest.mark.parametrize(
        "module_path, expectation",
        [
            ("backbone", does_not_raise()),
            ("decoder", does_not_raise()),
            ("backbone.0", does_not_raise()),
            ("", does_not_raise()),
            (
                "nonexistent_module",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Module path 'nonexistent_module' not found in "
                        "policy. Available top-level modules: "
                        "['backbone', 'decoder']"
                    ),
                ),
            ),
        ],
    )
    def test_module_path_validation(
        self,
        policy_with_submodules,
        compressor_factory,
        module_path,
        expectation,
    ):
        compressor = compressor_factory(
            modules=[CompressionTarget(module_path=module_path)],
        )

        with expectation:
            compressor.validate(
                policy=policy_with_submodules,
                modules=compressor.resolve_modules(),
            )

    def test_raises_when_mixing_pt2e_and_quantize_api(
        self,
        policy_with_submodules,
        compressor_factory,
    ):
        compressor = compressor_factory(
            modules=[
                CompressionTarget(
                    module_path="backbone",
                    quantization=PT2EStrategy(pt2e_backend=X86InductorBackend()),
                ),
                CompressionTarget(
                    module_path="decoder",
                    quantization=QuantizeApiStrategy(
                        quantize_config=MagicMock(spec=[])
                    ),
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "PT2E and quantize_() strategies cannot be combined. "
                "PT2E operates on the exported FX graph while "
                "quantize_() requires eager nn.Module submodules. "
                "Use one strategy per compression run."
            ),
        ):
            compressor.validate(
                policy=policy_with_submodules,
                modules=compressor.resolve_modules(),
            )

    def test_global_mode_validation_uses_resolved_modules(
        self,
        policy_with_submodules,
        compressor_factory,
    ):
        compressor = compressor_factory(
            quantization=PT2EStrategy(pt2e_backend=X86InductorBackend()),
        )
        # Global mode: self.modules is empty, resolve_modules() creates root target
        resolved = compressor.resolve_modules()
        assert len(resolved) == 1
        assert resolved[0].module_path == ""
        # Validation should run on resolved modules, not the empty self.modules
        compressor.validate(policy=policy_with_submodules, modules=resolved)

    def test_multiple_modules_with_one_invalid_path(
        self,
        policy_with_submodules,
        compressor_factory,
    ):
        compressor = compressor_factory(
            modules=[
                CompressionTarget(module_path="backbone"),
                CompressionTarget(module_path="nonexistent"),
            ],
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Module path 'nonexistent' not found in policy. "
                "Available top-level modules: ['backbone', 'decoder']"
            ),
        ):
            compressor.validate(
                policy=policy_with_submodules,
                modules=compressor.resolve_modules(),
            )


@pytest.mark.unit
class TestResolveModules:
    def test_returns_explicit_modules_when_provided(self, compressor_factory):
        explicit = [CompressionTarget(module_path="backbone")]
        compressor = compressor_factory(modules=explicit)

        assert compressor.resolve_modules() is explicit

    @pytest.mark.parametrize("generate_report", [True, False])
    def test_falls_back_to_global_settings_when_empty(
        self, compressor_factory, generate_report
    ):
        preparation = PreparationConfig()
        pruning = [MagicMock(), MagicMock()]
        quantization = PT2EStrategy(pt2e_backend=X86InductorBackend())
        compressor = compressor_factory(
            preparation=preparation,
            pruning=pruning,
            quantization=quantization,
            generate_report=generate_report,
        )

        assert compressor.generate_report is generate_report

        resolved = compressor.resolve_modules()

        assert len(resolved) == 1
        assert resolved[0].module_path == ""
        assert resolved[0].preparation is preparation
        assert resolved[0].pruning is pruning
        assert resolved[0].quantization is quantization


@pytest.mark.unit
class TestResolveOutputDirectory:
    def test_returns_explicit_directory_when_set(self, compressor_factory):
        compressor = compressor_factory(output_directory="/custom/output")

        assert compressor._resolve_output_directory() == "/custom/output"

    def test_generates_timestamped_directory_when_not_set(self, compressor_factory):
        compressor = compressor_factory(checkpoint_path="/tmp/ckpt")

        fixed_time = datetime(2026, 3, 24, 12, 30, 45)
        with patch(f"{COMPRESSOR_MODULE}.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            mock_datetime.strftime = datetime.strftime
            result = compressor._resolve_output_directory()

        assert result == "/tmp/ckpt/compressed/20260324_123045"


@pytest.mark.unit
class TestPrepareAndPrune:
    def test_calls_bn_preparation_and_fusion_on_submodule(
        self,
        mock_policy_factory,
        compressor_factory,
    ):
        submodule = MagicMock(spec=nn.Module)
        policy = mock_policy_factory(submodule_paths={"backbone": submodule})
        target = CompressionTarget(
            module_path="backbone",
            preparation=PreparationConfig(
                replace_frozen_batchnorm=True,
                fuse_conv_batchnorm=True,
            ),
        )
        compressor = compressor_factory()

        with (
            patch(
                f"{COMPRESSOR_MODULE}.prepare_batchnorms_for_quantization",
                return_value=5,
            ) as mock_bn_prep,
            patch(
                f"{COMPRESSOR_MODULE}.fuse_all_conv_batchnorm_pairs",
                return_value=3,
            ) as mock_fuse,
        ):
            compressor._prepare_and_prune(policy=policy, modules=[target])

        mock_bn_prep.assert_called_once_with(submodule)
        mock_fuse.assert_called_once_with(submodule)

    def test_skips_fusion_when_disabled(
        self,
        mock_policy_factory,
        compressor_factory,
    ):
        policy = mock_policy_factory()
        target = CompressionTarget(
            module_path="",
            preparation=PreparationConfig(
                replace_frozen_batchnorm=True,
                fuse_conv_batchnorm=False,
            ),
        )
        compressor = compressor_factory()

        with (
            patch(
                f"{COMPRESSOR_MODULE}.prepare_batchnorms_for_quantization",
                return_value=0,
            ),
            patch(
                f"{COMPRESSOR_MODULE}.fuse_all_conv_batchnorm_pairs",
            ) as mock_fuse,
        ):
            compressor._prepare_and_prune(policy=policy, modules=[target])

        mock_fuse.assert_not_called()

    def test_calls_pruners_sequentially(
        self,
        mock_policy_factory,
        mock_pruner_factory,
        compressor_factory,
    ):
        policy = mock_policy_factory()
        pruner_a = mock_pruner_factory(total_parameters=100, zero_parameters=30)
        pruner_b = mock_pruner_factory(total_parameters=100, zero_parameters=50)
        target = CompressionTarget(module_path="", pruning=[pruner_a, pruner_b])
        compressor = compressor_factory()

        compressor._prepare_and_prune(policy=policy, modules=[target])

        pruner_a.prune.assert_called_once_with(module=policy)
        pruner_b.prune.assert_called_once_with(module=policy)

    def test_skips_preparation_when_none(
        self,
        mock_policy_factory,
        compressor_factory,
    ):
        policy = mock_policy_factory()
        target = CompressionTarget(module_path="", preparation=None)
        compressor = compressor_factory()

        with (
            patch(
                f"{COMPRESSOR_MODULE}.prepare_batchnorms_for_quantization"
            ) as mock_bn,
            patch(f"{COMPRESSOR_MODULE}.fuse_all_conv_batchnorm_pairs") as mock_fuse,
        ):
            compressor._prepare_and_prune(policy=policy, modules=[target])

        mock_bn.assert_not_called()
        mock_fuse.assert_not_called()

    def test_uses_root_policy_when_module_path_empty(
        self,
        mock_policy_factory,
        mock_pruner_factory,
        compressor_factory,
    ):
        policy = mock_policy_factory()
        pruner = mock_pruner_factory()
        target = CompressionTarget(module_path="", pruning=[pruner])
        compressor = compressor_factory()

        compressor._prepare_and_prune(policy=policy, modules=[target])

        pruner.prune.assert_called_once_with(module=policy)
        policy.get_submodule.assert_not_called()

    def test_resolves_submodule_for_non_empty_path(
        self,
        mock_policy_factory,
        mock_pruner_factory,
        compressor_factory,
    ):
        submodule = MagicMock(spec=nn.Module)
        policy = mock_policy_factory(submodule_paths={"encoder.backbone": submodule})
        pruner = mock_pruner_factory()
        target = CompressionTarget(module_path="encoder.backbone", pruning=[pruner])
        compressor = compressor_factory()

        compressor._prepare_and_prune(policy=policy, modules=[target])

        policy.get_submodule.assert_called_once_with("encoder.backbone")
        pruner.prune.assert_called_once_with(module=submodule)


@pytest.fixture
def export_quantize_mocks(
    mock_policy_factory,
) -> Callable[..., dict[str, MagicMock]]:
    """Factory for the common mock objects needed by _export_and_quantize tests."""

    def factory() -> dict[str, MagicMock]:
        return {
            "policy": mock_policy_factory(),
            "policy_loader": MagicMock(),
            "exportable": MagicMock(),
            "exported": MagicMock(spec=nn.Module),
            "converted": MagicMock(spec=nn.Module),
            "example_inputs": (torch.zeros(2, 4),),
        }

    return factory


@pytest.mark.unit
class TestExportAndQuantize:
    def test_pt2e_strategy_calls_pt2e_quantization(
        self,
        export_quantize_mocks,
        compressor_factory,
    ):
        mocks = export_quantize_mocks()
        target = CompressionTarget(
            module_path="",
            quantization=PT2EStrategy(
                pt2e_backend=X86InductorBackend(is_dynamic=True),
            ),
        )
        compressor = compressor_factory()

        with (
            patch(f"{COMPRESSOR_MODULE}.export_policy", return_value=mocks["exported"]),
            patch(
                f"{COMPRESSOR_MODULE}.apply_pt2e_quantization",
                return_value=mocks["converted"],
            ) as mock_pt2e,
            patch(
                f"{COMPRESSOR_MODULE}.build_example_inputs",
                return_value=mocks["example_inputs"],
            ),
        ):
            _, converted, _, strategy = compressor._export_and_quantize(
                policy=mocks["policy"],
                policy_loader=mocks["policy_loader"],
                exportable=mocks["exportable"],
                modules=[target],
            )

        mock_pt2e.assert_called_once()
        assert converted is mocks["converted"]
        assert strategy == "pt2e"

    def test_quantize_api_strategy_calls_quantize_api(
        self,
        export_quantize_mocks,
        compressor_factory,
    ):
        mocks = export_quantize_mocks()
        target = CompressionTarget(
            module_path="",
            quantization=QuantizeApiStrategy(quantize_config=MagicMock(spec=[])),
        )
        compressor = compressor_factory()

        with (
            patch(f"{COMPRESSOR_MODULE}.export_policy", return_value=mocks["exported"]),
            patch(f"{COMPRESSOR_MODULE}.apply_quantize_api") as mock_qapi,
            patch(
                f"{COMPRESSOR_MODULE}.build_example_inputs",
                return_value=mocks["example_inputs"],
            ),
        ):
            _, converted, _, strategy = compressor._export_and_quantize(
                policy=mocks["policy"],
                policy_loader=mocks["policy_loader"],
                exportable=mocks["exportable"],
                modules=[target],
            )

        mock_qapi.assert_called_once()
        assert converted is mocks["exported"]
        assert strategy == "quantize_api"

    def test_no_quantization_returns_exported_as_converted(
        self,
        export_quantize_mocks,
        compressor_factory,
    ):
        mocks = export_quantize_mocks()
        target = CompressionTarget(module_path="", quantization=None)
        compressor = compressor_factory()

        with (
            patch(f"{COMPRESSOR_MODULE}.export_policy", return_value=mocks["exported"]),
            patch(
                f"{COMPRESSOR_MODULE}.build_example_inputs",
                return_value=mocks["example_inputs"],
            ),
        ):
            _, converted, _, strategy = compressor._export_and_quantize(
                policy=mocks["policy"],
                policy_loader=mocks["policy_loader"],
                exportable=mocks["exportable"],
                modules=[target],
            )

        assert converted is mocks["exported"]
        assert strategy == "pt2e"

    def test_static_pt2e_creates_calibration_provider(
        self,
        export_quantize_mocks,
        compressor_factory,
    ):
        mocks = export_quantize_mocks()
        target = CompressionTarget(
            module_path="",
            quantization=PT2EStrategy(
                pt2e_backend=X86InductorBackend(is_dynamic=False),
            ),
        )
        compressor = compressor_factory(calibration_steps=16)

        mock_calibration = MagicMock()
        mock_calibration.get_single_batch.return_value = mocks["example_inputs"]

        with (
            patch(f"{COMPRESSOR_MODULE}.export_policy", return_value=mocks["exported"]),
            patch(
                f"{COMPRESSOR_MODULE}.apply_pt2e_quantization",
                return_value=mocks["converted"],
            ),
            patch(
                f"{COMPRESSOR_MODULE}.get_dataloaders",
                return_value=(MagicMock(), None, None, None, None),
            ),
            patch(
                f"{COMPRESSOR_MODULE}.CalibrationDataProvider",
                return_value=mock_calibration,
            ) as mock_calib_cls,
        ):
            compressor._export_and_quantize(
                policy=mocks["policy"],
                policy_loader=mocks["policy_loader"],
                exportable=mocks["exportable"],
                modules=[target],
            )

        mock_calib_cls.assert_called_once()
        assert mock_calib_cls.call_args[1]["num_calibration_steps"] == 16

    def test_uses_calibration_batch_when_available(
        self,
        export_quantize_mocks,
        compressor_factory,
    ):
        mocks = export_quantize_mocks()
        calibration_batch = (torch.ones(2, 4),)
        target = CompressionTarget(
            module_path="",
            quantization=PT2EStrategy(
                pt2e_backend=X86InductorBackend(is_dynamic=False),
            ),
        )
        compressor = compressor_factory()

        mock_calibration = MagicMock()
        mock_calibration.get_single_batch.return_value = calibration_batch

        with (
            patch(f"{COMPRESSOR_MODULE}.export_policy", return_value=mocks["exported"]),
            patch(
                f"{COMPRESSOR_MODULE}.apply_pt2e_quantization",
                return_value=mocks["converted"],
            ),
            patch(
                f"{COMPRESSOR_MODULE}.get_dataloaders",
                return_value=(MagicMock(), None, None, None, None),
            ),
            patch(
                f"{COMPRESSOR_MODULE}.CalibrationDataProvider",
                return_value=mock_calibration,
            ),
            patch(f"{COMPRESSOR_MODULE}.build_example_inputs") as mock_build,
        ):
            _, _, inputs, _ = compressor._export_and_quantize(
                policy=mocks["policy"],
                policy_loader=mocks["policy_loader"],
                exportable=mocks["exportable"],
                modules=[target],
            )

        # Should use calibration batch, not build_example_inputs
        mock_build.assert_not_called()
        assert inputs is calibration_batch


@pytest.mark.integration
class TestPrepareAndPruneIntegration:
    def test_bn_fusion_replaces_batchnorm_with_identity(self, compressor_factory):
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        model.eval()
        target = CompressionTarget(
            module_path="",
            preparation=PreparationConfig(
                replace_frozen_batchnorm=True,
                fuse_conv_batchnorm=True,
            ),
        )
        compressor = compressor_factory()

        compressor._prepare_and_prune(policy=model, modules=[target])

        assert isinstance(model[1], nn.Identity)

    def test_sequential_pruning_increases_sparsity(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        compressor_factory,
    ):
        model = pruning_model_factory()
        target = CompressionTarget(
            module_path="",
            pruning=[
                StructuredPruner(amount=0.3),
                UnstructuredPruner(amount=0.3),
            ],
        )
        compressor = compressor_factory()

        compressor._prepare_and_prune(policy=model, modules=[target])

        total, zeroed = BasePruner.compute_sparsity(model)
        assert zeroed > 0

    def test_model_produces_finite_output_after_prepare_and_prune(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        compressor_factory,
        rng: np.random.Generator,
    ):
        model = pruning_model_factory()
        input_data = torch.from_numpy(
            rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
        )
        target = CompressionTarget(
            module_path="",
            pruning=[
                StructuredPruner(amount=0.3),
                UnstructuredPruner(amount=0.5),
            ],
        )
        compressor = compressor_factory()

        compressor._prepare_and_prune(policy=model, modules=[target])

        with torch.no_grad():
            output = model(input_data)
        assert output.shape == (2, 4)
        assert output.isfinite().all()
