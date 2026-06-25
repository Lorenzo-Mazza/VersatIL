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
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentArtifact,
)
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.post_training_compression.pruning.structured import StructuredPruner
from versatil.post_training_compression.pruning.unstructured import UnstructuredPruner
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.none import NoQuantizationWorkflow

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

    def test_global_mode_validation_uses_resolved_modules(
        self,
        policy_with_submodules,
        compressor_factory,
    ):
        compressor = compressor_factory()
        resolved = compressor.resolve_modules()

        assert len(resolved) == 1
        assert resolved[0].module_path == ""
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
        compressor = compressor_factory(
            preparation=preparation,
            pruning=pruning,
            generate_report=generate_report,
        )

        assert compressor.generate_report is generate_report

        resolved = compressor.resolve_modules()

        assert len(resolved) == 1
        assert resolved[0].module_path == ""
        assert resolved[0].preparation is preparation
        assert resolved[0].pruning is pruning


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


@pytest.mark.unit
class TestQuantizationSelection:
    def test_returns_configured_quantization(self, compressor_factory):
        quantization = MagicMock(spec=BaseQuantizationWorkflow)
        compressor = compressor_factory(quantization=quantization)

        result = compressor._resolve_quantization_workflow()

        assert result is quantization

    def test_returns_no_quantization_workflow_when_no_workflow_is_configured(
        self,
        compressor_factory,
    ):
        compressor = compressor_factory()

        result = compressor._resolve_quantization_workflow()

        assert isinstance(result, NoQuantizationWorkflow)
        assert result.quantization_mode == QuantizationMode.NONE.value

    def test_unknown_backend_is_rejected(self, compressor_factory):
        compressor = compressor_factory()

        with pytest.raises(
            ValueError,
            match=re.escape("Unknown deployment backend 'unknown_backend'."),
        ):
            compressor._validate_deployment_backend_compatibility(
                deployment_backend_name="unknown_backend",
                mode=QuantizationMode.EAGER.value,
            )


@pytest.mark.unit
class TestCompressOrchestration:
    def test_validates_quantization_targets_before_workflow_export(
        self,
        compressor_factory,
    ):
        quantization = MagicMock(spec=BaseQuantizationWorkflow)
        quantization.quantization_mode = QuantizationMode.EAGER.value
        context = MagicMock()
        policy = nn.Module()
        policy.input_keys = ["observation"]
        policy.output_keys = ["action"]
        policy.normalizer = MagicMock()
        context.policy = policy
        quantization.load_policy_context.return_value = context
        quantized = MagicMock()
        quantized.float_model = MagicMock(spec=nn.Module)
        quantized.quantized_model = MagicMock(spec=nn.Module)
        quantized.example_inputs = (MagicMock(),)
        quantized.quantization_workflow = QuantizationMode.EAGER.value
        quantization.quantize.return_value = quantized
        backend = MagicMock()
        backend.name = DeploymentBackendName.TORCH_INDUCTOR.value
        backend.export.return_value = DeploymentArtifact(
            converted_model=quantized.quantized_model,
            example_inputs=quantized.example_inputs,
            model_filename="compressed_policy.pt2",
            artifact_format=ArtifactFormat.TORCH_EXPORT_PT2,
            backend_name=DeploymentBackendName.TORCH_INDUCTOR.value,
        )
        exportable = MagicMock()
        exportable.observation_keys = ["observation"]
        exportable.action_keys = ["action"]
        hydra_config = MagicMock()
        compressor = compressor_factory(
            quantization=quantization,
            deployment_backend=backend,
            output_directory="/tmp/compressed",
        )

        with (
            patch(
                f"{COMPRESSOR_MODULE}.ExportablePolicy.from_policy",
                return_value=exportable,
            ) as mock_exportable_factory,
            patch(f"{COMPRESSOR_MODULE}.save_compressed_model") as mock_save,
        ):
            result = compressor.compress(hydra_config=hydra_config)

        quantization.load_policy_context.assert_called_once_with(
            checkpoint_path=compressor.checkpoint_path,
            checkpoint_name=compressor.checkpoint_name,
        )
        quantization.validate_targets.assert_called_once_with(model=policy)
        mock_exportable_factory.assert_called_once_with(policy)
        quantization.quantize.assert_called_once_with(
            context=context,
            exportable=exportable,
            calibration_steps=compressor.calibration_steps,
        )
        backend.export.assert_called_once_with(
            model=quantized.quantized_model,
            example_inputs=quantized.example_inputs,
        )
        mock_save.assert_called_once_with(
            converted_model=quantized.quantized_model,
            example_inputs=quantized.example_inputs,
            save_directory="/tmp/compressed",
            input_keys=policy.input_keys,
            output_keys=policy.output_keys,
            normalizer=policy.normalizer,
            training_checkpoint_path=compressor.checkpoint_path,
            quantization_config=hydra_config,
            quantization_workflow=quantized.quantization_workflow,
            model_filename="compressed_policy.pt2",
            artifact_format=ArtifactFormat.TORCH_EXPORT_PT2.value,
            backend_name=DeploymentBackendName.TORCH_INDUCTOR.value,
            model_bytes=None,
        )
        assert result == "/tmp/compressed"


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
        before_total, before_zeroed = BasePruner.compute_sparsity(model)
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
        assert total == before_total
        assert zeroed > before_zeroed

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
