"""Tests for versatil.inference.policy_runtime.compressed_runtime module."""

import json
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import get_test_device
from versatil.checkpoint_loading.compressed_policy import CompressedCheckpointLoader
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.inference.policy_runtime.compressed_runtime import CompressedPolicyRuntime
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    CompressionMetadataKey,
    DeploymentBackendName,
    QuantizationWorkflow,
)
from versatil.quantization.pt2e.backends.base import BasePT2EBackend

COMPRESSED_RUNTIME_MODULE = "versatil.inference.policy_runtime.compressed_runtime"
COMPRESSED_CHECKPOINT_MODULE = "versatil.checkpoint_loading.compressed_policy"


@pytest.fixture
def metadata_factory() -> Callable[..., dict]:
    """Factory for compression metadata dicts."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
        model_filename: str = "compressed_policy.pt2",
        normalizer_filename: str = "normalizer.pt",
        training_checkpoint_path: str = "/tmp/training_checkpoint",
        include_training_path: bool = True,
        artifact_format: str = ArtifactFormat.TORCH_EXPORT_PT2.value,
        backend_name: str = DeploymentBackendName.TORCH_INDUCTOR.value,
        quantization_workflow: str | None = None,
        exclude_keys: list[str] | None = None,
    ) -> dict:
        if input_keys is None:
            input_keys = ["depth", "left"]
        if output_keys is None:
            output_keys = ["orientation", "position"]

        metadata = {
            CompressionMetadataKey.MODEL_FILE.value: model_filename,
            CompressionMetadataKey.NORMALIZER_FILE.value: normalizer_filename,
            CompressionMetadataKey.ARTIFACT_FORMAT.value: artifact_format,
            CompressionMetadataKey.DEPLOYMENT_BACKEND.value: backend_name,
            CompressionMetadataKey.INPUT_KEYS.value: input_keys,
            CompressionMetadataKey.OUTPUT_KEYS.value: output_keys,
            CompressionMetadataKey.TORCHAO_VERSION.value: "0.16.0",
            CompressionMetadataKey.TORCH_VERSION.value: "2.10.0",
        }
        if quantization_workflow is not None:
            metadata[CompressionMetadataKey.QUANTIZATION_WORKFLOW.value] = (
                quantization_workflow
            )
        if include_training_path:
            metadata[CompressionMetadataKey.TRAINING_CHECKPOINT_PATH.value] = (
                training_checkpoint_path
            )
        if exclude_keys:
            for key in exclude_keys:
                metadata.pop(key, None)
        return metadata

    return factory


@pytest.fixture
def checkpoint_directory_factory(
    tmp_path: Path,
    rng: np.random.Generator,
    metadata_factory: Callable[..., dict],
) -> Callable[..., str]:
    """Factory that creates a fake compressed checkpoint directory."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
        create_metadata: bool = True,
        create_model: bool = True,
        create_normalizer: bool = True,
        include_training_path: bool = True,
        model_filename: str = "compressed_policy.pt2",
        artifact_format: str = ArtifactFormat.TORCH_EXPORT_PT2.value,
        backend_name: str = DeploymentBackendName.TORCH_INDUCTOR.value,
        quantization_workflow: str | None = None,
        exclude_metadata_keys: list[str] | None = None,
        quantization_config: dict | None = None,
    ) -> str:
        checkpoint_dir = tmp_path / f"checkpoint_{rng.integers(0, 99999)}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if create_metadata:
            metadata = metadata_factory(
                input_keys=input_keys,
                output_keys=output_keys,
                model_filename=model_filename,
                include_training_path=include_training_path,
                artifact_format=artifact_format,
                backend_name=backend_name,
                quantization_workflow=quantization_workflow,
                exclude_keys=exclude_metadata_keys,
            )
            with open(
                checkpoint_dir / CompressionFilename.COMPRESSION_METADATA.value,
                "w",
            ) as file:
                json.dump(metadata, file)

        if create_model:
            (checkpoint_dir / model_filename).write_text("dummy")
        if create_normalizer:
            (checkpoint_dir / "normalizer.pt").write_text("dummy")
        if quantization_config is not None:
            OmegaConf.save(
                config=OmegaConf.create(quantization_config),
                f=checkpoint_dir / "quantization_config.yaml",
            )

        return str(checkpoint_dir)

    return factory


@pytest.fixture
def mock_policy_factory() -> Callable[..., MagicMock]:
    """Factory for mock Policy objects with normalizer and spaces."""

    def factory() -> MagicMock:
        mock_policy = MagicMock()
        mock_policy.observation_space = MagicMock(spec=ObservationSpace)
        mock_policy.observation_space.depth_cameras = {}
        mock_policy.action_space = MagicMock(spec=ActionSpace)
        mock_policy.prediction_horizon = 16
        mock_policy.decoder.observation_horizon = 2
        mock_policy.denoising_thresholds.params_dict = {}
        mock_policy.normalizer.params_dict = {}
        mock_policy.tokenizer = None
        return mock_policy

    return factory


@pytest.fixture
def loaded_loader_factory(
    checkpoint_directory_factory: Callable[..., str],
) -> Callable[..., CompressedPolicyRuntime]:
    """Factory that creates a CompressedPolicyRuntime via patched __init__."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
        device: str = "cpu",
        checkpoint_kwargs: dict | None = None,
    ) -> CompressedPolicyRuntime:
        if checkpoint_kwargs is None:
            checkpoint_kwargs = {}
        checkpoint_path = checkpoint_directory_factory(
            input_keys=input_keys,
            output_keys=output_keys,
            **checkpoint_kwargs,
        )
        mock_exported_program = MagicMock()
        mock_exported_program.module.return_value = MagicMock()

        # torch.compile MUST be patched: calling it on a MagicMock
        # triggers infinite dynamo graph expansion (the tracer follows
        # mock attribute access endlessly), consuming all system memory.
        with (
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.export.load",
                return_value=mock_exported_program,
            ),
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.torch.load", return_value={}),
            patch.object(CompressedCheckpointLoader, "_load_training_config"),
            patch.object(
                CompressedCheckpointLoader,
                "_load_tokenizer",
                return_value=None,
            ),
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.compile", return_value=MagicMock()
            ),
        ):
            return CompressedPolicyRuntime(
                device=torch.device(device),
                checkpoint_path=checkpoint_path,
            )

    return factory


@pytest.fixture
def inference_loader_factory(
    rng: np.random.Generator,
    mock_policy_factory: Callable[..., MagicMock],
) -> Callable[..., CompressedPolicyRuntime]:
    """Factory for a CompressedPolicyRuntime bypassing __init__ for inference tests."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
        model_outputs: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        tokenizer: MagicMock | None = None,
        artifact_format: str = ArtifactFormat.TORCH_EXPORT_PT2.value,
    ) -> CompressedPolicyRuntime:
        if input_keys is None:
            input_keys = ["left"]
        if output_keys is None:
            output_keys = ["position"]

        mock_policy = mock_policy_factory()
        mock_policy.observation_space.observations_metadata = {}
        mock_policy.action_space.actions_metadata = {}

        mock_compressed_model = MagicMock()
        if model_outputs is None:
            model_outputs = tuple(
                torch.from_numpy(rng.standard_normal((1, 16, 3)).astype(np.float32))
                for _ in output_keys
            )
        mock_compressed_model.return_value = model_outputs

        loader = CompressedPolicyRuntime.__new__(CompressedPolicyRuntime)
        checkpoint_loader = MagicMock(spec=CompressedCheckpointLoader)
        checkpoint_loader.device = torch.device("cpu")
        checkpoint_loader.checkpoint_path = "/tmp/compressed"
        checkpoint_loader.config = MagicMock()
        checkpoint_loader.tokenizer = tokenizer
        checkpoint_loader.policy = mock_policy
        checkpoint_loader.observation_space = mock_policy.observation_space
        checkpoint_loader.action_space = mock_policy.action_space
        checkpoint_loader.prediction_horizon = mock_policy.prediction_horizon
        checkpoint_loader.observation_horizon = mock_policy.decoder.observation_horizon
        checkpoint_loader.denoising_thresholds = {}
        checkpoint_loader.depth_clamp_ranges = None
        checkpoint_loader.input_keys = input_keys
        checkpoint_loader.output_keys = output_keys
        checkpoint_loader.artifact_format = artifact_format
        checkpoint_loader.normalizer = LinearNormalizer()
        loader.checkpoint_loader = checkpoint_loader
        loader._client_identifier = checkpoint_loader.checkpoint_path
        loader._policy = mock_policy
        loader._compressed_model = mock_compressed_model
        return loader

    return factory


@pytest.fixture
def observation_dict_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for observation dicts with spatial tensors."""

    def factory(
        keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["left"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((1, 2, 3, 32, 32)).astype(np.float32)
            )
            for key in keys
        }

    return factory


@pytest.mark.unit
class TestCompressedPolicyRuntimeErrors:
    @pytest.mark.parametrize(
        "factory_kwargs, error_type, match_fragment",
        [
            (
                {"create_metadata": False},
                FileNotFoundError,
                "Compression metadata not found at",
            ),
            (
                {"create_model": False},
                FileNotFoundError,
                "Compressed model not found at",
            ),
            (
                {"create_normalizer": False},
                FileNotFoundError,
                "Normalizer not found at",
            ),
            (
                {"include_training_path": False},
                ValueError,
                "Compression metadata is missing",
            ),
        ],
        ids=["no_metadata", "no_model", "no_normalizer", "no_training_path"],
    )
    def test_init_errors(
        self,
        checkpoint_directory_factory,
        factory_kwargs,
        error_type,
        match_fragment,
    ):
        checkpoint_path = checkpoint_directory_factory(**factory_kwargs)

        with pytest.raises(error_type, match=match_fragment):
            CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )

    @pytest.mark.parametrize(
        "excluded_key",
        [
            CompressionMetadataKey.INPUT_KEYS.value,
            CompressionMetadataKey.MODEL_FILE.value,
        ],
    )
    def test_raises_key_error_when_metadata_key_missing(
        self,
        checkpoint_directory_factory,
        excluded_key,
    ):
        checkpoint_path = checkpoint_directory_factory(
            exclude_metadata_keys=[excluded_key],
        )

        with pytest.raises(KeyError, match=excluded_key):
            CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )


@pytest.mark.unit
class TestCompressedPolicyRuntimeMetadata:
    def test_populates_input_and_output_keys(self, loaded_loader_factory):
        input_keys = ["depth", "left", "right"]
        output_keys = ["gripper", "orientation", "position"]

        loader = loaded_loader_factory(
            input_keys=input_keys,
            output_keys=output_keys,
        )

        assert loader.input_keys == input_keys
        assert loader.output_keys == output_keys

    def test_stores_device(self, loaded_loader_factory):
        loader = loaded_loader_factory()
        assert loader.device == torch.device("cpu")

    @pytest.mark.parametrize(
        "property_name",
        ["input_keys", "output_keys"],
    )
    def test_key_properties_return_copies(self, loaded_loader_factory, property_name):
        loader = loaded_loader_factory()
        original = getattr(loader, property_name)

        returned = getattr(loader, property_name)
        returned.append("mutated")

        assert getattr(loader, property_name) == original


@pytest.mark.unit
class TestCompressedPolicyRuntimeArtifacts:
    def test_pte_artifact_uses_executorch_adapter_without_torch_export(
        self,
        checkpoint_directory_factory: Callable[..., str],
    ):
        checkpoint_path = checkpoint_directory_factory(
            model_filename=CompressionFilename.EXECUTORCH_MODEL.value,
            artifact_format=ArtifactFormat.EXECUTORCH_PTE.value,
            backend_name=DeploymentBackendName.EXECUTORCH_XNNPACK.value,
            quantization_workflow=QuantizationWorkflow.EAGER.value,
        )
        adapter = MagicMock()

        with (
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.export.load",
            ) as mock_export_load,
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.ExecuTorchModuleAdapter",
                return_value=adapter,
            ) as mock_adapter_class,
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.torch.load", return_value={}),
            patch.object(CompressedCheckpointLoader, "_load_training_config"),
            patch.object(
                CompressedCheckpointLoader,
                "_load_tokenizer",
                return_value=None,
            ),
            patch(f"{COMPRESSED_RUNTIME_MODULE}.torch.compile") as mock_compile,
        ):
            loader = CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )

        mock_export_load.assert_not_called()
        mock_compile.assert_not_called()
        mock_adapter_class.assert_called_once_with(
            model_path=str(
                Path(checkpoint_path) / CompressionFilename.EXECUTORCH_MODEL.value
            ),
        )
        assert loader._compressed_model is adapter

    def test_pte_artifact_rejects_non_cpu_device(
        self,
        checkpoint_directory_factory: Callable[..., str],
    ):
        checkpoint_path = checkpoint_directory_factory(
            model_filename=CompressionFilename.EXECUTORCH_MODEL.value,
            artifact_format=ArtifactFormat.EXECUTORCH_PTE.value,
            backend_name=DeploymentBackendName.EXECUTORCH_XNNPACK.value,
        )

        with (
            patch.object(CompressedCheckpointLoader, "_load_training_config"),
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.torch.load", return_value={}),
            patch.object(
                CompressedCheckpointLoader,
                "_load_tokenizer",
                return_value=None,
            ),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "ExecuTorch XNNPACK artifacts support CPU inference only, "
                    "got 'cuda'."
                ),
            ),
        ):
            CompressedPolicyRuntime(
                device=torch.device("cuda"),
                checkpoint_path=checkpoint_path,
            )

    def test_unknown_artifact_format_raises(self, checkpoint_directory_factory):
        checkpoint_path = checkpoint_directory_factory(artifact_format="unknown")

        with (
            patch.object(CompressedCheckpointLoader, "_load_training_config"),
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.torch.load", return_value={}),
            patch.object(
                CompressedCheckpointLoader,
                "_load_tokenizer",
                return_value=None,
            ),
            pytest.raises(
                ValueError,
                match=re.escape("Unsupported compression artifact format 'unknown'."),
            ),
        ):
            CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )


@pytest.mark.unit
class TestCompressedPolicyRuntimeProperties:
    def test_delegates_properties_to_policy(
        self,
        inference_loader_factory,
    ):
        loader = inference_loader_factory()
        mock_policy = loader._policy

        assert loader.observation_space == mock_policy.observation_space
        assert loader.action_space == mock_policy.action_space
        assert loader.prediction_horizon == 16
        assert loader.observation_horizon == 2
        assert loader.denoising_thresholds == {}
        assert loader.depth_clamp_ranges is None

    def test_client_identifier_returns_compressed_checkpoint_path(
        self,
        inference_loader_factory,
    ):
        loader = inference_loader_factory()
        loader._client_identifier = "/tmp/compressed"

        assert loader.client_identifier == "/tmp/compressed"


@pytest.mark.unit
class TestCompressedPolicyRuntimeRunInference:
    def test_returns_dict_with_output_keys(
        self,
        inference_loader_factory,
        observation_dict_factory,
        rng: np.random.Generator,
    ):
        output_keys = ["orientation", "position"]
        orientation = torch.from_numpy(
            rng.standard_normal((1, 16, 1)).astype(np.float32)
        )
        position = torch.from_numpy(rng.standard_normal((1, 16, 3)).astype(np.float32))
        loader = inference_loader_factory(
            output_keys=output_keys,
            model_outputs=(orientation, position),
        )

        result = loader.run_inference(obs_dict=observation_dict_factory())

        assert set(result.keys()) == set(output_keys)
        assert torch.equal(result["orientation"], orientation)
        assert torch.equal(result["position"], position)

    def test_calls_model_with_positional_tensors_in_key_order(
        self,
        inference_loader_factory,
        rng: np.random.Generator,
    ):
        input_keys = ["depth", "left"]
        loader = inference_loader_factory(input_keys=input_keys)

        depth = torch.from_numpy(
            rng.standard_normal((1, 2, 1, 32, 32)).astype(np.float32)
        )
        left = torch.from_numpy(
            rng.standard_normal((1, 2, 3, 32, 32)).astype(np.float32)
        )

        loader.run_inference(obs_dict={"depth": depth, "left": left})

        call_args = loader._compressed_model.call_args[0]
        assert len(call_args) == 2
        assert torch.equal(call_args[0], depth)
        assert torch.equal(call_args[1], left)

    def test_wraps_single_tensor_output_as_tuple(
        self,
        inference_loader_factory,
        observation_dict_factory,
        rng: np.random.Generator,
    ):
        single_output = torch.from_numpy(
            rng.standard_normal((1, 16, 3)).astype(np.float32)
        )
        loader = inference_loader_factory(
            model_outputs=single_output,
        )

        result = loader.run_inference(obs_dict=observation_dict_factory())

        assert "position" in result
        assert torch.equal(result["position"], single_output)

    def test_pte_model_receives_observation_tuple(
        self,
        inference_loader_factory,
        rng: np.random.Generator,
    ):
        input_keys = ["depth", "left"]
        loader = inference_loader_factory(
            input_keys=input_keys,
            artifact_format=ArtifactFormat.EXECUTORCH_PTE.value,
        )
        depth = torch.from_numpy(
            rng.standard_normal((1, 2, 1, 32, 32)).astype(np.float32)
        )
        left = torch.from_numpy(
            rng.standard_normal((1, 2, 3, 32, 32)).astype(np.float32)
        )

        loader.run_inference(obs_dict={"depth": depth, "left": left})

        call_args = loader._compressed_model.call_args.args
        assert len(call_args) == 1
        observation_tensors = call_args[0]
        assert torch.equal(observation_tensors[0], depth)
        assert torch.equal(observation_tensors[1], left)

    def test_raises_when_output_count_mismatches_metadata(
        self,
        inference_loader_factory,
        observation_dict_factory,
        rng: np.random.Generator,
    ):
        output = torch.from_numpy(rng.standard_normal((1, 16, 3)).astype(np.float32))
        loader = inference_loader_factory(
            output_keys=["orientation", "position"],
            model_outputs=(output,),
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Compressed model returned 1 tensors, "
                "but metadata declares 2 output keys."
            ),
        ):
            loader.run_inference(obs_dict=observation_dict_factory())


@pytest.mark.unit
class TestCompressedPolicyRuntimeNormalizationPipeline:
    def test_calls_normalize_observation(
        self,
        inference_loader_factory,
        observation_dict_factory,
    ):
        normalizer = LinearNormalizer()
        loader = inference_loader_factory()
        loader.checkpoint_loader.normalizer = normalizer
        obs_dict = observation_dict_factory()

        with patch(
            f"{COMPRESSED_RUNTIME_MODULE}.normalize_observation",
            return_value=obs_dict,
        ) as mock_normalize:
            loader.run_inference(obs_dict=obs_dict)

        mock_normalize.assert_called_once()
        assert mock_normalize.call_args[1]["normalizer"] is normalizer

    def test_calls_tokenize_when_tokenizer_present(
        self,
        inference_loader_factory,
        observation_dict_factory,
    ):
        mock_tokenizer = MagicMock()
        mock_obs_tokenizer = MagicMock()
        mock_tokenizer.observation_tokenizer = mock_obs_tokenizer
        loader = inference_loader_factory(tokenizer=mock_tokenizer)
        obs_dict = observation_dict_factory()

        with (
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.normalize_observation",
                return_value=obs_dict,
            ),
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.tokenize_observation",
                return_value=obs_dict,
            ) as mock_tokenize,
        ):
            loader.run_inference(obs_dict=obs_dict)

        mock_tokenize.assert_called_once()
        assert mock_tokenize.call_args[1]["obs_tokenizer"] is mock_obs_tokenizer

    def test_skips_tokenization_when_observation_tokenizer_is_none(
        self,
        inference_loader_factory,
        observation_dict_factory,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.observation_tokenizer = None
        loader = inference_loader_factory(tokenizer=mock_tokenizer)

        with (
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.normalize_observation",
                return_value=observation_dict_factory(),
            ),
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.tokenize_observation",
            ) as mock_tokenize,
        ):
            loader.run_inference(obs_dict=observation_dict_factory())

        mock_tokenize.assert_not_called()

    def test_calls_unnormalize_actions(
        self,
        inference_loader_factory,
        observation_dict_factory,
    ):
        normalizer = LinearNormalizer()
        loader = inference_loader_factory()
        loader.checkpoint_loader.normalizer = normalizer
        expected_result = {"position": torch.zeros(1, 16, 3)}

        with patch(
            f"{COMPRESSED_RUNTIME_MODULE}.unnormalize_actions",
            return_value=expected_result,
        ) as mock_unnormalize:
            result = loader.run_inference(obs_dict=observation_dict_factory())

        mock_unnormalize.assert_called_once()
        assert mock_unnormalize.call_args[1]["normalizer"] is normalizer
        assert result is expected_result


@pytest.mark.unit
class TestCompileModelForInference:
    @patch(f"{COMPRESSED_RUNTIME_MODULE}.torch.compile")
    def test_pt2e_backend_activates_environment_before_compile(self, mock_compile):
        model = MagicMock(spec=torch.nn.Module)
        compiled_model = MagicMock(spec=torch.nn.Module)
        mock_compile.return_value = compiled_model
        mock_backend = MagicMock(spec=BasePT2EBackend)

        result = CompressedPolicyRuntime._compile_model_for_inference(
            model=model,
            backend=mock_backend,
        )

        mock_backend.activate_environment.assert_called_once_with()
        mock_compile.assert_called_once_with(model)
        assert result is compiled_model

    @patch(f"{COMPRESSED_RUNTIME_MODULE}.torch.compile")
    def test_without_backend_compiles_model(self, mock_compile):
        model = MagicMock(spec=torch.nn.Module)
        compiled_model = MagicMock(spec=torch.nn.Module)
        mock_compile.return_value = compiled_model

        result = CompressedPolicyRuntime._compile_model_for_inference(
            model=model,
            backend=None,
        )

        mock_compile.assert_called_once_with(model)
        assert result is compiled_model


@pytest.mark.unit
class TestLoadBackend:
    @pytest.mark.parametrize(
        "workflow",
        [QuantizationWorkflow.EAGER.value, None],
        ids=["eager", "none"],
    )
    def test_returns_none_for_non_pt2e_workflow(self, loaded_loader_factory, workflow):
        loader = loaded_loader_factory()

        assert loader._load_backend(workflow=workflow) is None

    def test_returns_none_when_config_file_missing(self, loaded_loader_factory):
        loader = loaded_loader_factory()

        assert (
            loader._load_backend(
                workflow=QuantizationWorkflow.PT2E.value,
            )
            is None
        )

    def test_returns_none_when_full_config_has_no_pt2e_workflow(
        self, loaded_loader_factory
    ):
        loader = loaded_loader_factory(
            checkpoint_kwargs={
                "quantization_config": {
                    "_target_": "versatil.post_training_compression.compressor.PostTrainingCompressor",
                    "checkpoint_path": "/tmp/test",
                    "modules": [],
                    "preparation": {
                        "replace_frozen_batchnorm": True,
                        "fuse_conv_batchnorm": True,
                    },
                },
            },
        )

        assert (
            loader._load_backend(
                workflow=QuantizationWorkflow.PT2E.value,
            )
            is None
        )

    def test_instantiates_backend_from_saved_config(self, loaded_loader_factory):
        loader = loaded_loader_factory(
            checkpoint_kwargs={
                "quantization_config": {
                    "_target_": "versatil.post_training_compression.compressor.PostTrainingCompressor",
                    "checkpoint_path": "/tmp/test",
                    "modules": [],
                    "preparation": {
                        "replace_frozen_batchnorm": True,
                        "fuse_conv_batchnorm": True,
                    },
                    "quantization": {
                        "_target_": "versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow",
                        "targets": [
                            {
                                "_target_": "versatil.quantization.module_target.PT2EQuantizationModuleTarget",
                                "module_path": "",
                                "pt2e_backend": {
                                    "_target_": "versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend",
                                    "is_dynamic": False,
                                },
                            }
                        ],
                    },
                },
            },
        )

        backend = loader._load_backend(
            workflow=QuantizationWorkflow.PT2E.value,
        )

        assert isinstance(backend, BasePT2EBackend)
        assert backend.supported_device_types == ("cpu",)


@pytest.mark.unit
class TestValidateDevice:
    @pytest.mark.parametrize(
        "supported_types, expectation_factory",
        [
            (
                lambda actual_type: (actual_type,),
                lambda actual_type: does_not_raise(),
            ),
            (
                lambda actual_type: ("other",),
                lambda actual_type: pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Backend MagicMock supports devices ('other',), "
                        f"got '{actual_type}'."
                    ),
                ),
            ),
        ],
        ids=["supported", "unsupported"],
    )
    def test_device_validation(
        self,
        loaded_loader_factory,
        supported_types,
        expectation_factory,
    ):
        actual_device = get_test_device()
        loader = loaded_loader_factory(device=actual_device.type)
        mock_backend = MagicMock(spec=BasePT2EBackend)
        mock_backend.supported_device_types = supported_types(actual_device.type)

        with expectation_factory(actual_device.type):
            loader._validate_device(backend=mock_backend)


@pytest.mark.unit
class TestCompileModelFlag:
    @pytest.mark.parametrize("compile_model", [True, False])
    def test_compile_flag_controls_model_compilation(
        self,
        checkpoint_directory_factory: Callable[..., str],
        compile_model: bool,
    ):
        checkpoint_path = checkpoint_directory_factory()
        mock_exported_program = MagicMock()
        raw_module = MagicMock()
        raw_module.to.return_value = raw_module
        mock_exported_program.module.return_value = raw_module
        compiled_module = MagicMock()

        with (
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.export.load",
                return_value=mock_exported_program,
            ),
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.torch.load", return_value={}),
            patch.object(CompressedCheckpointLoader, "_load_training_config"),
            patch.object(
                CompressedCheckpointLoader,
                "_load_tokenizer",
                return_value=None,
            ),
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.compile",
                return_value=compiled_module,
            ),
        ):
            loader = CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
                compile_model=compile_model,
            )

        if compile_model:
            assert loader._compressed_model is compiled_module
        else:
            assert loader._compressed_model is raw_module


@pytest.mark.unit
class TestArtifactDevicePlacement:
    @pytest.mark.parametrize(
        "quantization_workflow, expects_move",
        [
            (QuantizationWorkflow.EAGER.value, True),
            (QuantizationWorkflow.NONE.value, True),
            (None, True),
            (QuantizationWorkflow.PT2E.value, False),
        ],
        ids=["eager_moves", "none_moves", "missing_workflow_moves", "pt2e_stays"],
    )
    def test_moves_pt2_module_to_device_for_non_pt2e_workflows(
        self,
        checkpoint_directory_factory: Callable[..., str],
        quantization_workflow: str | None,
        expects_move: bool,
    ):
        checkpoint_path = checkpoint_directory_factory(
            quantization_workflow=quantization_workflow,
        )
        mock_exported_program = MagicMock()
        raw_module = MagicMock()
        raw_module.to.return_value = raw_module
        mock_exported_program.module.return_value = raw_module
        device = torch.device("cpu")

        with (
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.export.load",
                return_value=mock_exported_program,
            ),
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.torch.load", return_value={}),
            patch.object(CompressedCheckpointLoader, "_load_training_config"),
            patch.object(
                CompressedCheckpointLoader,
                "_load_tokenizer",
                return_value=None,
            ),
            patch(
                f"{COMPRESSED_RUNTIME_MODULE}.torch.compile",
                return_value=MagicMock(),
            ),
        ):
            CompressedPolicyRuntime(
                device=device,
                checkpoint_path=checkpoint_path,
            )

        if expects_move:
            raw_module.to.assert_called_once_with(device)
        else:
            raw_module.to.assert_not_called()


@pytest.mark.unit
class TestShouldCompile:
    @pytest.mark.parametrize(
        "workflow, device_type, expected",
        [
            (QuantizationWorkflow.PT2E.value, "cpu", True),
            (QuantizationWorkflow.PT2E.value, "cuda", True),
            (QuantizationWorkflow.EAGER.value, "cpu", True),
            (QuantizationWorkflow.EAGER.value, "cuda", False),
            (None, "cuda", True),
        ],
        ids=[
            "pt2e_cpu",
            "pt2e_cuda",
            "eager_cpu",
            "eager_cuda_skips",
            "none_cuda",
        ],
    )
    def test_should_compile_decision(self, workflow, device_type, expected):
        result = CompressedPolicyRuntime._should_compile(
            workflow=workflow,
            device=torch.device(device_type),
        )

        assert result == expected

    def test_eager_cuda_logs_warning(self, caplog):
        CompressedPolicyRuntime._should_compile(
            workflow=QuantizationWorkflow.EAGER.value,
            device=torch.device("cuda"),
        )

        assert "Skipping torch.compile" in caplog.text
        assert "torch._int_mm" in caplog.text
