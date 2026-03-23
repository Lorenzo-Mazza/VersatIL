"""Tests for versatil.inference.compressed_policy_loader module."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import Cameras
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.inference.policy_loading import BasePolicyLoader, CompressedPolicyLoader
from versatil.post_training_compression.constants import (
    CompressionFilename,
    CompressionMetadataKey,
)

COMPRESSED_LOADER_MODULE = "versatil.inference.policy_loading.compressed_loader"


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
        exclude_keys: list[str] | None = None,
    ) -> dict:
        if input_keys is None:
            input_keys = ["depth", "left"]
        if output_keys is None:
            output_keys = ["orientation", "position"]

        metadata = {
            CompressionMetadataKey.MODEL_FILE.value: model_filename,
            CompressionMetadataKey.NORMALIZER_FILE.value: normalizer_filename,
            CompressionMetadataKey.INPUT_KEYS.value: input_keys,
            CompressionMetadataKey.OUTPUT_KEYS.value: output_keys,
            CompressionMetadataKey.TORCHAO_VERSION.value: "0.16.0",
            CompressionMetadataKey.TORCH_VERSION.value: "2.10.0",
        }
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
        exclude_metadata_keys: list[str] | None = None,
    ) -> str:
        checkpoint_dir = tmp_path / f"checkpoint_{rng.integers(0, 99999)}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if create_metadata:
            metadata = metadata_factory(
                input_keys=input_keys,
                output_keys=output_keys,
                include_training_path=include_training_path,
                exclude_keys=exclude_metadata_keys,
            )
            with open(
                checkpoint_dir / CompressionFilename.COMPRESSION_METADATA.value,
                "w",
            ) as file:
                json.dump(metadata, file)

        if create_model:
            (checkpoint_dir / "compressed_policy.pt2").write_text("dummy")
        if create_normalizer:
            (checkpoint_dir / "normalizer.pt").write_text("dummy")

        return str(checkpoint_dir)

    return factory


@pytest.fixture
def mock_policy_factory() -> Callable[..., MagicMock]:
    """Factory for mock Policy objects with normalizer and spaces."""

    def factory() -> MagicMock:
        mock_policy = MagicMock()
        mock_policy.observation_space = MagicMock(spec=ObservationSpace)
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
) -> Callable[..., CompressedPolicyLoader]:
    """Factory that creates a CompressedPolicyLoader via patched __init__."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
    ) -> CompressedPolicyLoader:
        checkpoint_path = checkpoint_directory_factory(
            input_keys=input_keys,
            output_keys=output_keys,
        )
        mock_exported_program = MagicMock()
        mock_exported_program.module.return_value = MagicMock()

        with (
            patch(
                f"{COMPRESSED_LOADER_MODULE}.torch.export.load",
                return_value=mock_exported_program,
            ),
            patch(f"{COMPRESSED_LOADER_MODULE}.torch.load", return_value={}),
            patch.object(CompressedPolicyLoader, "_load_training_config"),
            patch.object(
                CompressedPolicyLoader,
                "_load_tokenizer",
                return_value=None,
            ),
        ):
            return CompressedPolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )

    return factory


@pytest.fixture
def inference_loader_factory(
    rng: np.random.Generator,
    mock_policy_factory: Callable[..., MagicMock],
) -> Callable[..., CompressedPolicyLoader]:
    """Factory for a CompressedPolicyLoader bypassing __init__ for inference tests."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
        model_outputs: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        tokenizer: MagicMock | None = None,
    ) -> CompressedPolicyLoader:
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

        loader = CompressedPolicyLoader.__new__(CompressedPolicyLoader)
        loader._device = torch.device("cpu")
        loader._policy = mock_policy
        loader._input_keys = input_keys
        loader._output_keys = output_keys
        loader._compressed_model = mock_compressed_model
        loader._normalizer = LinearNormalizer()
        loader._tokenizer = tokenizer
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
class TestCompressedPolicyLoaderErrors:
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
            CompressedPolicyLoader(
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
            CompressedPolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )


@pytest.mark.unit
class TestCompressedPolicyLoaderMetadata:
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
class TestCompressedPolicyLoaderProperties:
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
        assert loader.depth_clamp_range is None


@pytest.mark.unit
class TestDepthClampRange:
    def test_returns_none_when_depth_not_in_normalizer(
        self,
        mock_policy_factory: Callable[..., MagicMock],
    ):
        loader = CompressedPolicyLoader.__new__(CompressedPolicyLoader)
        loader._normalizer = LinearNormalizer()
        loader._policy = mock_policy_factory()

        assert loader.depth_clamp_range is None

    def test_returns_min_max_when_depth_stats_present(
        self,
        mock_policy_factory: Callable[..., MagicMock],
    ):
        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                Cameras.DEPTH.value: torch.tensor([[0.1], [0.5], [0.9], [0.2]]),
            }
        )

        loader = CompressedPolicyLoader.__new__(CompressedPolicyLoader)
        loader._normalizer = normalizer
        loader._policy = mock_policy_factory()

        result = loader.depth_clamp_range

        assert result is not None
        minimum, maximum = result
        assert minimum == pytest.approx(0.1, abs=1e-5)
        assert maximum == pytest.approx(0.9, abs=1e-5)


@pytest.mark.unit
class TestCompressedPolicyLoaderRunInference:
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


@pytest.mark.unit
class TestBasePolicyLoaderRunInference:
    def test_raises_not_implemented_error(self, tmp_path: Path):
        loader = BasePolicyLoader(
            device=torch.device("cpu"),
            checkpoint_path=str(tmp_path),
        )

        with pytest.raises(NotImplementedError):
            loader.run_inference(obs_dict={})


@pytest.mark.unit
class TestCompressedPolicyLoaderNormalizationPipeline:
    def test_calls_normalize_observation(
        self,
        inference_loader_factory,
        observation_dict_factory,
    ):
        normalizer = LinearNormalizer()
        loader = inference_loader_factory()
        loader._normalizer = normalizer
        obs_dict = observation_dict_factory()

        with patch(
            f"{COMPRESSED_LOADER_MODULE}.normalize_observation",
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
                f"{COMPRESSED_LOADER_MODULE}.normalize_observation",
                return_value=obs_dict,
            ),
            patch(
                f"{COMPRESSED_LOADER_MODULE}.tokenize_observation",
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
                f"{COMPRESSED_LOADER_MODULE}.normalize_observation",
                return_value=observation_dict_factory(),
            ),
            patch(
                f"{COMPRESSED_LOADER_MODULE}.tokenize_observation",
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
        loader._normalizer = normalizer
        expected_result = {"position": torch.zeros(1, 16, 3)}

        with patch(
            f"{COMPRESSED_LOADER_MODULE}.unnormalize_actions",
            return_value=expected_result,
        ) as mock_unnormalize:
            result = loader.run_inference(obs_dict=observation_dict_factory())

        mock_unnormalize.assert_called_once()
        assert mock_unnormalize.call_args[1]["normalizer"] is normalizer
        assert result is expected_result


@pytest.mark.unit
class TestCompressedPolicyLoaderTrainingConfig:
    def test_falls_back_to_remote_config_when_local_absent(self):
        loader = CompressedPolicyLoader.__new__(CompressedPolicyLoader)
        loader._device = torch.device("cpu")
        loader._checkpoint_path = "/tmp/compressed"

        mock_config = MagicMock()
        mock_config.policy = MagicMock()

        with patch.object(
            BasePolicyLoader,
            "_load_config",
            return_value=mock_config,
        ) as mock_load_config:
            loader._load_training_config(
                training_checkpoint_path="/tmp/train",
            )

        mock_load_config.assert_called_once_with(
            config_path="/tmp/train/config.yaml",
        )
        assert loader._policy is mock_config.policy
        mock_config.policy.to.assert_called_once_with(torch.device("cpu"))

    def test_prefers_local_config_when_present(self, tmp_path: Path):
        compressed_dir = tmp_path / "compressed"
        compressed_dir.mkdir()
        (compressed_dir / "config.yaml").write_text("local: true")

        loader = CompressedPolicyLoader.__new__(CompressedPolicyLoader)
        loader._device = torch.device("cpu")
        loader._checkpoint_path = str(compressed_dir)

        mock_config = MagicMock()
        mock_config.policy = MagicMock()

        with patch.object(
            BasePolicyLoader,
            "_load_config",
            return_value=mock_config,
        ) as mock_load_config:
            loader._load_training_config(
                training_checkpoint_path="/tmp/train",
            )

        mock_load_config.assert_called_once_with(
            config_path=str(compressed_dir / "config.yaml"),
        )
