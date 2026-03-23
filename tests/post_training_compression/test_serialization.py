"""Tests for versatil.post_training_compression.serialization module."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.post_training_compression.constants import CompressionMetadataKey
from versatil.post_training_compression.serialization import (
    _strip_redundant_weights,
    load_compression_metadata,
    load_quantization_metadata,
    save_compressed_model,
    save_quantized_model,
    verify_reload_fidelity,
)
from versatil.quantization.constants import QuantizationMetadataKey

TORCHAO_VERSION_PATCH = patch(
    "versatil.post_training_compression.serialization._get_torchao_version",
    return_value="0.1.0",
)


@dataclass
class MockQuantizationConfig:
    backend: str = "x86_inductor"
    strategy: str = "static_ptq"
    compile_backend: str = "inductor"


@pytest.fixture
def serialization_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for a simple model used in serialization tests."""

    def factory(
        input_features: int = 4,
        output_features: int = 2,
    ) -> nn.Module:
        model = nn.Linear(input_features, output_features, bias=False)
        with torch.no_grad():
            data = rng.standard_normal(model.weight.shape).astype(np.float32)
            model.weight.copy_(torch.from_numpy(data))
        return model

    return factory


@pytest.fixture
def normalizer_factory(
    rng: np.random.Generator,
) -> Callable[..., LinearNormalizer]:
    """Factory for a fitted normalizer."""

    def factory(
        keys: list[str] | None = None,
    ) -> LinearNormalizer:
        if keys is None:
            keys = ["left", "position"]
        normalizer = LinearNormalizer()
        data = {
            key: torch.from_numpy(rng.standard_normal((100, 3)).astype(np.float32))
            for key in keys
        }
        normalizer.fit(data)
        return normalizer

    return factory


@pytest.fixture
def state_dict_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for state dicts with configurable dtypes."""

    def factory(
        entries: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key, (shape, dtype) in entries.items():
            if dtype == torch.int8:
                result[key] = torch.from_numpy(
                    rng.integers(-128, 127, size=shape).astype(np.int8)
                )
            else:
                result[key] = torch.from_numpy(
                    rng.standard_normal(shape).astype(np.float32)
                )
        return result

    return factory


@pytest.fixture
def training_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    """Factory for training checkpoint directories with optional tokenizer."""

    def factory(
        has_tokenizer: bool = False,
    ) -> Path:
        train_dir = tmp_path / "training"
        train_dir.mkdir(exist_ok=True)
        (train_dir / "config.yaml").write_text("training_config: true")
        if has_tokenizer:
            tokenizer_dir = train_dir / "tokenizer"
            tokenizer_dir.mkdir()
            (tokenizer_dir / "vocab.txt").write_text("hello\nworld")
        return train_dir

    return factory


@pytest.fixture
def saved_quantized_dir(
    tmp_path: Path,
    serialization_model_factory: Callable[..., nn.Module],
) -> Callable[..., tuple[Path, nn.Module]]:
    """Save a quantized model and return (output_dir, model)."""

    def factory(
        observation_keys: list[str] | None = None,
        action_keys: list[str] | None = None,
    ) -> tuple[Path, nn.Module]:
        if observation_keys is None:
            observation_keys = ["left"]
        if action_keys is None:
            action_keys = ["position"]
        model = serialization_model_factory()
        config = MockQuantizationConfig()
        output_dir = tmp_path / "quantized_output"

        with TORCHAO_VERSION_PATCH:
            save_quantized_model(
                quantized_model=model,
                save_directory=str(output_dir),
                observation_keys=observation_keys,
                action_keys=action_keys,
                quantization_config=config,
                training_checkpoint_path="/tmp/training",
            )
        return output_dir, model

    return factory


@pytest.fixture
def saved_compressed_dir(
    tmp_path: Path,
    serialization_model_factory: Callable[..., nn.Module],
    normalizer_factory: Callable[..., LinearNormalizer],
    training_dir_factory: Callable[..., Path],
) -> Callable[..., tuple[Path, nn.Module]]:
    """Save a compressed model and return (output_dir, model)."""

    def factory(
        input_keys: list[str] | None = None,
        output_keys: list[str] | None = None,
        has_tokenizer: bool = False,
    ) -> tuple[Path, nn.Module]:
        if input_keys is None:
            input_keys = ["left"]
        if output_keys is None:
            output_keys = ["position"]
        model = serialization_model_factory()
        normalizer = normalizer_factory()
        example_inputs = (torch.zeros(2, 4),)
        train_dir = training_dir_factory(has_tokenizer=has_tokenizer)
        output_dir = tmp_path / "compressed_output"

        with TORCHAO_VERSION_PATCH:
            save_compressed_model(
                converted_model=model,
                example_inputs=example_inputs,
                save_directory=str(output_dir),
                input_keys=input_keys,
                output_keys=output_keys,
                normalizer=normalizer,
                training_checkpoint_path=str(train_dir),
                quantization_config=MockQuantizationConfig(),
            )
        return output_dir, model

    return factory


@pytest.mark.unit
class TestSaveQuantizedModel:
    def test_creates_expected_files(self, saved_quantized_dir):
        output_dir, _ = saved_quantized_dir()

        assert (output_dir / "quantized_policy_int8.pt").exists()
        assert (output_dir / "quantization_config.yaml").exists()
        assert (output_dir / "quantization_metadata.json").exists()

    def test_metadata_contains_all_fields(self, saved_quantized_dir):
        output_dir, _ = saved_quantized_dir(
            observation_keys=["depth", "left"],
            action_keys=["orientation", "position"],
        )

        with open(output_dir / "quantization_metadata.json") as file:
            metadata = json.load(file)

        assert (
            metadata[QuantizationMetadataKey.WEIGHTS_FILE.value]
            == "quantized_policy_int8.pt"
        )
        assert metadata[QuantizationMetadataKey.OBSERVATION_KEYS.value] == [
            "depth",
            "left",
        ]
        assert metadata[QuantizationMetadataKey.ACTION_KEYS.value] == [
            "orientation",
            "position",
        ]
        assert QuantizationMetadataKey.TORCHAO_VERSION.value in metadata
        assert QuantizationMetadataKey.TORCH_VERSION.value in metadata

    def test_config_yaml_matches_input(self, saved_quantized_dir):
        output_dir, _ = saved_quantized_dir()

        loaded_config = OmegaConf.load(output_dir / "quantization_config.yaml")
        assert loaded_config.backend == "x86_inductor"
        assert loaded_config.strategy == "static_ptq"

    def test_saved_weights_roundtrip(self, saved_quantized_dir):
        output_dir, model = saved_quantized_dir()
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        loaded_state = torch.load(
            output_dir / "quantized_policy_int8.pt", weights_only=True
        )

        for key in original_state:
            assert torch.equal(loaded_state[key], original_state[key])

    def test_returns_save_directory_path(
        self,
        tmp_path,
        serialization_model_factory,
    ):
        model = serialization_model_factory()

        with TORCHAO_VERSION_PATCH:
            result = save_quantized_model(
                quantized_model=model,
                save_directory=str(tmp_path / "output"),
                observation_keys=["left"],
                action_keys=["position"],
                quantization_config=MockQuantizationConfig(),
                training_checkpoint_path="/tmp/training",
            )

        assert result == tmp_path / "output"


@pytest.mark.unit
class TestSaveCompressedModel:
    def test_creates_expected_files(self, saved_compressed_dir):
        output_dir, _ = saved_compressed_dir()

        assert (output_dir / "compressed_policy.pt2").exists()
        assert (output_dir / "normalizer.pt").exists()
        assert (output_dir / "compression_metadata.json").exists()
        assert (output_dir / "quantization_config.yaml").exists()
        assert (output_dir / "config.yaml").exists()

    def test_metadata_contains_all_fields(self, saved_compressed_dir):
        output_dir, _ = saved_compressed_dir(
            input_keys=["depth", "left"],
            output_keys=["orientation", "position"],
        )

        with open(output_dir / "compression_metadata.json") as file:
            metadata = json.load(file)

        assert (
            metadata[CompressionMetadataKey.MODEL_FILE.value] == "compressed_policy.pt2"
        )
        assert metadata[CompressionMetadataKey.NORMALIZER_FILE.value] == "normalizer.pt"
        assert metadata[CompressionMetadataKey.INPUT_KEYS.value] == ["depth", "left"]
        assert metadata[CompressionMetadataKey.OUTPUT_KEYS.value] == [
            "orientation",
            "position",
        ]
        assert CompressionMetadataKey.TORCHAO_VERSION.value in metadata
        assert CompressionMetadataKey.TORCH_VERSION.value in metadata

    def test_normalizer_roundtrip(self, saved_compressed_dir):
        output_dir, _ = saved_compressed_dir()

        loaded_state = torch.load(output_dir / "normalizer.pt", weights_only=True)
        reloaded = LinearNormalizer()
        reloaded.load_state_dict(loaded_state)
        assert "left" in reloaded.params_dict
        assert "position" in reloaded.params_dict

    def test_copies_config_yaml_from_training(self, saved_compressed_dir):
        output_dir, _ = saved_compressed_dir()

        assert (output_dir / "config.yaml").read_text() == "training_config: true"

    def test_copies_tokenizer_directory(self, saved_compressed_dir):
        output_dir, _ = saved_compressed_dir(has_tokenizer=True)

        tokenizer_dir = output_dir / "tokenizer"
        assert tokenizer_dir.exists()
        assert (tokenizer_dir / "vocab.txt").read_text() == "hello\nworld"

    def test_returns_save_directory_path(
        self,
        tmp_path,
        serialization_model_factory,
        normalizer_factory,
        training_dir_factory,
    ):
        model = serialization_model_factory()
        train_dir = training_dir_factory()

        with TORCHAO_VERSION_PATCH:
            result = save_compressed_model(
                converted_model=model,
                example_inputs=(torch.zeros(2, 4),),
                save_directory=str(tmp_path / "output"),
                input_keys=["left"],
                output_keys=["position"],
                normalizer=normalizer_factory(),
                training_checkpoint_path=str(train_dir),
                quantization_config=MockQuantizationConfig(),
            )

        assert result == tmp_path / "output"


@pytest.mark.unit
class TestStripRedundantWeights:
    @pytest.mark.parametrize(
        "entries, removed_key, kept_keys",
        [
            (
                {
                    "layer.weight": ((10, 5), torch.float32),
                    "layer._packed": ((10, 5), torch.int8),
                },
                "layer.weight",
                ["layer._packed"],
            ),
            (
                {
                    "layer.weight": ((10, 5), torch.float32),
                    "layer.bias": ((5,), torch.float32),
                },
                None,
                ["layer.weight", "layer.bias"],
            ),
            (
                {
                    "layer.weight": ((10, 5), torch.float32),
                    "other._packed": ((10, 3), torch.int8),
                },
                None,
                ["layer.weight", "other._packed"],
            ),
            (
                {
                    "layer.bias": ((5,), torch.float32),
                    "layer._packed": ((5,), torch.int8),
                },
                None,
                ["layer.bias", "layer._packed"],
            ),
        ],
        ids=[
            "removes_float32_with_int8_match",
            "keeps_without_int8",
            "keeps_with_different_numel",
            "keeps_non_weight_key",
        ],
    )
    def test_stripping_logic(
        self,
        state_dict_factory,
        entries,
        removed_key,
        kept_keys,
    ):
        state_dict = state_dict_factory(entries=entries)

        result = _strip_redundant_weights(state_dict=state_dict)

        for key in kept_keys:
            assert key in result
        if removed_key is not None:
            assert removed_key not in result


@pytest.mark.unit
class TestLoadQuantizationMetadata:
    def test_loads_json_metadata(self, tmp_path: Path):
        metadata = {
            QuantizationMetadataKey.WEIGHTS_FILE.value: "quantized_policy_int8.pt",
            QuantizationMetadataKey.OBSERVATION_KEYS.value: ["left"],
            QuantizationMetadataKey.ACTION_KEYS.value: ["position"],
        }
        with open(tmp_path / "quantization_metadata.json", "w") as file:
            json.dump(metadata, file)

        result = load_quantization_metadata(
            metadata_path=str(tmp_path / "quantization_metadata.json"),
        )

        assert (
            result[QuantizationMetadataKey.WEIGHTS_FILE.value]
            == "quantized_policy_int8.pt"
        )
        assert result[QuantizationMetadataKey.OBSERVATION_KEYS.value] == ["left"]

    def test_merges_config_yaml_with_flat_fields(self, tmp_path: Path):
        with open(tmp_path / "quantization_metadata.json", "w") as file:
            json.dump({QuantizationMetadataKey.WEIGHTS_FILE.value: "test.pt"}, file)
        OmegaConf.save(
            config=OmegaConf.create(
                {
                    "is_dynamic": True,
                    "is_qat": False,
                    "reduce_range": True,
                }
            ),
            f=tmp_path / "quantization_config.yaml",
        )

        result = load_quantization_metadata(
            metadata_path=str(tmp_path / "quantization_metadata.json"),
        )

        assert result[QuantizationMetadataKey.IS_DYNAMIC.value] is True
        assert result[QuantizationMetadataKey.REDUCE_RANGE.value] is True

    def test_merges_config_yaml_with_pt2e_nested_backend(self, tmp_path: Path):
        with open(tmp_path / "quantization_metadata.json", "w") as file:
            json.dump({QuantizationMetadataKey.WEIGHTS_FILE.value: "test.pt"}, file)
        OmegaConf.save(
            config=OmegaConf.create(
                {
                    "pt2e_backend": {
                        "is_dynamic": True,
                        "is_qat": False,
                        "reduce_range": False,
                    },
                }
            ),
            f=tmp_path / "quantization_config.yaml",
        )

        result = load_quantization_metadata(
            metadata_path=str(tmp_path / "quantization_metadata.json"),
        )

        assert result[QuantizationMetadataKey.IS_DYNAMIC.value] is True
        assert result[QuantizationMetadataKey.IS_QAT.value] is False

    def test_uses_defaults_when_config_fields_missing(self, tmp_path: Path):
        with open(tmp_path / "quantization_metadata.json", "w") as file:
            json.dump({QuantizationMetadataKey.WEIGHTS_FILE.value: "test.pt"}, file)
        OmegaConf.save(
            config=OmegaConf.create({}), f=tmp_path / "quantization_config.yaml"
        )

        result = load_quantization_metadata(
            metadata_path=str(tmp_path / "quantization_metadata.json"),
        )

        assert result[QuantizationMetadataKey.IS_DYNAMIC.value] is False
        assert result[QuantizationMetadataKey.IS_QAT.value] is False


@pytest.mark.unit
class TestLoadCompressionMetadata:
    def test_loads_json_metadata(self, tmp_path: Path):
        metadata = {
            CompressionMetadataKey.MODEL_FILE.value: "compressed_policy.pt2",
            CompressionMetadataKey.INPUT_KEYS.value: ["left"],
            CompressionMetadataKey.OUTPUT_KEYS.value: ["position"],
        }
        with open(tmp_path / "compression_metadata.json", "w") as file:
            json.dump(metadata, file)

        result = load_compression_metadata(
            metadata_path=str(tmp_path / "compression_metadata.json"),
        )

        assert (
            result[CompressionMetadataKey.MODEL_FILE.value] == "compressed_policy.pt2"
        )
        assert result[CompressionMetadataKey.INPUT_KEYS.value] == ["left"]

    def test_uses_defaults_when_config_empty(self, tmp_path: Path):
        with open(tmp_path / "compression_metadata.json", "w") as file:
            json.dump({CompressionMetadataKey.MODEL_FILE.value: "test.pt2"}, file)
        OmegaConf.save(
            config=OmegaConf.create({}), f=tmp_path / "quantization_config.yaml"
        )

        result = load_compression_metadata(
            metadata_path=str(tmp_path / "compression_metadata.json"),
        )

        assert result.get(QuantizationMetadataKey.IS_DYNAMIC.value) is False


@pytest.mark.unit
class TestVerifyReloadFidelity:
    @pytest.mark.parametrize(
        "original_outputs, reloaded_outputs, expected",
        [
            (
                (torch.tensor([1.0, 2.0]),),
                (torch.tensor([1.0, 2.0]),),
                True,
            ),
            (
                (torch.tensor([1.0, 2.0]),),
                (torch.tensor([1.0, 3.0]),),
                False,
            ),
            (
                (torch.tensor([1.0]),),
                (torch.tensor([1.0]), torch.tensor([2.0])),
                False,
            ),
        ],
        ids=["identical", "different_values", "different_counts"],
    )
    def test_fidelity_check(
        self,
        original_outputs,
        reloaded_outputs,
        expected,
    ):
        original_model = MagicMock(spec=nn.Module)
        original_model.return_value = original_outputs
        reloaded_model = MagicMock(spec=nn.Module)
        reloaded_model.return_value = reloaded_outputs

        result = verify_reload_fidelity(
            original_model=original_model,
            reloaded_model=reloaded_model,
            example_inputs=(torch.zeros(2, 4),),
        )

        assert result is expected
