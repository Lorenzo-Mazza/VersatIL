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

from tests.post_training_compression.conftest import verify_reload_fidelity
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.post_training_compression.constants import (
    CompressionMetadataKey,
    QuantizationStrategy,
)
from versatil.post_training_compression.serialization import (
    load_compression_metadata,
    save_compressed_model,
)

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
        quantization_strategy: str = QuantizationStrategy.PT2E.value,
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
                quantization_strategy=quantization_strategy,
            )
        return output_dir, model

    return factory


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
        assert (
            metadata[CompressionMetadataKey.QUANTIZATION_STRATEGY.value]
            == QuantizationStrategy.PT2E.value
        )

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
                quantization_strategy=QuantizationStrategy.PT2E.value,
            )

        assert result == tmp_path / "output"


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

        assert result.get(CompressionMetadataKey.IS_DYNAMIC.value) is False


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
