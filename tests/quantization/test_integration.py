"""Integration tests for the versatil.quantization pipeline."""

import os
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch._inductor.config as inductor_config
from omegaconf import OmegaConf
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

from versatil.models.layers.normalization.frozen_batchnorm import (
    FrozenBatchNorm2d,
)
from versatil.post_training_compression.constants import PrunableLayerType
from versatil.post_training_compression.preparation.batchnorm import (
    prepare_batchnorms_for_quantization,
    replace_frozen_batchnorm,
)
from versatil.post_training_compression.preparation.fusion import (
    fuse_all_conv_batchnorm_pairs,
)
from versatil.post_training_compression.pruning import (
    StructuredPruner,
    UnstructuredPruner,
)
from versatil.post_training_compression.serialization import (
    _strip_redundant_weights,
    load_quantization_metadata,
    save_quantized_model,
)
from versatil.quantization.constants import QuantizationMetadataKey
from versatil.quantization.torch_patches import patch_get_source_partitions


@pytest.fixture(autouse=True, scope="session")
def _configure_quantization_env():
    """Set env vars and patch source partitions once per session."""
    original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")
    original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_cpp = inductor_config.cpp_wrapper
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    inductor_config.cpp_wrapper = True
    patch_get_source_partitions()
    yield
    if original_freezing is None:
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
    else:
        os.environ["TORCHINDUCTOR_FREEZING"] = original_freezing
    if original_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
    inductor_config.cpp_wrapper = original_cpp


class _SyntheticModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dimension: int,
        use_frozen_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.batchnorm = (
            FrozenBatchNorm2d(dimension=hidden_channels)
            if use_frozen_bn
            else nn.BatchNorm2d(num_features=hidden_channels)
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_channels, output_dimension)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.relu(self.batchnorm(self.conv(image)))
        return (self.linear(self.flatten(self.pool(features))),)


class _TwoPartModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_dimension),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.decoder(self.encoder(image)),)


@pytest.fixture
def synthetic_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for _SyntheticModel with deterministic weights."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_dimension: int = 4,
        use_frozen_bn: bool = False,
    ) -> nn.Module:
        model = _SyntheticModel(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
            use_frozen_bn=use_frozen_bn,
        )
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**31)))
        for parameter in model.parameters():
            nn.init.normal_(parameter, generator=generator)
        if use_frozen_bn:
            for buffer in model.buffers():
                buffer.data.normal_(generator=generator)
            model.batchnorm.running_var.data.abs_()
        model.eval()
        return model

    return factory


@pytest.fixture
def example_inputs_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Factory for spatial input tensors as tuple."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
        image_size: int = 16,
    ) -> tuple[torch.Tensor, ...]:
        image = torch.from_numpy(
            rng.standard_normal((batch_size, channels, image_size, image_size)).astype(
                np.float32
            )
        )
        return (image,)

    return factory


@pytest.fixture
def two_part_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for _TwoPartModel with deterministic weights."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_dimension: int = 4,
    ) -> nn.Module:
        model = _TwoPartModel(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
        )
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**31)))
        for parameter in model.parameters():
            nn.init.normal_(parameter, generator=generator)
        model.eval()
        return model

    return factory


def _prepare_and_export(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> nn.Module:
    """Prepare BN, fuse, export. Shared pipeline helper."""
    prepare_batchnorms_for_quantization(model)
    fuse_all_conv_batchnorm_pairs(model)
    return torch.export.export(model, example_inputs, strict=False).module()


def _quantize_pt2e(
    exported: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    is_dynamic: bool = False,
    calibration_steps: int = 3,
) -> nn.Module:
    """Apply PT2E quantization pipeline. Shared helper."""
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        get_default_x86_inductor_quantization_config(is_dynamic=is_dynamic)
    )
    prepared = prepare_pt2e(exported, quantizer)
    for _ in range(calibration_steps):
        prepared(*example_inputs)
    return convert_pt2e(prepared)


def _assert_has_quantize_ops(quantized_model: nn.Module) -> None:
    """Assert the FX graph contains quantize/dequantize ops."""
    graph_str = str(quantized_model.graph)
    quantize_keywords = [
        "quantize_per_tensor",
        "dequantize_per_tensor",
        "quantize_per_channel",
        "dequantize_per_channel",
        "choose_qparams",
    ]
    assert any(keyword in graph_str for keyword in quantize_keywords), (
        f"Graph has no quantize/dequantize ops.\nGraph:\n{graph_str}"
    )


@pytest.fixture
def quantized_model_factory(
    synthetic_model_factory: Callable[..., nn.Module],
    example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
) -> Callable[..., tuple[nn.Module, nn.Module, tuple[torch.Tensor, ...]]]:
    """Factory producing (float_model, quantized_model, example_inputs)."""

    def factory(
        is_dynamic: bool = False,
        hidden_channels: int = 16,
    ) -> tuple[nn.Module, nn.Module, tuple[torch.Tensor, ...]]:
        float_model = synthetic_model_factory(
            hidden_channels=hidden_channels,
        )
        example_inputs = example_inputs_factory()

        exported = _prepare_and_export(
            model=float_model,
            example_inputs=example_inputs,
        )
        quantized = _quantize_pt2e(
            exported=exported,
            example_inputs=example_inputs,
            is_dynamic=is_dynamic,
        )
        return float_model, quantized, example_inputs

    return factory


@pytest.mark.integration
class TestPreparationIntegration:
    def test_batchnorm_replacement_preserves_output(
        self,
        synthetic_model_factory,
        example_inputs_factory,
    ):
        model = synthetic_model_factory(use_frozen_bn=True)
        example_inputs = example_inputs_factory()

        with torch.no_grad():
            original_output = model(*example_inputs)

        replace_frozen_batchnorm(model)

        with torch.no_grad():
            replaced_output = model(*example_inputs)

        assert torch.allclose(original_output[0], replaced_output[0], atol=1e-5)

    def test_conv_bn_fusion_preserves_output(
        self,
        synthetic_model_factory,
        example_inputs_factory,
    ):
        model = synthetic_model_factory()
        example_inputs = example_inputs_factory()

        with torch.no_grad():
            original_output = model(*example_inputs)

        prepare_batchnorms_for_quantization(model)
        fuse_all_conv_batchnorm_pairs(model)

        with torch.no_grad():
            fused_output = model(*example_inputs)

        assert torch.allclose(original_output[0], fused_output[0], atol=1e-5)

    def test_frozen_bn_replacement_then_fusion_then_quantization(
        self,
        synthetic_model_factory,
        example_inputs_factory,
    ):
        model = synthetic_model_factory(use_frozen_bn=True)
        example_inputs = example_inputs_factory()

        with torch.no_grad():
            float_output = model(*example_inputs)

        exported = _prepare_and_export(model=model, example_inputs=example_inputs)
        quantized = _quantize_pt2e(exported=exported, example_inputs=example_inputs)

        with torch.no_grad():
            quantized_output = quantized(*example_inputs)

        _assert_has_quantize_ops(quantized)
        assert torch.isfinite(quantized_output[0]).all()
        # Frozen BN replacement + fusion introduces small numerical drift
        max_diff = (float_output[0] - quantized_output[0]).abs().max().item()
        assert max_diff < 1.0


@pytest.mark.integration
class TestPT2EQuantizationPipeline:
    @pytest.mark.parametrize("is_dynamic", [False, True])
    def test_quantization_produces_quantized_ops(
        self,
        quantized_model_factory,
        is_dynamic,
    ):
        _, quantized_model, _ = quantized_model_factory(
            is_dynamic=is_dynamic,
        )
        _assert_has_quantize_ops(quantized_model)

    def test_quantized_output_shape_matches_float(
        self,
        quantized_model_factory,
    ):
        float_model, quantized_model, example_inputs = quantized_model_factory()

        with torch.no_grad():
            float_output = float_model(*example_inputs)
            quantized_output = quantized_model(*example_inputs)

        for float_tensor, quant_tensor in zip(float_output, quantized_output):
            assert float_tensor.shape == quant_tensor.shape

    def test_quantized_output_is_close_to_float(
        self,
        quantized_model_factory,
    ):
        float_model, quantized_model, example_inputs = quantized_model_factory()

        with torch.no_grad():
            float_output = float_model(*example_inputs)
            quantized_output = quantized_model(*example_inputs)

        # Quantization-only divergence is bounded by int8 precision
        max_diff = (float_output[0] - quantized_output[0]).abs().max().item()
        assert max_diff < 0.5


@pytest.mark.integration
class TestExportPipeline:
    def test_exported_output_matches_eager(
        self,
        synthetic_model_factory,
        example_inputs_factory,
    ):
        model = synthetic_model_factory()
        example_inputs = example_inputs_factory()

        prepare_batchnorms_for_quantization(model)
        fuse_all_conv_batchnorm_pairs(model)

        with torch.no_grad():
            eager_output = model(*example_inputs)

        exported = torch.export.export(model, example_inputs, strict=False).module()

        with torch.no_grad():
            exported_output = exported(*example_inputs)

        assert torch.allclose(eager_output[0], exported_output[0], atol=1e-6)


@pytest.mark.integration
class TestSaveLoadRoundtrip:
    def test_state_dict_roundtrip_is_exact(
        self,
        tmp_path,
        quantized_model_factory,
    ):
        _, quantized_model, example_inputs = quantized_model_factory()

        with torch.no_grad():
            original_output = quantized_model(*example_inputs)

        torch.save(quantized_model.state_dict(), tmp_path / "weights.pt")
        loaded = torch.load(tmp_path / "weights.pt", weights_only=True)
        quantized_model.load_state_dict(loaded)

        with torch.no_grad():
            reloaded_output = quantized_model(*example_inputs)

        for original, reloaded in zip(original_output, reloaded_output):
            assert torch.equal(original, reloaded)

    def test_stripped_state_dict_has_int8_weights_and_produces_valid_output(
        self,
        quantized_model_factory,
    ):
        _, quantized_model, example_inputs = quantized_model_factory()

        with torch.no_grad():
            original_output = quantized_model(*example_inputs)

        stripped = _strip_redundant_weights(state_dict=quantized_model.state_dict())

        has_int8 = any(t.dtype == torch.int8 for t in stripped.values())
        assert has_int8

        # Verify stripped dict can be loaded and produces same output
        quantized_model.load_state_dict(stripped, strict=False)
        with torch.no_grad():
            stripped_output = quantized_model(*example_inputs)
        assert torch.allclose(original_output[0], stripped_output[0], atol=1e-6)


@pytest.mark.integration
class TestPruningIntegration:
    @pytest.mark.parametrize(
        "pruner_factory",
        [
            lambda: UnstructuredPruner(amount=0.3),
            lambda: StructuredPruner(
                amount=0.25,
                layer_types=[PrunableLayerType.CONV2D.value],
            ),
        ],
        ids=["unstructured", "structured"],
    )
    def test_pruning_then_quantization(
        self,
        synthetic_model_factory,
        example_inputs_factory,
        pruner_factory,
    ):
        model = synthetic_model_factory()
        example_inputs = example_inputs_factory()

        with torch.no_grad():
            float_output = model(*example_inputs)

        pruner = pruner_factory()
        _, zero_parameters = pruner.prune(model)
        assert zero_parameters > 0

        exported = _prepare_and_export(model=model, example_inputs=example_inputs)
        quantized = _quantize_pt2e(exported=exported, example_inputs=example_inputs)

        with torch.no_grad():
            result = quantized(*example_inputs)

        assert result[0].shape == (2, 4)
        assert torch.isfinite(result[0]).all()
        assert result[0].abs().sum() > 0

        # Pruning + quantization together can introduce larger error
        max_diff = (float_output[0] - result[0]).abs().max().item()
        assert max_diff < 5.0


@pytest.mark.integration
class TestPerModuleCompression:
    def test_pruning_only_affects_targeted_module(
        self,
        two_part_model_factory,
        example_inputs_factory,
    ):
        model = two_part_model_factory()
        example_inputs = example_inputs_factory()

        pruner = UnstructuredPruner(amount=0.3)
        pruner.prune(module=model.encoder)

        encoder_zeros = sum((p == 0).sum().item() for p in model.encoder.parameters())
        decoder_zeros = sum((p == 0).sum().item() for p in model.decoder.parameters())
        assert encoder_zeros > 0
        assert decoder_zeros == 0

        with torch.no_grad():
            result = model(*example_inputs)
        assert torch.isfinite(result[0]).all()
        assert result[0].abs().sum() > 0


@pytest.mark.integration
class TestLegacySaveLoadRoundtrip:
    def test_save_and_load_preserves_metadata(
        self,
        tmp_path,
        quantized_model_factory,
    ):
        _, quantized_model, _ = quantized_model_factory()
        observation_keys = ["image"]
        action_keys = ["output_0"]

        save_quantized_model(
            quantized_model=quantized_model,
            save_directory=str(tmp_path),
            observation_keys=observation_keys,
            action_keys=action_keys,
            quantization_config=OmegaConf.create({"is_dynamic": False}),
            training_checkpoint_path="/tmp/training",
        )

        metadata = load_quantization_metadata(
            metadata_path=str(tmp_path / "quantization_metadata.json"),
        )

        assert (
            metadata[QuantizationMetadataKey.OBSERVATION_KEYS.value] == observation_keys
        )
        assert metadata[QuantizationMetadataKey.ACTION_KEYS.value] == action_keys
        assert (
            metadata[QuantizationMetadataKey.TRAINING_CHECKPOINT_PATH.value]
            == "/tmp/training"
        )
        assert QuantizationMetadataKey.WEIGHTS_FILE.value in metadata
        assert QuantizationMetadataKey.TORCHAO_VERSION.value in metadata
        assert QuantizationMetadataKey.TORCH_VERSION.value in metadata

    def test_save_with_custom_filename(
        self,
        tmp_path,
        quantized_model_factory,
    ):
        _, quantized_model, _ = quantized_model_factory()

        save_quantized_model(
            quantized_model=quantized_model,
            save_directory=str(tmp_path),
            observation_keys=["obs"],
            action_keys=["act"],
            quantization_config=OmegaConf.create({"is_dynamic": False}),
            weights_filename="special_weights.pt",
            training_checkpoint_path="/tmp/training",
        )

        assert (tmp_path / "special_weights.pt").exists()
