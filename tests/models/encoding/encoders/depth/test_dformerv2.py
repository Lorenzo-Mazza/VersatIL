"""Tests for versatil.models.encoding.encoders.depth.dformerv2 module."""
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras, RGB_CAMERAS
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.depth.dformerv2 import (
    DFormerEncoder,
    DFormerStage,
    DFormerVariant,
)
from versatil.models.layers.constants import AttentionDecompositionMode


def _mock_build_backbone(self, drop_path_rate, layer_scale_init_value=1e-6, initial_decay=2.0):
    """Side-effect to set self.stages as empty ModuleList."""
    self.stages = nn.ModuleList()


def _mock_setup_pooling(self):
    """Side-effect to set pooling-related attributes."""
    self.pooling_head = None
    self.output_dim = self.feature_dim


@pytest.fixture
def dformer_encoder_factory() -> Callable[..., DFormerEncoder]:
    """Factory for DFormerEncoder with mocked backbone and pooling."""
    def factory(
        input_keys: str | list[str] = [Cameras.LEFT.value, Cameras.DEPTH.value],
        variant: str = DFormerVariant.SMALL.value,
        decomposition_mode: str = AttentionDecompositionMode.SEPARABLE.value,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        initial_decay: float = 2.0,
        pretrained: bool = False,
        frozen: bool = False,
        checkpoint_path: str | None = None,
        pooling_method: str = PoolingMethod.AVERAGE.value,
    ) -> DFormerEncoder:
        with (
            patch.object(DFormerEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(DFormerEncoder, "_setup_pooling", _mock_setup_pooling),
            patch.object(DFormerEncoder, "__init_subclass__", lambda **kw: None),
        ):
            # Also mock PatchEmbedding since it is created in __init__ before _build_backbone
            with patch(
                "versatil.models.encoding.encoders.depth.dformerv2.PatchEmbedding",
            ) as mock_patch_embed:
                mock_patch_embed.return_value = MagicMock()
                return DFormerEncoder(
                    input_keys=input_keys,
                    variant=variant,
                    decomposition_mode=decomposition_mode,
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    initial_decay=initial_decay,
                    pretrained=pretrained,
                    frozen=frozen,
                    checkpoint_path=checkpoint_path,
                    pooling_method=pooling_method,
                )
    return factory


@pytest.fixture
def dformer_stage_factory() -> Callable[..., DFormerStage]:
    """Factory for DFormerStage with small dimensions for unit testing."""
    def factory(
        embedding_dimension: int = 16,
        num_heads: int = 2,
        num_blocks: int = 1,
        decomposition_mode: AttentionDecompositionMode = AttentionDecompositionMode.SEPARABLE,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        ffn_expansion_factor: int = 4,
        downsample: nn.Module | None = None,
    ) -> DFormerStage:
        return DFormerStage(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            num_blocks=num_blocks,
            decomposition_mode=decomposition_mode,
            drop_path_rate=drop_path_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            initial_decay=initial_decay,
            decay_range=decay_range,
            ffn_expansion_factor=ffn_expansion_factor,
            downsample=downsample,
        )
    return factory


class TestDFormerStageInitialization:

    @pytest.mark.parametrize("embedding_dimension", [16, 32])
    @pytest.mark.parametrize("num_blocks", [1, 3])
    def test_stores_configuration(
        self,
        dformer_stage_factory: Callable[..., DFormerStage],
        embedding_dimension: int,
        num_blocks: int,
    ):
        stage = dformer_stage_factory(
            embedding_dimension=embedding_dimension,
            num_blocks=num_blocks,
        )
        assert stage.embedding_dimension == embedding_dimension
        assert len(stage.blocks) == num_blocks
        assert stage.downsample is None
        assert stage.norm.normalized_shape == (embedding_dimension,)

    def test_downsample_module_is_used_in_forward(
        self,
        dformer_stage_factory: Callable[..., DFormerStage],
        rng: np.random.Generator,
    ):
        embedding_dimension = 16
        output_dimension = 32
        batch_size = 2
        height = 8
        width = 8
        mock_downsample = MagicMock(spec=nn.Module)
        mock_downsample.return_value = torch.zeros(
            batch_size, height // 2, width // 2, output_dimension,
        )
        stage = dformer_stage_factory(
            embedding_dimension=embedding_dimension,
            downsample=mock_downsample,
        )
        rgb_features = torch.from_numpy(
            rng.standard_normal(
                (batch_size, height, width, embedding_dimension)
            ).astype(np.float32)
        )
        depth_map = torch.from_numpy(
            rng.standard_normal(
                (batch_size, 1, height, width)
            ).astype(np.float32)
        )
        stage(rgb_features=rgb_features, depth_map=depth_map)
        mock_downsample.assert_called_once()


class TestDFormerStageForward:

    @pytest.fixture
    def stage_input_factory(
        self,
        rng: np.random.Generator,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """Factory for DFormerStage input tensors."""
        def factory(
            batch_size: int = 2,
            height: int = 8,
            width: int = 8,
            embedding_dimension: int = 16,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rgb_features = torch.from_numpy(
                rng.standard_normal(
                    (batch_size, height, width, embedding_dimension)
                ).astype(np.float32)
            )
            depth_map = torch.from_numpy(
                rng.standard_normal(
                    (batch_size, 1, height, width)
                ).astype(np.float32)
            )
            return rgb_features, depth_map
        return factory

    def test_forward_without_downsample_preserves_shape(
        self,
        dformer_stage_factory: Callable[..., DFormerStage],
        stage_input_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        embedding_dimension = 16
        height = 8
        width = 8
        batch_size = 2
        stage = dformer_stage_factory(
            embedding_dimension=embedding_dimension,
            downsample=None,
        )
        rgb_features, depth_map = stage_input_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            embedding_dimension=embedding_dimension,
        )
        output_features, next_features, output_depth = stage(
            rgb_features=rgb_features,
            depth_map=depth_map,
        )
        assert output_features.shape == (batch_size, height, width, embedding_dimension)
        assert next_features.shape == output_features.shape
        assert output_depth.shape == depth_map.shape

    def test_forward_with_downsample_reduces_spatial(
        self,
        dformer_stage_factory: Callable[..., DFormerStage],
        stage_input_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        embedding_dimension = 16
        output_dimension = 32
        height = 8
        width = 8
        batch_size = 2
        mock_downsample = MagicMock(spec=nn.Module)
        downsampled_height = height // 2
        downsampled_width = width // 2
        mock_downsample.return_value = torch.zeros(
            batch_size, downsampled_height, downsampled_width, output_dimension,
        )
        stage = dformer_stage_factory(
            embedding_dimension=embedding_dimension,
            downsample=mock_downsample,
        )
        rgb_features, depth_map = stage_input_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            embedding_dimension=embedding_dimension,
        )
        output_features, next_features, output_depth = stage(
            rgb_features=rgb_features,
            depth_map=depth_map,
        )
        assert output_features.shape == (batch_size, height, width, embedding_dimension)
        assert next_features.shape == (batch_size, downsampled_height, downsampled_width, output_dimension)
        assert output_depth.shape == (batch_size, 1, downsampled_height, downsampled_width)

    def test_returns_three_tensors(
        self,
        dformer_stage_factory: Callable[..., DFormerStage],
        stage_input_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        stage = dformer_stage_factory()
        rgb_features, depth_map = stage_input_factory()
        result = stage(rgb_features=rgb_features, depth_map=depth_map)
        assert len(result) == 3


class TestDFormerEncoderInitialization:

    @pytest.mark.parametrize("variant, expectation", [
        (DFormerVariant.SMALL.value, does_not_raise()),
        (DFormerVariant.BASE.value, does_not_raise()),
        (DFormerVariant.LARGE.value, does_not_raise()),
        ("invalid", pytest.raises(ValueError, match="not supported")),
    ])
    def test_variant_validation(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        variant: str,
        expectation,
    ):
        with expectation:
            dformer_encoder_factory(variant=variant)

    def test_pretrained_without_checkpoint_raises(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        with pytest.raises(ValueError, match="checkpoint_path"):
            dformer_encoder_factory(pretrained=True, checkpoint_path=None)

    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
    ])
    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.SEPARABLE.value,
        AttentionDecompositionMode.FULL.value,
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.AVERAGE.value,
        PoolingMethod.SPATIAL_SOFTMAX.value,
    ])
    def test_stores_configuration(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        variant: str,
        decomposition_mode: str,
        pooling_method: str,
    ):
        encoder = dformer_encoder_factory(
            variant=variant,
            decomposition_mode=decomposition_mode,
            pooling_method=pooling_method,
        )
        assert encoder.variant == variant
        assert encoder.decomposition_mode == AttentionDecompositionMode(decomposition_mode)
        assert encoder.pooling_method == pooling_method
        assert encoder.embed_dims == DFormerEncoder.VARIANT_CONFIGS[variant]["embed_dims"]
        assert encoder.feature_dim == encoder.embed_dims[-1]

    def test_has_encoder_interface(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        assert hasattr(encoder, "forward")
        assert hasattr(encoder, "get_output_specification")
        assert hasattr(encoder, "input_specification")

    def test_requires_depth_in_input_keys(self):
        with pytest.raises(ValueError, match="Missing required inputs"):
            with (
                patch.object(DFormerEncoder, "_build_backbone", _mock_build_backbone),
                patch.object(DFormerEncoder, "_setup_pooling", _mock_setup_pooling),
                patch(
                    "versatil.models.encoding.encoders.depth.dformerv2.PatchEmbedding",
                ) as mock_patch_embed,
            ):
                mock_patch_embed.return_value = MagicMock()
                DFormerEncoder(
                    input_keys=Cameras.LEFT.value,
                    checkpoint_path=None,
                )

    def test_requires_rgb_camera_in_input_keys(self):
        with pytest.raises(ValueError, match="Exactly one from"):
            with (
                patch.object(DFormerEncoder, "_build_backbone", _mock_build_backbone),
                patch.object(DFormerEncoder, "_setup_pooling", _mock_setup_pooling),
                patch(
                    "versatil.models.encoding.encoders.depth.dformerv2.PatchEmbedding",
                ) as mock_patch_embed,
            ):
                mock_patch_embed.return_value = MagicMock()
                DFormerEncoder(
                    input_keys=Cameras.DEPTH.value,
                    checkpoint_path=None,
                )

    def test_input_specification_requires_depth_camera(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        assert Cameras.DEPTH.value in encoder.input_specification.required

    def test_input_specification_requires_one_rgb_camera(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        assert encoder.input_specification.one_of_groups == [RGB_CAMERAS]


class TestDFormerEncoderLoadCheckpoint:

    @pytest.mark.parametrize("checkpoint_format, wrapper_key", [
        ("model_key", "model"),
        ("state_dict_key", "state_dict"),
        ("plain", None),
    ])
    def test_unwraps_checkpoint_format(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        checkpoint_format: str,
        wrapper_key: str | None,
    ):
        encoder = dformer_encoder_factory()
        raw_weights = {"layer.weight": torch.tensor([1.0])}
        if wrapper_key is not None:
            checkpoint_data = {wrapper_key: raw_weights}
        else:
            checkpoint_data = raw_weights

        with (
            patch("versatil.models.encoding.encoders.depth.dformerv2.torch.load", return_value=checkpoint_data),
            patch.object(DFormerEncoder, "load_state_dict") as mock_load,
        ):
            encoder._load_checkpoint(checkpoint_path="/fake/path.pth")
            mock_load.assert_called_once_with(raw_weights, strict=False)

    def test_strips_backbone_prefix_from_keys(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        checkpoint_data = {
            "backbone.layer1.weight": torch.tensor([1.0]),
            "backbone.layer2.bias": torch.tensor([2.0]),
            "head.fc.weight": torch.tensor([3.0]),
        }
        expected_cleaned = {
            "layer1.weight": torch.tensor([1.0]),
            "layer2.bias": torch.tensor([2.0]),
            "head.fc.weight": torch.tensor([3.0]),
        }

        with (
            patch("versatil.models.encoding.encoders.depth.dformerv2.torch.load", return_value=checkpoint_data),
            patch.object(DFormerEncoder, "load_state_dict") as mock_load,
        ):
            encoder._load_checkpoint(checkpoint_path="/fake/path.pth")
            mock_load.assert_called_once()
            actual_state_dict = mock_load.call_args[0][0]
            assert set(actual_state_dict.keys()) == set(expected_cleaned.keys())
            for key in expected_cleaned:
                assert torch.equal(actual_state_dict[key], expected_cleaned[key])

    def test_strips_backbone_prefix_inside_wrapper(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        checkpoint_data = {
            "model": {
                "backbone.encoder.weight": torch.tensor([1.0]),
                "classifier.weight": torch.tensor([2.0]),
            }
        }

        with (
            patch("versatil.models.encoding.encoders.depth.dformerv2.torch.load", return_value=checkpoint_data),
            patch.object(DFormerEncoder, "load_state_dict") as mock_load,
        ):
            encoder._load_checkpoint(checkpoint_path="/fake/path.pth")
            actual_state_dict = mock_load.call_args[0][0]
            assert "encoder.weight" in actual_state_dict
            assert "backbone.encoder.weight" not in actual_state_dict
            assert "classifier.weight" in actual_state_dict
            mock_load.assert_called_once_with(actual_state_dict, strict=False)


class TestDFormerEncoderGetOutputSpecification:

    def test_returns_rgbd_feature_with_correct_dimension(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        specification = encoder.get_output_specification()
        assert specification.features == [EncoderOutputKeys.RGBD.value]
        assert specification.dimensions[EncoderOutputKeys.RGBD.value] == encoder.output_dim


class TestDFormerEncoderIntegration:

    @pytest.mark.integration
    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
    ])
    def test_forward_pass(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        variant: str,
    ):
        batch_size = 1
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pretrained=False,
            checkpoint_path=None,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        inputs = rgbd_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.ndim == 2
        assert features.shape[0] == batch_size

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [None, 2])
    def test_temporal_reshaping(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
    ):
        batch_size = 1
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
            checkpoint_path=None,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        if time_steps is not None:
            assert features.shape == (batch_size, time_steps, encoder.output_dim)
        else:
            assert features.shape == (batch_size, encoder.output_dim)
