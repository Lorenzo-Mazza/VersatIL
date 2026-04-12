"""Tests for versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2 module."""

import os
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.constants import RGB_CAMERAS, Cameras
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2 import (
    DFormerEncoder,
    DFormerStage,
    DFormerVariant,
)
from versatil.models.layers.constants import AttentionDecompositionMode

DFORMER_CHECKPOINT_PATH = (
    Path(os.environ.get("VERSATIL_PRETRAINED_DIR", "."))
    / "pretrained_dformer"
    / "DFormerv2_Small_NYU.pth"
)


def _mock_build_backbone(
    self, drop_path_rate, layer_scale_init_value=1e-6, initial_decay=2.0
):
    """Side-effect to set self.stages as empty ModuleList."""
    self.stages = nn.ModuleList()


def _mock_setup_pooling(self, spatial_height: int, spatial_width: int):
    """Side-effect to create a mock pooling head with correct output dim."""
    self.pooling_head = MagicMock()
    self.pooling_head.return_value = torch.zeros(1, self.feature_dim)
    self.output_dim = self.feature_dim


_TINY_VARIANT = {
    "embed_dims": [8, 16],
    "depths": [1, 1],
    "num_heads": [2, 2],
    "decay_ranges": [3, 3],
    "use_layer_scales": [False, True],
}


@pytest.fixture
def dformer_encoder_factory() -> Callable[..., DFormerEncoder]:
    """Factory for DFormerEncoder with mocked backbone.

    By default bypasses ``_build_backbone`` and ``PatchEmbedding`` via
    side-effect mocks for fast shape/spec tests. Pass ``real_build=True``
    to exercise the real ``_build_backbone`` method against a tiny 2-stage
    variant injected into ``VARIANT_CONFIGS``. Exposes the tiny variant
    via ``DFormerVariant`` so construction works end-to-end.
    """

    def factory(
        input_keys: str | list[str] | None = None,
        variant: str = DFormerVariant.SMALL.value,
        decomposition_mode: str = AttentionDecompositionMode.SEPARABLE.value,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        initial_decay: float = 2.0,
        pretrained: bool = False,
        frozen: bool = False,
        checkpoint_path: str | None = None,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        real_build: bool = False,
    ) -> DFormerEncoder:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value, Cameras.DEPTH.value]
        if real_build:
            test_variant = "tiny_test"
            patched_configs = {
                **DFormerEncoder.VARIANT_CONFIGS,
                test_variant: _TINY_VARIANT,
            }
            with patch.dict(DFormerEncoder.VARIANT_CONFIGS, patched_configs):
                return DFormerEncoder(
                    input_keys=input_keys,
                    variant=test_variant,
                    decomposition_mode=decomposition_mode,
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    initial_decay=initial_decay,
                    pretrained=pretrained,
                    frozen=frozen,
                    checkpoint_path=checkpoint_path,
                    pooling_method=pooling_method,
                )
        with (
            patch.object(DFormerEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(DFormerEncoder, "__init_subclass__", lambda **kw: None),
            patch(
                "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.PatchEmbedding",
            ) as mock_patch_embed,
        ):
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
            batch_size,
            height // 2,
            width // 2,
            output_dimension,
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
            rng.standard_normal((batch_size, 1, height, width)).astype(np.float32)
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
                rng.standard_normal((batch_size, 1, height, width)).astype(np.float32)
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
            batch_size,
            downsampled_height,
            downsampled_width,
            output_dimension,
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
        assert next_features.shape == (
            batch_size,
            downsampled_height,
            downsampled_width,
            output_dimension,
        )
        assert output_depth.shape == (
            batch_size,
            1,
            downsampled_height,
            downsampled_width,
        )

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
    @pytest.mark.parametrize(
        "variant, expectation",
        [
            (DFormerVariant.SMALL.value, does_not_raise()),
            (DFormerVariant.BASE.value, does_not_raise()),
            (DFormerVariant.LARGE.value, does_not_raise()),
            ("invalid", pytest.raises(ValueError, match="not supported")),
        ],
    )
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

    @pytest.mark.parametrize(
        "variant",
        [
            DFormerVariant.SMALL.value,
            DFormerVariant.BASE.value,
        ],
    )
    @pytest.mark.parametrize(
        "decomposition_mode",
        [
            AttentionDecompositionMode.SEPARABLE.value,
            AttentionDecompositionMode.FULL.value,
        ],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.AVERAGE.value,
            PoolingMethod.SPATIAL_SOFTMAX.value,
        ],
    )
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
        assert encoder.decomposition_mode == AttentionDecompositionMode(
            decomposition_mode
        )
        assert encoder.pooling_method == pooling_method
        assert (
            encoder.embed_dims == DFormerEncoder.VARIANT_CONFIGS[variant]["embed_dims"]
        )
        assert encoder.feature_dim == encoder.embed_dims[-1]

    def test_has_encoder_interface(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        spec = encoder.get_output_specification()
        feature_keys = [m.key for m in spec]
        assert feature_keys == [EncoderOutputKeys.RGBD.value]

    def test_requires_depth_in_input_keys(self):
        with (
            pytest.raises(
                ValueError,
                match=re.escape("Missing required inputs: {'depth'}"),
            ),
            patch.object(DFormerEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(DFormerEncoder, "_setup_pooling", _mock_setup_pooling),
            patch(
                "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.PatchEmbedding",
            ) as mock_patch_embed,
        ):
            mock_patch_embed.return_value = MagicMock()
            DFormerEncoder(
                input_keys=Cameras.LEFT.value,
                checkpoint_path=None,
            )

    def test_requires_rgb_camera_in_input_keys(self):
        with (
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"Exactly one from {RGB_CAMERAS} required, got {set()}"
                ),
            ),
            patch.object(DFormerEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(DFormerEncoder, "_setup_pooling", _mock_setup_pooling),
            patch(
                "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.PatchEmbedding",
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
    @pytest.mark.parametrize(
        "checkpoint_format, wrapper_key",
        [
            ("model_key", "model"),
            ("state_dict_key", "state_dict"),
            ("plain", None),
        ],
    )
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
            patch(
                "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.torch.load",
                return_value=checkpoint_data,
            ),
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
            patch(
                "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.torch.load",
                return_value=checkpoint_data,
            ),
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
            patch(
                "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.torch.load",
                return_value=checkpoint_data,
            ),
            patch.object(DFormerEncoder, "load_state_dict") as mock_load,
        ):
            encoder._load_checkpoint(checkpoint_path="/fake/path.pth")
            actual_state_dict = mock_load.call_args[0][0]
            assert "encoder.weight" in actual_state_dict
            assert "backbone.encoder.weight" not in actual_state_dict
            assert "classifier.weight" in actual_state_dict
            mock_load.assert_called_once_with(actual_state_dict, strict=False)


class TestDFormerEncoderMixin:
    def test_camera_group_includes_rgb_and_depth(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        assert Cameras.LEFT.value in encoder._camera_group
        assert Cameras.DEPTH.value in encoder._camera_group

    def test_output_modality_is_rgbd(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        assert encoder._output_modality == EncoderOutputKeys.RGBD.value

    def test_encode_single_image_raises(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "DFormerEncoder processes RGB+depth jointly. Use encode() instead."
            ),
        ):
            encoder._encode_single_image(torch.zeros(1, 3, 32, 32))


class TestDFormerEncoderGetOutputSpecification:
    def test_returns_rgbd_feature_with_correct_dimension(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGBD.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.RGBD.value
        ).dimension == (encoder.output_dim,)


class TestDFormerEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "key, metadata, expected_error",
        [
            (
                Cameras.LEFT.value,
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                Cameras.LEFT.value,
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                f"Expected 3-channel RGB for '{Cameras.LEFT.value}', got 1 channels",
            ),
            (
                Cameras.DEPTH.value,
                CameraMetadata(
                    camera_key="depth",
                    dtype="float32",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                Cameras.DEPTH.value,
                CameraMetadata(
                    camera_key="depth",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                f"Expected single-channel depth for '{Cameras.DEPTH.value}', got 3 channels",
            ),
            (
                Cameras.LEFT.value,
                MagicMock(spec=BaseMetadata),
                f"Expected CameraMetadata for '{Cameras.LEFT.value}', got MagicMock",
            ),
        ],
    )
    def test_validates_rgb_and_depth_metadata(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        key: str,
        metadata,
        expected_error: str | None,
    ):
        encoder = dformer_encoder_factory()
        result = encoder.validate_input_metadata(key=key, metadata=metadata)
        assert result == expected_error


class TestDFormerEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "variant",
        [
            DFormerVariant.SMALL.value,
        ],
    )
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
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 1
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
            checkpoint_path=None,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)


@pytest.mark.integration
class TestDFormerEncoderPretrainedCheckpoint:
    @pytest.mark.skipif(
        not DFORMER_CHECKPOINT_PATH.exists(),
        reason=f"DFormer checkpoint not found at {DFORMER_CHECKPOINT_PATH}",
    )
    def test_loads_pretrained_weights_and_produces_output(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=True,
            checkpoint_path=str(DFORMER_CHECKPOINT_PATH),
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        encoder.eval()
        batch_size = 1
        inputs = rgbd_input_factory(batch_size=batch_size, height=224, width=224)

        with torch.no_grad():
            output = encoder(inputs)

        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)
        assert features.std() > 1e-6

    @pytest.mark.skipif(
        not DFORMER_CHECKPOINT_PATH.exists(),
        reason=f"DFormer checkpoint not found at {DFORMER_CHECKPOINT_PATH}",
    )
    def test_pretrained_features_differ_from_random_init(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        pretrained_encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=True,
            checkpoint_path=str(DFORMER_CHECKPOINT_PATH),
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        random_encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
            checkpoint_path=None,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        pretrained_encoder.set_image_size(image_height=224, image_width=224)
        random_encoder.set_image_size(image_height=224, image_width=224)
        pretrained_encoder.eval()
        random_encoder.eval()
        inputs = rgbd_input_factory(batch_size=1, height=224, width=224)

        with torch.no_grad():
            pretrained_features = pretrained_encoder(inputs)[
                EncoderOutputKeys.RGBD.value
            ]
            random_features = random_encoder(inputs)[EncoderOutputKeys.RGBD.value]

        assert not torch.allclose(pretrained_features, random_features, atol=1e-3)


class TestDFormerEncoderRealBuild:
    @pytest.mark.unit
    def test_build_backbone_instantiates_real_stages(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory(real_build=True)
        assert len(encoder.stages) == 2
        assert encoder.embed_dims == _TINY_VARIANT["embed_dims"]
        assert encoder.feature_dim == _TINY_VARIANT["embed_dims"][-1]
        # First stage has a downsample (patch merging), last doesn't
        assert encoder.stages[0].downsample is not None
        assert encoder.stages[1].downsample is None

    @pytest.mark.unit
    def test_set_image_size_creates_pooling_head(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        encoder = dformer_encoder_factory(real_build=True)
        assert encoder.pooling_head is None
        encoder.set_image_size(image_height=32, image_width=32)
        assert encoder.pooling_head is not None
        assert encoder.output_dim == encoder.pooling_head.output_dim

    @pytest.mark.unit
    def test_encode_returns_rgbd_features_end_to_end(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = dformer_encoder_factory(real_build=True)
        encoder.set_image_size(image_height=32, image_width=32)
        encoder.eval()
        inputs = rgbd_input_factory(batch_size=1, height=32, width=32)
        with torch.no_grad():
            output = encoder(inputs)
        assert EncoderOutputKeys.RGBD.value in output
        assert output[EncoderOutputKeys.RGBD.value].shape[0] == 1

    @pytest.mark.unit
    def test_encode_raises_when_pooling_head_not_initialized(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = dformer_encoder_factory(real_build=True)
        inputs = rgbd_input_factory(batch_size=1, height=32, width=32)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "pooling_head is not initialized. Call set_image_size() before forward."
            ),
        ):
            encoder(inputs)

    @pytest.mark.unit
    def test_load_checkpoint_handles_model_and_state_dict_keys(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
        tmp_path,
    ):
        encoder = dformer_encoder_factory(real_build=True)
        # Synthesize a checkpoint in each of the three supported top-level
        # shapes: raw state_dict, {"model": state_dict}, {"state_dict": ...}
        raw_state = {
            f"backbone.{key}": value.clone()
            for key, value in encoder.state_dict().items()
        }
        checkpoint_path = tmp_path / "dformer.pth"
        torch.save({"model": raw_state}, checkpoint_path)
        # _load_checkpoint strips "backbone." prefix and does a non-strict load
        encoder._load_checkpoint(str(checkpoint_path))

        torch.save({"state_dict": raw_state}, checkpoint_path)
        encoder._load_checkpoint(str(checkpoint_path))

        torch.save(raw_state, checkpoint_path)
        encoder._load_checkpoint(str(checkpoint_path))

    @pytest.mark.unit
    def test_pretrained_requires_checkpoint_path(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Pretrained=True requires a valid checkpoint_path for DFormerEncoder."
            ),
        ):
            dformer_encoder_factory(real_build=True, pretrained=True)

    @pytest.mark.unit
    def test_frozen_calls_freeze_weights_on_init(
        self,
        dformer_encoder_factory: Callable[..., DFormerEncoder],
    ):
        with patch.object(EncodingMixin, "_freeze_weights") as mock_freeze:
            dformer_encoder_factory(real_build=True, frozen=True)
        mock_freeze.assert_called_once()
