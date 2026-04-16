"""Tests for versatil.models.encoding.encoders.rgb.flat module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    FlatBackboneType,
    PoolingMethod,
    SpatialBackboneType,
)
from versatil.models.encoding.encoders.rgb.flat import FlatRGBEncoder

FLAT_BACKBONES = list(FlatBackboneType)
FLAT_VALID_BACKBONES = [e.value for e in FlatBackboneType]

FEATURE_DIM = 768
SEQUENCE_LENGTH = 196


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.num_features = FEATURE_DIM
    self.expected_image_size = None
    self.requires_strict_image_size = False
    self.patch_size = None


def _make_mock_timm_backbone(
    num_features: int = FEATURE_DIM,
    num_prefix_tokens: int = 1,
    patch_embed_img_size: tuple[int, int] | None = None,
    patch_size: tuple[int, int] = (14, 14),
    strict_img_size: bool = False,
) -> MagicMock:
    backbone = MagicMock()
    backbone.num_features = num_features
    backbone.num_prefix_tokens = num_prefix_tokens
    if patch_embed_img_size is None:
        backbone.patch_embed = None
    else:
        patch_embed = MagicMock()
        patch_embed.strict_img_size = strict_img_size
        patch_embed.img_size = patch_embed_img_size
        patch_embed.patch_size = patch_size
        backbone.patch_embed = patch_embed
    return backbone


@pytest.fixture
def mock_timm_backend():
    """Patches timm.get_pretrained_cfg and timm.create_model for the whole test.

    Exposes ``configure(...)`` to set the mock pretrained config and the
    per-call backbone template (used to synthesize a fresh mock backbone on
    every ``timm.create_model`` call, so rebuilds via ``set_image_size`` see
    a new instance). The ``create_model`` mock is reachable at
    ``backend.create_model_mock`` for spies / assertions.
    """

    class _Backend:
        def __init__(self):
            self.cfg = MagicMock()
            self.cfg.fixed_input_size = False
            self.cfg.input_size = (3, 518, 518)
            self.backbone_kwargs = {
                "num_features": FEATURE_DIM,
                "num_prefix_tokens": 1,
                "patch_embed_img_size": (224, 224),
                "patch_size": (14, 14),
                "strict_img_size": False,
            }

        def configure(
            self,
            fixed_input_size: bool | None = None,
            pretrained_input_size: tuple[int, int, int] | None = None,
            num_prefix_tokens: int | None = None,
            patch_embed_img_size: tuple[int, int] | None = ...,
            patch_size: tuple[int, int] | None = None,
            strict_img_size: bool | None = None,
        ) -> None:
            if fixed_input_size is not None:
                self.cfg.fixed_input_size = fixed_input_size
            if pretrained_input_size is not None:
                self.cfg.input_size = pretrained_input_size
            if num_prefix_tokens is not None:
                self.backbone_kwargs["num_prefix_tokens"] = num_prefix_tokens
            if patch_embed_img_size is not ...:
                self.backbone_kwargs["patch_embed_img_size"] = patch_embed_img_size
            if patch_size is not None:
                self.backbone_kwargs["patch_size"] = patch_size
            if strict_img_size is not None:
                self.backbone_kwargs["strict_img_size"] = strict_img_size

        def _side_effect(self, *args, **kwargs) -> MagicMock:
            kwargs_override = dict(self.backbone_kwargs)
            img_size_kwarg = kwargs.get("img_size")
            if isinstance(img_size_kwarg, tuple):
                kwargs_override["patch_embed_img_size"] = img_size_kwarg
            elif isinstance(img_size_kwarg, int):
                kwargs_override["patch_embed_img_size"] = (
                    img_size_kwarg,
                    img_size_kwarg,
                )
            return _make_mock_timm_backbone(**kwargs_override)

    backend = _Backend()
    cfg_patcher = patch(
        "versatil.models.encoding.encoders.rgb.flat.timm.get_pretrained_cfg",
        return_value=backend.cfg,
    )
    model_patcher = patch(
        "versatil.models.encoding.encoders.rgb.flat.timm.create_model",
        side_effect=backend._side_effect,
    )
    cfg_patcher.start()
    backend.create_model_mock = model_patcher.start()
    yield backend
    cfg_patcher.stop()
    model_patcher.stop()


@pytest.fixture
def flat_rgb_encoder_factory(
    mock_timm_backend,
) -> Callable[..., FlatRGBEncoder]:
    """Factory for FlatRGBEncoder with mocked backbone.

    By default bypasses ``_build_backbone`` via a side-effect mock for
    fast shape/forward tests. Pass ``real_build=True`` to exercise the
    real ``_build_backbone`` method — the ``mock_timm_backend`` fixture
    keeps the timm patches active for the whole test, so subsequent
    ``set_image_size`` calls also hit the mocks. Configure the backend
    via ``mock_timm_backend.configure(...)`` before calling the factory.
    """

    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = FlatBackboneType.DINOV2_VITB14.value,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        pretrained: bool = False,
        frozen: bool = False,
        real_build: bool = False,
    ) -> FlatRGBEncoder:
        if not real_build:
            with patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone):
                return FlatRGBEncoder(
                    input_keys=input_keys,
                    backbone=backbone,
                    pooling_method=pooling_method,
                    pretrained=pretrained,
                    frozen=frozen,
                )
        return FlatRGBEncoder(
            input_keys=input_keys,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=pretrained,
            frozen=frozen,
        )

    return factory


@pytest.fixture
def mock_backbone_output_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for mock backbone output tensor (last_hidden_state)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = SEQUENCE_LENGTH + 1,
        feature_dim: int = FEATURE_DIM,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, feature_dim)).astype(
                np.float32
            )
        )

    return factory


class TestFlatRGBEncoderInitialization:
    @pytest.mark.parametrize(
        "backbone, expectation",
        [
            (FlatBackboneType.DINOV2_VITB14.value, does_not_raise()),
            (FlatBackboneType.VIT_BASE.value, does_not_raise()),
            (FlatBackboneType.DINOV2_VITS14.value, does_not_raise()),
            (
                SpatialBackboneType.RESNET18.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone '{SpatialBackboneType.RESNET18.value}'. "
                        f"Must be one of: {FLAT_VALID_BACKBONES}"
                    ),
                ),
            ),
            (
                SpatialBackboneType.SWIN_TINY.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone '{SpatialBackboneType.SWIN_TINY.value}'. "
                        f"Must be one of: {FLAT_VALID_BACKBONES}"
                    ),
                ),
            ),
            (
                "invalid_backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone 'invalid_backbone'. "
                        f"Must be one of: {FLAT_VALID_BACKBONES}"
                    ),
                ),
            ),
        ],
    )
    def test_backbone_validation(
        self,
        backbone: str,
        expectation,
    ):
        with (
            expectation,
            patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone),
        ):
            FlatRGBEncoder(
                input_keys="left",
                pretrained=False,
                frozen=False,
                pooling_method=PoolingMethod.DEFAULT.value,
                backbone=backbone,
            )

    @pytest.mark.parametrize(
        "input_keys, expectation",
        [
            ("left", does_not_raise()),
            ("right", does_not_raise()),
            (["left", "right"], does_not_raise()),
            (
                "invalid_camera",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"At least one from {RGB_CAMERAS} required, got {set()}"
                    ),
                ),
            ),
        ],
    )
    def test_input_keys_validation(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            flat_rgb_encoder_factory(input_keys=input_keys)

    @pytest.mark.parametrize(
        "pooling_method, expectation",
        [
            (PoolingMethod.DEFAULT.value, does_not_raise()),
            (PoolingMethod.NONE.value, does_not_raise()),
            (PoolingMethod.AVERAGE.value, does_not_raise()),
            (PoolingMethod.LEARNED_AGGREGATION.value, does_not_raise()),
            (
                PoolingMethod.SPATIAL_SOFTMAX.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Pooling method '{PoolingMethod.SPATIAL_SOFTMAX.value}' "
                        f"is not compatible with token sequences. Use one of: "
                        f"{[p.value for p in PoolingMethod if p.supports_sequential]}"
                    ),
                ),
            ),
            (
                PoolingMethod.MAX.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Pooling method '{PoolingMethod.MAX.value}' "
                        f"is not compatible with token sequences. Use one of: "
                        f"{[p.value for p in PoolingMethod if p.supports_sequential]}"
                    ),
                ),
            ),
        ],
    )
    def test_pooling_method_sequential_validation(
        self,
        pooling_method: str,
        expectation,
    ):
        with (
            expectation,
            patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone),
        ):
            FlatRGBEncoder(
                input_keys="left",
                pretrained=False,
                frozen=False,
                pooling_method=pooling_method,
                backbone=FlatBackboneType.DINOV2_VITB14.value,
            )

    @pytest.mark.parametrize("input_keys", ["left", "right"])
    @pytest.mark.parametrize(
        "backbone",
        [
            FlatBackboneType.DINOV2_VITS14.value,
            FlatBackboneType.DINOV2_VITB14.value,
        ],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.DEFAULT.value,
            PoolingMethod.NONE.value,
        ],
    )
    def test_stores_configuration(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        input_keys: str | list[str],
        backbone: str,
        pooling_method: str,
    ):
        encoder = flat_rgb_encoder_factory(
            input_keys=input_keys,
            backbone=backbone,
            pooling_method=pooling_method,
        )
        expected_keys = [input_keys] if isinstance(input_keys, str) else input_keys
        assert encoder.backbone_name == backbone
        assert encoder.pooling_method == pooling_method
        assert encoder.feature_dim == FEATURE_DIM
        assert encoder.input_specification.keys == expected_keys

    def test_none_pooling_sets_output_dim_to_tuple(self):
        with patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone):
            encoder = FlatRGBEncoder(
                input_keys="left",
                backbone=FlatBackboneType.DINOV2_VITB14.value,
                pooling_method=PoolingMethod.NONE.value,
                pretrained=False,
                frozen=False,
            )
        assert encoder.output_dim == (-1, FEATURE_DIM)

    def test_non_none_pooling_sets_output_dim_to_int(self):
        with patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone):
            encoder = FlatRGBEncoder(
                input_keys="left",
                backbone=FlatBackboneType.DINOV2_VITB14.value,
                pooling_method=PoolingMethod.DEFAULT.value,
                pretrained=False,
                frozen=False,
            )
        assert encoder.output_dim == FEATURE_DIM


class TestFlatRGBEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
        time_steps: int,
    ):
        batch_size = 2
        encoder = flat_rgb_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        effective_batch = batch_size * time_steps
        backbone_output = mock_backbone_output_factory(batch_size=effective_batch)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, FEATURE_DIM)

    def test_none_pooling_output_shape_with_time(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        time_steps = 3
        encoder = flat_rgb_encoder_factory(pooling_method=PoolingMethod.NONE.value)
        effective_batch = batch_size * time_steps
        backbone_output = mock_backbone_output_factory(batch_size=effective_batch)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, SEQUENCE_LENGTH, FEATURE_DIM)


class TestFlatRGBEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
    ):
        encoder = flat_rgb_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGB.value]
        output_dim = encoder.output_dim
        expected_dim = output_dim if isinstance(output_dim, tuple) else (output_dim,)
        assert (
            next(
                m for m in specification if m.key == EncoderOutputKeys.RGB.value
            ).dimension
            == expected_dim
        )


class TestFlatRGBEncoderBuildBackbone:
    @pytest.mark.integration
    def test_fixed_input_size_models_have_strict_image_size(self):
        encoder = FlatRGBEncoder(
            input_keys="left",
            backbone=FlatBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
            frozen=False,
        )
        assert encoder.requires_strict_image_size
        assert encoder.expected_image_size == (518, 518)

    @pytest.mark.integration
    def test_set_image_size_rebuilds_backbone(self):
        encoder = FlatRGBEncoder(
            input_keys="left",
            backbone=FlatBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
        original_size = encoder.expected_image_size
        encoder.set_image_size(image_height=256, image_width=256)
        assert encoder.expected_image_size == (256, 256)
        assert encoder.expected_image_size != original_size

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixed_input_size",
        [False, True],
        ids=["dynamic", "fixed"],
    )
    def test_build_backbone_dispatches_on_fixed_input_size(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        mock_timm_backend,
        fixed_input_size: bool,
    ):
        mock_timm_backend.configure(fixed_input_size=fixed_input_size)
        flat_rgb_encoder_factory(real_build=True)
        call_kwargs = mock_timm_backend.create_model_mock.call_args.kwargs
        if fixed_input_size:
            # initial build has self.image_size=None → uses pretrained input_size[-1]
            assert call_kwargs["img_size"] == 518
        else:
            assert "img_size" not in call_kwargs

    @pytest.mark.unit
    def test_build_backbone_reads_patch_embed_properties(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        mock_timm_backend,
    ):
        mock_timm_backend.configure(
            patch_embed_img_size=(224, 224),
            patch_size=(16, 16),
            strict_img_size=True,
        )
        encoder = flat_rgb_encoder_factory(real_build=True)
        assert encoder.requires_strict_image_size is True
        assert encoder.expected_image_size == (224, 224)
        assert encoder.patch_size == (16, 16)

    @pytest.mark.unit
    def test_build_backbone_without_patch_embed_is_permissive(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        mock_timm_backend,
    ):
        mock_timm_backend.configure(patch_embed_img_size=None)
        encoder = flat_rgb_encoder_factory(real_build=True)
        assert encoder.requires_strict_image_size is False
        assert encoder.expected_image_size is None
        assert encoder.patch_size is None

    @pytest.mark.unit
    def test_set_image_size_rebuilds_and_updates_feature_dim(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        mock_timm_backend,
    ):
        mock_timm_backend.configure(fixed_input_size=True)
        encoder = flat_rgb_encoder_factory(real_build=True)
        initial_calls = mock_timm_backend.create_model_mock.call_count
        encoder.set_image_size(image_height=384, image_width=384)
        assert mock_timm_backend.create_model_mock.call_count == initial_calls + 1
        assert mock_timm_backend.create_model_mock.call_args.kwargs["img_size"] == (
            384,
            384,
        )
        assert encoder.feature_dim == FEATURE_DIM

    @pytest.mark.unit
    def test_set_image_size_refreezes_when_frozen(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
    ):
        with patch.object(EncodingMixin, "_freeze_weights") as mock_freeze:
            encoder = flat_rgb_encoder_factory(
                real_build=True,
                frozen=True,
            )
            calls_after_init = mock_freeze.call_count
            encoder.set_image_size(image_height=256, image_width=256)
        assert mock_freeze.call_count == calls_after_init + 1

    @pytest.mark.unit
    def test_encode_resizes_when_expected_image_size_set(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        encoder = flat_rgb_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        encoder.expected_image_size = (64, 64)
        encoder.backbone.forward_features.return_value = mock_backbone_output_factory(
            batch_size=2
        )
        inputs = image_input_factory(key="left", batch_size=2, height=32, width=32)
        with patch(
            "versatil.models.encoding.encoders.rgb.flat.resize_to_target_size",
            wraps=lambda images, target_height, target_width: images.new_zeros(
                images.shape[0], 3, target_height, target_width
            ),
        ) as mock_resize:
            encoder(inputs)
        mock_resize.assert_called_once()
        assert mock_resize.call_args.kwargs["target_height"] == 64
        assert mock_resize.call_args.kwargs["target_width"] == 64


class TestFlatRGBEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
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
                CameraMetadata(
                    camera_key="depth",
                    dtype="uint8",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                MagicMock(spec=BaseMetadata),
                "Expected CameraMetadata for 'left', got MagicMock",
            ),
        ],
    )
    def test_validates_camera_metadata(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = flat_rgb_encoder_factory()
        result = encoder.validate_input_metadata(key="left", metadata=metadata)
        assert result == expected_error


class TestFlatRGBEncoderMultiCamera:
    @pytest.mark.parametrize(
        "input_keys, expected_feature_count, expected_multi_camera",
        [
            ("left", 1, False),
            (["left", "right"], 2, True),
        ],
    )
    def test_output_specification_scales_with_cameras(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        input_keys: str | list[str],
        expected_feature_count: int,
        expected_multi_camera: bool,
    ):
        encoder = flat_rgb_encoder_factory(input_keys=input_keys)
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert len(feature_keys) == expected_feature_count
        assert encoder.is_multi_camera is expected_multi_camera
        if expected_multi_camera:
            camera_list = input_keys if isinstance(input_keys, list) else [input_keys]
            for camera_key in camera_list:
                feature_name = f"{EncoderOutputKeys.RGB.value}:{camera_key}"
                assert feature_name in feature_keys
        else:
            assert feature_keys == [EncoderOutputKeys.RGB.value]

    def test_multi_camera_forward_produces_per_camera_features(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        encoder = flat_rgb_encoder_factory(
            input_keys=["left", "right"],
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        backbone_output = mock_backbone_output_factory(batch_size=batch_size)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        output = encoder(inputs)
        rgb = EncoderOutputKeys.RGB.value
        assert f"{rgb}:left" in output
        assert f"{rgb}:right" in output

    def test_multi_camera_backbone_called_per_camera(
        self,
        flat_rgb_encoder_factory: Callable[..., FlatRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        encoder = flat_rgb_encoder_factory(
            input_keys=["left", "right"],
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        backbone_output = mock_backbone_output_factory(batch_size=batch_size)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        encoder(inputs)
        assert encoder.backbone.forward_features.call_count == 2


class TestFlatRGBEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in FLAT_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = FlatRGBEncoder(
            input_keys="left",
            backbone=backbone,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
        inputs = image_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = FlatRGBEncoder(
            input_keys="left",
            backbone=FlatBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "frozen, expected_requires_grad",
        [
            (False, True),
            (True, False),
        ],
    )
    def test_frozen_flag_controls_gradients(
        self,
        frozen: bool,
        expected_requires_grad: bool,
    ):
        encoder = FlatRGBEncoder(
            input_keys="left",
            backbone=FlatBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad

    @pytest.mark.integration
    @pytest.mark.parametrize("frozen", [True, False])
    @pytest.mark.parametrize(
        "pooling_method",
        [PoolingMethod.DEFAULT.value, PoolingMethod.LEARNED_AGGREGATION.value],
    )
    def test_frozen_preserved_after_set_image_size(
        self,
        frozen: bool,
        pooling_method: str,
    ):
        encoder = FlatRGBEncoder(
            input_keys="left",
            backbone=FlatBackboneType.DINOV2_VITS14.value,
            pooling_method=pooling_method,
            pretrained=False,
            frozen=frozen,
        )
        encoder.set_image_size(image_height=518, image_width=518)
        for parameter in encoder.parameters():
            assert parameter.requires_grad is not frozen


def _real_flat_build_backbone(self):
    """Side-effect installing a real nn.Linear as backbone so .to(dtype) has effect."""
    backbone = torch.nn.Linear(16, FEATURE_DIM)
    backbone.num_features = FEATURE_DIM
    backbone.num_prefix_tokens = 1
    backbone.patch_embed = None
    self.backbone = backbone
    self.expected_image_size = None
    self.requires_strict_image_size = False
    self.patch_size = None


class TestFlatRGBEncoderModelDtype:
    @pytest.mark.unit
    def test_apply_model_dtype_called_once_in_init(self):
        with (
            patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(FlatRGBEncoder, "_apply_model_dtype") as mock_apply,
        ):
            FlatRGBEncoder(
                input_keys="left",
                backbone=FlatBackboneType.DINOV2_VITS14.value,
                pretrained=False,
                frozen=False,
            )
        mock_apply.assert_called_once()

    @pytest.mark.unit
    def test_apply_model_dtype_called_again_in_set_image_size(self):
        with (
            patch.object(FlatRGBEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(FlatRGBEncoder, "_apply_model_dtype") as mock_apply,
        ):
            encoder = FlatRGBEncoder(
                input_keys="left",
                backbone=FlatBackboneType.DINOV2_VITS14.value,
                pretrained=False,
                frozen=False,
            )
            mock_apply.reset_mock()
            encoder.set_image_size(image_height=224, image_width=224)
        mock_apply.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, expected_dtype",
        [
            (None, torch.float32),
            ("32", torch.float32),
            ("bf16-mixed", torch.bfloat16),
        ],
    )
    def test_all_parameters_share_model_dtype_after_init(
        self,
        model_dtype: str | None,
        expected_dtype: torch.dtype,
    ):
        with patch.object(FlatRGBEncoder, "_build_backbone", _real_flat_build_backbone):
            encoder = FlatRGBEncoder(
                input_keys="left",
                backbone=FlatBackboneType.DINOV2_VITS14.value,
                pooling_method=PoolingMethod.DEFAULT.value,
                pretrained=False,
                frozen=False,
                model_dtype=model_dtype,
            )
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, expected_dtype",
        [("32", torch.float32), ("bf16-mixed", torch.bfloat16)],
    )
    def test_set_image_size_rebuild_preserves_model_dtype(
        self,
        model_dtype: str,
        expected_dtype: torch.dtype,
    ):
        with patch.object(FlatRGBEncoder, "_build_backbone", _real_flat_build_backbone):
            encoder = FlatRGBEncoder(
                input_keys="left",
                backbone=FlatBackboneType.DINOV2_VITS14.value,
                pooling_method=PoolingMethod.DEFAULT.value,
                pretrained=False,
                frozen=False,
                model_dtype=model_dtype,
            )
            encoder.set_image_size(image_height=224, image_width=224)
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype
