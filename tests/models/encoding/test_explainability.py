"""Tests for versatil.models.encoding.explainability module."""

import re
from unittest.mock import MagicMock

import pytest
import timm
import torch.nn as nn

from versatil.models.encoding.encoders.constants import SpatialBackboneType
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
    resolve_timm_feature_info_layer,
)

REAL_TIMM_FEATURE_INFO_BACKBONES = [
    SpatialBackboneType.RESNET18.value,
    SpatialBackboneType.EFFICIENTNET_B0.value,
    SpatialBackboneType.CONVNEXT_NANO.value,
    SpatialBackboneType.TINY_VIT_21M.value,
    SpatialBackboneType.SWIN_TINY.value,
]


class TestVisionExplanationTargetValidation:
    def test_accepts_valid_target_metadata(self):
        layer = nn.Identity()
        target = VisionExplanationTarget(
            layer=layer,
            target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            activation_layout=ActivationLayout.NCHW.value,
        )
        assert target.layer is layer

    def test_raises_for_invalid_target_kind(self):
        invalid_kind = "invalid"
        valid_kinds = [kind.value for kind in ExplanationTargetKind]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Invalid target_kind '{invalid_kind}'. Must be one of: {valid_kinds}"
            ),
        ):
            VisionExplanationTarget(
                layer=nn.Identity(),
                target_kind=invalid_kind,
                activation_layout=ActivationLayout.NCHW.value,
            )

    def test_raises_for_negative_prefix_token_count(self):
        prefix_token_count = -1
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"prefix_token_count must be non-negative. Got: {prefix_token_count}"
            ),
        ):
            VisionExplanationTarget(
                layer=nn.Identity(),
                target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
                activation_layout=ActivationLayout.NLC.value,
                prefix_token_count=prefix_token_count,
            )


class TestResolveTimmFeatureInfoLayer:
    def test_returns_module_from_direct_feature_info_name(self):
        target_layer = nn.Identity()
        backbone = nn.Module()
        backbone.layer4 = target_layer
        backbone.feature_info = MagicMock()
        backbone.feature_info.module_name.return_value = "layer4"

        result = resolve_timm_feature_info_layer(backbone=backbone, layer_index=4)

        assert result is target_layer

    def test_returns_module_from_flattened_feature_info_name(self):
        target_layer = nn.Identity()
        backbone = nn.Module()
        backbone.stages_3 = target_layer
        backbone.feature_info = MagicMock()
        backbone.feature_info.module_name.return_value = "stages.3"

        result = resolve_timm_feature_info_layer(backbone=backbone, layer_index=3)

        assert result is target_layer

    def test_returns_none_without_feature_info(self):
        result = resolve_timm_feature_info_layer(
            backbone=nn.Identity(),
            layer_index=0,
        )

        assert result is None

    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", REAL_TIMM_FEATURE_INFO_BACKBONES)
    def test_resolves_real_timm_feature_info_module(self, backbone: str):
        model = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
        )
        layer_index = len(model.feature_info.channels()) - 1
        module_name = model.feature_info.module_name(layer_index)
        named_modules = dict(model.named_modules())
        expected_layer = named_modules.get(module_name)
        if expected_layer is None:
            expected_layer = named_modules[module_name.replace(".", "_")]

        result = resolve_timm_feature_info_layer(
            backbone=model,
            layer_index=layer_index,
        )

        assert result is expected_layer
