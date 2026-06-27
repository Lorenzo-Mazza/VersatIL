"""Tests for versatil.explainability.vision_modules module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from versatil.data.constants import Cameras, ProprioKey
from versatil.explainability.constants import VisionCaptureMode
from versatil.explainability.vision_modules import (
    get_vision_explainable_modules,
    resolve_camera_explanation_targets,
    select_explainability_target,
)
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)
from versatil.models.input_specification import InputSpecification


class _ExplainableEncoder(nn.Module):
    def __init__(
        self,
        input_keys: list[str],
        targets: list[VisionExplanationTarget],
        camera_keys: list[str] | None = None,
        is_multi_camera: bool = False,
    ) -> None:
        super().__init__()
        self.layer = nn.Identity()
        self.input_specification = InputSpecification(keys=input_keys)
        self._targets = targets
        if camera_keys is not None:
            self.camera_keys = camera_keys
        self.is_multi_camera = is_multi_camera

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        return self._targets


@pytest.fixture
def vision_target_factory() -> Callable[[], VisionExplanationTarget]:
    def factory() -> VisionExplanationTarget:
        return VisionExplanationTarget(
            layer=nn.Identity(),
            target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            activation_layout=ActivationLayout.NCHW.value,
        )

    return factory


@pytest.fixture
def vision_policy_factory() -> Callable[..., MagicMock]:
    def factory(
        encoders: dict[str, nn.Module] | None = None,
        conditional_encoders: dict[str, nn.Module] | None = None,
        decoder: nn.Module | None = None,
        camera_keys: list[str] | None = None,
    ) -> MagicMock:
        if camera_keys is None:
            camera_keys = [Cameras.LEFT.value]
        policy = MagicMock()
        policy.observation_space.cameras = {key: MagicMock() for key in camera_keys}
        policy.encoding_pipeline.encoders = encoders if encoders is not None else {}
        policy.encoding_pipeline.conditional_encoders = (
            conditional_encoders if conditional_encoders is not None else {}
        )
        policy.decoder = decoder if decoder is not None else nn.Module()
        return policy

    return factory


class TestGetVisionExplainableModules:
    def test_raises_when_module_exposes_multiple_compatible_targets(self):
        layer = nn.Identity()
        targets = [
            VisionExplanationTarget(
                layer=layer,
                target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
                activation_layout=ActivationLayout.NCHW.value,
            ),
            VisionExplanationTarget(
                layer=layer,
                target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
                activation_layout=ActivationLayout.NLC.value,
            ),
        ]
        expected_message = (
            "Visual module 'hybrid_encoder' exposes multiple compatible "
            "explainability targets ['spatial_feature_map', 'token_sequence']. "
            "Configure the module to expose exactly one target until per-target "
            "selection is supported."
        )

        with pytest.raises(RuntimeError, match=re.escape(expected_message)):
            select_explainability_target(
                targets=targets,
                module_name="hybrid_encoder",
            )

    def test_returns_encoder_and_conditional_encoder_targets(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        rgb_encoder = _ExplainableEncoder(
            input_keys=[Cameras.LEFT.value],
            targets=[vision_target_factory()],
        )
        conditional_encoder = _ExplainableEncoder(
            input_keys=[Cameras.RIGHT.value],
            targets=[vision_target_factory()],
        )
        proprio_encoder = _ExplainableEncoder(
            input_keys=[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value],
            targets=[],
        )
        policy = vision_policy_factory(
            encoders={
                "rgb_encoder": rgb_encoder,
                "proprio_encoder": proprio_encoder,
            },
            conditional_encoders={"conditional_rgb": conditional_encoder},
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        )

        result = get_vision_explainable_modules(policy=policy)

        assert [module.name for module in result] == [
            "rgb_encoder",
            "conditional_rgb",
        ]

    def test_raises_when_policy_exposes_no_visual_targets(
        self,
        vision_policy_factory: Callable[..., MagicMock],
    ):
        policy = vision_policy_factory()

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "No compatible vision explainability modules found. "
                "Explainability requires visual modules that expose target metadata "
                "through get_explainability_targets()."
            ),
        ):
            get_vision_explainable_modules(policy=policy)

    def test_returns_decoder_vlm_vision_encoder_targets(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        tower = _ExplainableEncoder(
            input_keys=[Cameras.LEFT.value],
            targets=[vision_target_factory()],
        )
        vlm_backbone = nn.Module()
        vlm_backbone.camera_keys = [Cameras.LEFT.value]
        vlm_backbone.vision_encoders = nn.ModuleList([tower])
        decoder = nn.Module()
        decoder.vlm_backbone = vlm_backbone
        policy = vision_policy_factory(decoder=decoder)

        result = get_vision_explainable_modules(policy=policy)

        assert [module.name for module in result] == [
            "decoder.vlm_backbone.vision_encoders.0"
        ]

    def test_returns_decoder_vlm_direct_stacked_camera_target(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        vlm_backbone = nn.Module()
        vlm_backbone.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
        vlm_backbone.is_stacked_camera_batch = True
        vlm_backbone.get_explainability_targets = MagicMock(
            return_value=[vision_target_factory()]
        )
        decoder = nn.Module()
        decoder.vlm_backbone = vlm_backbone
        policy = vision_policy_factory(
            decoder=decoder,
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        )

        result = get_vision_explainable_modules(policy=policy)

        assert [module.name for module in result] == ["decoder.vlm_backbone"]
        assert result[0].capture_mode == VisionCaptureMode.STACKED_CAMERA_BATCH.value

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "policy_case_name, expected_module_name, expected_capture_mode, expected_kind",
        [
            (
                "spatial_resnet18",
                "vision",
                VisionCaptureMode.SINGLE_CALL.value,
                ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            ),
            (
                "spatial_efficientnet_b0",
                "vision",
                VisionCaptureMode.SINGLE_CALL.value,
                ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            ),
            (
                "spatial_convnext_nano",
                "vision",
                VisionCaptureMode.SINGLE_CALL.value,
                ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            ),
            (
                "spatial_tiny_vit",
                "vision",
                VisionCaptureMode.SINGLE_CALL.value,
                ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            ),
            (
                "flat_deit_tiny",
                "vision",
                VisionCaptureMode.SINGLE_CALL.value,
                ExplanationTargetKind.TOKEN_SEQUENCE.value,
            ),
            (
                "flat_deit_small",
                "vision",
                VisionCaptureMode.SINGLE_CALL.value,
                ExplanationTargetKind.TOKEN_SEQUENCE.value,
            ),
            (
                "smolvla",
                "decoder.vlm_backbone",
                VisionCaptureMode.STACKED_CAMERA_BATCH.value,
                ExplanationTargetKind.TOKEN_SEQUENCE.value,
            ),
        ],
    )
    def test_returns_real_policy_visual_targets(
        self,
        real_explainability_policy_case_factory: Callable,
        policy_case_name: str,
        expected_module_name: str,
        expected_capture_mode: str,
        expected_kind: str,
    ):
        case = real_explainability_policy_case_factory(case_name=policy_case_name)

        result = get_vision_explainable_modules(policy=case.policy)

        assert expected_module_name in [module.name for module in result]
        module = next(
            candidate for candidate in result if candidate.name == expected_module_name
        )
        assert module.capture_mode == expected_capture_mode
        assert module.target.target_kind == expected_kind


class TestResolveCameraExplanationTargets:
    def test_assigns_camera_invocation_for_multi_camera_modules(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        encoder = _ExplainableEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
            targets=[vision_target_factory()],
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
            is_multi_camera=True,
        )
        policy = vision_policy_factory(
            encoders={"rgb_encoder": encoder},
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        )

        result = resolve_camera_explanation_targets(
            policy=policy,
            target_camera=Cameras.RIGHT.value,
        )

        assert len(result) == 1
        assert result[0].camera_key == Cameras.RIGHT.value
        assert result[0].vision_module_name == "rgb_encoder"
        assert result[0].capture_mode == VisionCaptureMode.PER_CAMERA_CALL.value
        assert result[0].invocation_index == 1

    def test_assigns_stacked_camera_index_for_stacked_vlm_modules(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        vlm_backbone = nn.Module()
        vlm_backbone.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
        vlm_backbone.is_stacked_camera_batch = True
        vlm_backbone.get_explainability_targets = MagicMock(
            return_value=[vision_target_factory()]
        )
        decoder = nn.Module()
        decoder.vlm_backbone = vlm_backbone
        policy = vision_policy_factory(
            decoder=decoder,
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        )

        result = resolve_camera_explanation_targets(
            policy=policy,
            target_camera=Cameras.RIGHT.value,
        )

        assert len(result) == 1
        assert result[0].camera_key == Cameras.RIGHT.value
        assert result[0].capture_mode == VisionCaptureMode.STACKED_CAMERA_BATCH.value
        assert result[0].stacked_camera_index == 1
        assert result[0].stacked_camera_count == 2

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "policy_case_name, expected_capture_mode, expected_stacked_index",
        [
            ("spatial_resnet18", VisionCaptureMode.SINGLE_CALL.value, None),
            ("spatial_efficientnet_b0", VisionCaptureMode.SINGLE_CALL.value, None),
            ("spatial_convnext_nano", VisionCaptureMode.SINGLE_CALL.value, None),
            ("spatial_tiny_vit", VisionCaptureMode.SINGLE_CALL.value, None),
            ("flat_deit_tiny", VisionCaptureMode.SINGLE_CALL.value, None),
            ("flat_deit_small", VisionCaptureMode.SINGLE_CALL.value, None),
            ("smolvla", VisionCaptureMode.STACKED_CAMERA_BATCH.value, 0),
        ],
    )
    def test_resolves_real_policy_camera_target(
        self,
        real_explainability_policy_case_factory: Callable,
        policy_case_name: str,
        expected_capture_mode: str,
        expected_stacked_index: int | None,
    ):
        case = real_explainability_policy_case_factory(case_name=policy_case_name)

        result = resolve_camera_explanation_targets(
            policy=case.policy,
            target_camera=case.target_camera,
            target_vision_module_names=case.target_vision_module_names,
        )

        assert len(result) == 1
        assert result[0].vision_module_name == case.expected_vision_module_names[0]
        assert result[0].camera_key == case.expected_camera
        assert result[0].capture_mode == expected_capture_mode
        assert result[0].stacked_camera_index == expected_stacked_index

    @pytest.mark.integration
    def test_resolves_real_smolvla_right_camera_stacked_target(
        self,
        real_explainability_policy_case_factory: Callable,
    ):
        case = real_explainability_policy_case_factory(case_name="smolvla")

        result = resolve_camera_explanation_targets(
            policy=case.policy,
            target_camera=Cameras.RIGHT.value,
            target_vision_module_names=case.target_vision_module_names,
        )

        assert len(result) == 1
        assert result[0].vision_module_name == "decoder.vlm_backbone"
        assert result[0].camera_key == Cameras.RIGHT.value
        assert result[0].capture_mode == VisionCaptureMode.STACKED_CAMERA_BATCH.value
        assert result[0].stacked_camera_index == 1
        assert result[0].stacked_camera_count == 2

    def test_filters_by_visual_module_name(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        left_encoder = _ExplainableEncoder(
            input_keys=[Cameras.LEFT.value],
            targets=[vision_target_factory()],
        )
        right_encoder = _ExplainableEncoder(
            input_keys=[Cameras.RIGHT.value],
            targets=[vision_target_factory()],
        )
        policy = vision_policy_factory(
            encoders={
                "left_encoder": left_encoder,
                "right_encoder": right_encoder,
            },
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        )

        result = resolve_camera_explanation_targets(
            policy=policy,
            target_vision_module_names=["right_encoder"],
        )

        assert [target.camera_key for target in result] == [Cameras.RIGHT.value]

    def test_raises_when_camera_filter_matches_no_module(
        self,
        vision_target_factory: Callable[[], VisionExplanationTarget],
        vision_policy_factory: Callable[..., MagicMock],
    ):
        encoder = _ExplainableEncoder(
            input_keys=[Cameras.LEFT.value],
            targets=[vision_target_factory()],
        )
        policy = vision_policy_factory(encoders={"rgb_encoder": encoder})
        expected_message = (
            f"target_camera='{Cameras.RIGHT.value}' did not match visual module "
            f"cameras: ['{Cameras.LEFT.value}']"
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_camera_explanation_targets(
                policy=policy,
                target_camera=Cameras.RIGHT.value,
            )
