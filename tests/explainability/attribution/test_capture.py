"""Tests for versatil.explainability.attribution.capture module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras
from versatil.explainability.attribution.capture import (
    ActivationCapture,
    GradientCapture,
)
from versatil.explainability.attribution.policy import run_policy_for_explanation
from versatil.explainability.constants import VisionCaptureMode
from versatil.explainability.typedefs import CameraExplanationTarget
from versatil.explainability.vision_modules import resolve_camera_explanation_targets
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)


def test_gradient_capture_selects_stacked_camera_gradient_from_full_output():
    layer = nn.Identity()
    target = VisionExplanationTarget(
        layer=layer,
        target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
        activation_layout=ActivationLayout.NLC.value,
        patch_grid=(1, 1),
    )
    camera_target = CameraExplanationTarget(
        camera_key=Cameras.RIGHT.value,
        vision_module_name="decoder.vlm_backbone",
        target=target,
        capture_mode=VisionCaptureMode.STACKED_CAMERA_BATCH.value,
        stacked_camera_index=1,
        stacked_camera_count=2,
    )
    capture = GradientCapture(target=camera_target)
    module_output = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    module_output.requires_grad_()

    capture.forward_hook(
        module=layer,
        module_input=(),
        module_output=module_output,
    )
    module_output.sum().backward()
    activation, gradient = capture.require_tensors()

    assert activation.shape == (2, 1, 2)
    torch.testing.assert_close(
        activation, module_output.detach().reshape(2, 2, 1, 2)[:, 1]
    )
    torch.testing.assert_close(gradient, torch.ones(2, 1, 2))


@pytest.mark.integration
def test_activation_capture_selects_real_smolvla_stacked_camera_activation(
    real_explainability_policy_case_factory: Callable,
):
    case = real_explainability_policy_case_factory(
        case_name="smolvla",
        batch_size=1,
    )
    camera_target = resolve_camera_explanation_targets(
        policy=case.policy,
        target_camera=Cameras.RIGHT.value,
        target_vision_module_names=case.target_vision_module_names,
    )[0]
    capture = ActivationCapture(target=camera_target)
    handle = camera_target.target.layer.register_forward_hook(capture.forward_hook)
    try:
        run_policy_for_explanation(
            policy=case.policy,
            observation=case.observation,
            preprocess_observation=False,
        )
    finally:
        handle.remove()

    activation = capture.require_activation()

    assert activation.dim() == 3
    assert activation.shape[0] == 1
    assert (
        activation.shape[1]
        == case.policy.decoder.vlm_backbone.num_image_tokens_per_camera
    )


@pytest.mark.integration
@pytest.mark.parametrize("policy_case_name", ["paligemma_vlm", "prismatic_vlm"])
def test_activation_capture_selects_real_per_camera_vlm_activation(
    real_explainability_policy_case_factory: Callable,
    policy_case_name: str,
):
    case = real_explainability_policy_case_factory(
        case_name=policy_case_name,
        batch_size=1,
    )
    camera_target = resolve_camera_explanation_targets(
        policy=case.policy,
        target_camera=case.target_camera,
        target_vision_module_names=case.target_vision_module_names,
    )[0]
    capture = ActivationCapture(target=camera_target)
    handle = camera_target.target.layer.register_forward_hook(capture.forward_hook)
    try:
        run_policy_for_explanation(
            policy=case.policy,
            observation=case.observation,
            preprocess_observation=False,
        )
    finally:
        handle.remove()

    activation = capture.require_activation()

    assert activation.dim() == 3
    assert activation.shape[0] == 1
    assert camera_target.invocation_index == 1
