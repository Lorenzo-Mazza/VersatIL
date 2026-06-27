"""Tests for versatil.explainability.attribution.maps module."""

import re
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.explainability.attribution.capture import ActivationCapture
from versatil.explainability.attribution.maps import (
    nchw_to_target_tensor,
    target_tensor_to_nchw,
)
from versatil.explainability.attribution.policy import run_policy_for_explanation
from versatil.explainability.vision_modules import resolve_camera_explanation_targets
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)


class TestNchwToTargetTensor:
    def test_converts_spatial_tensor_to_target_layout(self):
        target = VisionExplanationTarget(
            layer=nn.Identity(),
            target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
            activation_layout=ActivationLayout.NHWC.value,
        )
        tensor = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        original_tensor = torch.empty(1, 3, 4, 2)

        result = nchw_to_target_tensor(
            tensor=tensor,
            target=target,
            original_tensor=original_tensor,
        )

        torch.testing.assert_close(result, tensor.permute(0, 2, 3, 1).contiguous())

    def test_restores_token_sequence_with_prefix_tokens(self):
        target = VisionExplanationTarget(
            layer=nn.Identity(),
            target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
            activation_layout=ActivationLayout.NLC.value,
            prefix_token_count=1,
            patch_grid=(2, 2),
        )
        tensor = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        original_tensor = torch.full((1, 5, 3), -1.0)

        result = nchw_to_target_tensor(
            tensor=tensor,
            target=target,
            original_tensor=original_tensor,
        )

        expected_patch_tokens = tensor.permute(0, 2, 3, 1).reshape(1, 4, 3)
        expected = torch.cat([original_tensor[:, :1], expected_patch_tokens], dim=1)
        torch.testing.assert_close(result, expected)

    def test_raises_when_token_grid_does_not_match_tensor_grid(self):
        target = VisionExplanationTarget(
            layer=nn.Identity(),
            target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
            activation_layout=ActivationLayout.NLC.value,
            prefix_token_count=1,
            patch_grid=(1, 4),
        )
        tensor = torch.zeros(1, 3, 2, 2)
        original_tensor = torch.zeros(1, 5, 3)

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Ablated token grid (2, 2) does not match target patch grid (1, 4)."
            ),
        ):
            nchw_to_target_tensor(
                tensor=tensor,
                target=target,
                original_tensor=original_tensor,
            )

    def test_raises_when_prefix_token_count_exceeds_sequence_length(self):
        target = VisionExplanationTarget(
            layer=nn.Identity(),
            target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
            activation_layout=ActivationLayout.NLC.value,
            prefix_token_count=6,
            patch_grid=(1, 1),
        )
        tensor = torch.zeros(1, 3, 1, 1)
        original_tensor = torch.zeros(1, 5, 3)

        with pytest.raises(
            ValueError,
            match=re.escape("prefix_token_count 6 exceeds original token count 5."),
        ):
            nchw_to_target_tensor(
                tensor=tensor,
                target=target,
                original_tensor=original_tensor,
            )


@pytest.mark.integration
@pytest.mark.parametrize(
    "policy_case_name",
    [
        "spatial_resnet18",
        "flat_deit_tiny",
        "smolvla",
    ],
)
def test_target_tensor_to_nchw_supports_real_model_targets(
    real_explainability_policy_case_factory: Callable,
    policy_case_name: str,
):
    batch_size = 1 if policy_case_name == "smolvla" else 2
    case = real_explainability_policy_case_factory(
        case_name=policy_case_name,
        batch_size=batch_size,
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

    result = target_tensor_to_nchw(
        tensor=capture.require_activation(),
        target=camera_target.target,
        tensor_name="activations",
    )

    assert result.dim() == 4
    assert result.shape[0] == batch_size
    assert torch.isfinite(result).all()
