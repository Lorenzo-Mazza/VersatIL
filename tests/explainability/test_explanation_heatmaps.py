"""Tests for versatil.explainability.explanation_heatmaps module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import Cameras
from versatil.explainability.constants import ExplanationType
from versatil.explainability.explanation_heatmaps import to_explanation_heatmaps


def test_to_explanation_heatmaps_dispatches_supported_methods():
    policy = MagicMock()
    observation = {Cameras.LEFT.value: torch.zeros(1, 1, 3, 4, 4)}
    gradient_heatmaps = {Cameras.LEFT.value: torch.ones(1, 1, 4, 4)}
    ablation_heatmaps = {Cameras.LEFT.value: torch.full((1, 1, 4, 4), 0.5)}

    with (
        patch(
            "versatil.explainability.explanation_heatmaps."
            "compute_gradient_maps_for_policy",
            return_value=gradient_heatmaps,
        ) as mock_gradient_heatmaps,
        patch(
            "versatil.explainability.explanation_heatmaps."
            "compute_ablation_maps_for_policy",
            return_value=ablation_heatmaps,
        ) as mock_ablation_heatmaps,
    ):
        heatmap_functions = to_explanation_heatmaps(channel_batch_size=7)
        gradcam_result = heatmap_functions[ExplanationType.GRADCAM.value](
            policy=policy,
            observation=observation,
            actions=None,
            target_camera=Cameras.LEFT.value,
            target_vision_module_names=None,
            preprocess_observation=False,
        )
        gradcam_plus_plus_result = heatmap_functions[
            ExplanationType.GRADCAM_PLUS_PLUS.value
        ](
            policy=policy,
            observation=observation,
            actions=None,
            target_camera=Cameras.LEFT.value,
            target_vision_module_names=None,
            preprocess_observation=False,
        )
        ablation_result = heatmap_functions[ExplanationType.ABLATION_CAM.value](
            policy=policy,
            observation=observation,
            actions=None,
            target_camera=Cameras.LEFT.value,
            target_vision_module_names=None,
            preprocess_observation=False,
        )

    assert set(heatmap_functions) == {member.value for member in ExplanationType}
    assert torch.equal(
        gradcam_result[Cameras.LEFT.value], gradient_heatmaps[Cameras.LEFT.value]
    )
    assert torch.equal(
        gradcam_plus_plus_result[Cameras.LEFT.value],
        gradient_heatmaps[Cameras.LEFT.value],
    )
    assert torch.equal(
        ablation_result[Cameras.LEFT.value], ablation_heatmaps[Cameras.LEFT.value]
    )
    mock_gradient_heatmaps.assert_any_call(
        policy=policy,
        observation=observation,
        actions=None,
        explanation_type=ExplanationType.GRADCAM.value,
        target_camera=Cameras.LEFT.value,
        target_vision_module_names=None,
        preprocess_observation=False,
    )
    mock_gradient_heatmaps.assert_any_call(
        policy=policy,
        observation=observation,
        actions=None,
        explanation_type=ExplanationType.GRADCAM_PLUS_PLUS.value,
        target_camera=Cameras.LEFT.value,
        target_vision_module_names=None,
        preprocess_observation=False,
    )
    mock_ablation_heatmaps.assert_called_once_with(
        policy=policy,
        observation=observation,
        actions=None,
        channel_batch_size=7,
        target_camera=Cameras.LEFT.value,
        target_vision_module_names=None,
        preprocess_observation=False,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "policy_case_name, explanation_type",
    [
        ("spatial_resnet18", ExplanationType.GRADCAM.value),
        ("flat_deit_tiny", ExplanationType.GRADCAM_PLUS_PLUS.value),
        ("smolvla", ExplanationType.ABLATION_CAM.value),
        ("paligemma_vlm", ExplanationType.GRADCAM.value),
        ("prismatic_vlm", ExplanationType.GRADCAM_PLUS_PLUS.value),
    ],
)
def test_to_explanation_heatmaps_dispatches_real_policy_classes(
    real_explainability_policy_case_factory: Callable,
    policy_case_name: str,
    explanation_type: str,
):
    batch_size = (
        1 if policy_case_name.endswith("_vlm") or policy_case_name == "smolvla" else 2
    )
    case = real_explainability_policy_case_factory(
        case_name=policy_case_name,
        batch_size=batch_size,
    )
    heatmap_functions = to_explanation_heatmaps(channel_batch_size=64)

    result = heatmap_functions[explanation_type](
        policy=case.policy,
        observation=case.observation,
        actions=None,
        target_camera=case.target_camera,
        target_vision_module_names=case.target_vision_module_names,
        preprocess_observation=False,
    )

    assert set(result) == {case.expected_camera}
    assert result[case.expected_camera].shape[0] == batch_size
    assert torch.isfinite(result[case.expected_camera]).all()
