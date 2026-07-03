"""Tests for versatil.explainability.attribution.gradients module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras
from versatil.explainability.attribution.gradients import (
    compute_gradient_maps_for_policy,
)
from versatil.explainability.constants import ExplanationType
from versatil.models.policy import Policy


def test_compute_gradient_maps_for_policy_returns_camera_heatmaps(
    explainability_encoding_pipeline_factory: Callable[[], nn.Module],
    explainability_policy_factory: Callable[..., Policy],
    camera_observation_factory: Callable[[], dict[str, torch.Tensor]],
):
    policy = explainability_policy_factory(
        encoding_pipeline=explainability_encoding_pipeline_factory()
    )
    observation = camera_observation_factory()

    heatmaps = compute_gradient_maps_for_policy(
        policy=policy,
        observation=observation,
        explanation_type=ExplanationType.GRADCAM.value,
        preprocess_observation=False,
    )

    heatmap = heatmaps[Cameras.LEFT.value]
    assert heatmap.shape == (1, 1, 4, 4)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_compute_gradient_maps_for_policy_selects_requested_camera_invocation(
    multi_camera_encoding_pipeline_factory: Callable[[], nn.Module],
    explainability_policy_factory: Callable[..., Policy],
    multi_camera_observation_factory: Callable[[], dict[str, torch.Tensor]],
):
    policy = explainability_policy_factory(
        encoding_pipeline=multi_camera_encoding_pipeline_factory()
    )
    observation = multi_camera_observation_factory()

    heatmaps = compute_gradient_maps_for_policy(
        policy=policy,
        observation=observation,
        explanation_type=ExplanationType.GRADCAM.value,
        target_camera=Cameras.LEFT.value,
        preprocess_observation=False,
    )

    assert set(heatmaps) == {Cameras.LEFT.value}
    assert heatmaps[Cameras.LEFT.value].sum() > 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "explanation_type",
    [
        ExplanationType.GRADCAM.value,
        ExplanationType.GRADCAM_PLUS_PLUS.value,
    ],
)
@pytest.mark.parametrize(
    "policy_case_name",
    [
        "spatial_resnet18",
        "spatial_efficientnet_b0",
        "spatial_convnext_nano",
        "spatial_tiny_vit",
        "flat_deit_tiny",
        "flat_deit_small",
        "smolvla",
        "paligemma_vlm",
        "prismatic_vlm",
    ],
)
def test_compute_gradient_maps_for_policy_supports_real_model_wiring(
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

    heatmaps = compute_gradient_maps_for_policy(
        policy=case.policy,
        observation=case.observation,
        explanation_type=explanation_type,
        target_camera=case.target_camera,
        target_vision_module_names=case.target_vision_module_names,
        preprocess_observation=False,
    )

    heatmap = heatmaps[case.expected_camera]
    image_tensor = case.observation[case.expected_camera]
    assert set(heatmaps) == {case.expected_camera}
    assert heatmap.shape == (
        batch_size,
        image_tensor.shape[1],
        image_tensor.shape[-2],
        image_tensor.shape[-1],
    )
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


@pytest.mark.integration
def test_compute_gradient_maps_for_policy_supports_fully_frozen_towers(
    real_explainability_policy_case_factory: Callable,
):
    case = real_explainability_policy_case_factory(
        case_name="spatial_resnet18",
        batch_size=2,
    )
    for parameter in case.policy.parameters():
        parameter.requires_grad_(False)

    heatmaps = compute_gradient_maps_for_policy(
        policy=case.policy,
        observation=case.observation,
        explanation_type=ExplanationType.GRADCAM.value,
        target_camera=case.target_camera,
        target_vision_module_names=case.target_vision_module_names,
        preprocess_observation=False,
    )

    heatmap = heatmaps[case.expected_camera]
    assert torch.isfinite(heatmap).all()
    assert heatmap.abs().sum() > 0
