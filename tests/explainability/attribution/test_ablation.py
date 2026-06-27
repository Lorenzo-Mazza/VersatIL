"""Tests for versatil.explainability.attribution.ablation module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras
from versatil.explainability.attribution.ablation import (
    compute_ablation_maps_for_policy,
)
from versatil.models.policy import Policy


def test_compute_ablation_maps_for_policy_returns_camera_heatmaps(
    explainability_encoding_pipeline_factory: Callable[[], nn.Module],
    explainability_policy_factory: Callable[..., Policy],
    camera_observation_factory: Callable[[], dict[str, torch.Tensor]],
):
    policy = explainability_policy_factory(
        encoding_pipeline=explainability_encoding_pipeline_factory()
    )
    observation = camera_observation_factory()

    heatmaps = compute_ablation_maps_for_policy(
        policy=policy,
        observation=observation,
        channel_batch_size=1,
        preprocess_observation=False,
    )

    heatmap = heatmaps[Cameras.LEFT.value]
    assert heatmap.shape == (1, 1, 4, 4)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_compute_ablation_maps_for_policy_selects_requested_camera_invocation(
    multi_camera_encoding_pipeline_factory: Callable[[], nn.Module],
    explainability_policy_factory: Callable[..., Policy],
    multi_camera_observation_factory: Callable[[], dict[str, torch.Tensor]],
):
    policy = explainability_policy_factory(
        encoding_pipeline=multi_camera_encoding_pipeline_factory()
    )
    observation = multi_camera_observation_factory()

    heatmaps = compute_ablation_maps_for_policy(
        policy=policy,
        observation=observation,
        target_camera=Cameras.LEFT.value,
        channel_batch_size=1,
        preprocess_observation=False,
    )

    assert set(heatmaps) == {Cameras.LEFT.value}
    assert heatmaps[Cameras.LEFT.value].sum() > 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "policy_case_name",
    [
        "spatial_resnet18",
        "flat_deit_tiny",
        "smolvla",
        "paligemma_vlm",
        "prismatic_vlm",
    ],
)
def test_compute_ablation_maps_for_policy_supports_real_model_classes(
    real_explainability_policy_case_factory: Callable,
    policy_case_name: str,
):
    case = real_explainability_policy_case_factory(
        case_name=policy_case_name,
        batch_size=1,
    )

    heatmaps = compute_ablation_maps_for_policy(
        policy=case.policy,
        observation=case.observation,
        target_camera=case.target_camera,
        target_vision_module_names=case.target_vision_module_names,
        channel_batch_size=64,
        preprocess_observation=False,
    )

    heatmap = heatmaps[case.expected_camera]
    image_tensor = case.observation[case.expected_camera]
    assert set(heatmaps) == {case.expected_camera}
    assert heatmap.shape == (
        1,
        image_tensor.shape[1],
        image_tensor.shape[-2],
        image_tensor.shape[-1],
    )
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1
