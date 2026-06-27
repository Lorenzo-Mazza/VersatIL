"""Tests for versatil.endpoints.explain module."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from versatil.endpoints.explain import main
from versatil.explainability.constants import ExplanationSourceType, ExplanationType
from versatil.explainability.sources.typedefs import ExplanationBatch


@pytest.mark.integration
@pytest.mark.parametrize(
    "policy_case_name",
    ["spatial_resnet18", "flat_deit_tiny", "smolvla"],
)
def test_main_writes_real_policy_gradcam_heatmaps(
    tmp_path: Path,
    real_explainability_policy_case_factory: Callable,
    policy_case_name: str,
):
    batch_size = 1 if policy_case_name == "smolvla" else 2
    case = real_explainability_policy_case_factory(
        case_name=policy_case_name,
        batch_size=batch_size,
    )
    checkpoint_loader = MagicMock()
    checkpoint_loader.config = MagicMock()
    checkpoint_loader.policy = case.policy
    batch = ExplanationBatch(
        observation=case.observation,
        actions=None,
        display_observation={
            case.expected_camera: case.observation[case.expected_camera],
        },
        metadata={
            "source": ExplanationSourceType.DATASET.value,
            "split": "all",
            "sample_indices": [0],
        },
        preprocess_observation=False,
    )
    output_directory = tmp_path / "explain_outputs"
    config = OmegaConf.create(
        {
            "_target_": "versatil.explainability.runner.ExplainabilityRunner",
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "checkpoint_name": "last.ckpt",
            "output_directory": str(output_directory),
            "device": "cpu",
            "source": ExplanationSourceType.DATASET.value,
            "split": "all",
            "sample_stride": 1,
            "max_samples": 1,
            "batch_size": batch_size,
            "channel_batch_size": 64,
            "explanation_types": [ExplanationType.GRADCAM.value],
            "target_camera_keys": [case.target_camera],
            "target_vision_module_names": case.target_vision_module_names,
            "save_raw_heatmaps": True,
            "save_overlays": False,
        }
    )

    with (
        patch(
            "versatil.explainability.runner.FloatCheckpointLoader",
            return_value=checkpoint_loader,
        ),
        patch(
            "versatil.explainability.runner.DatasetExplanationSource",
            return_value=[batch],
        ),
    ):
        main(config)

    output_path = (
        output_directory
        / ExplanationSourceType.DATASET.value
        / "all"
        / f"batch_0_{ExplanationType.GRADCAM.value}.pt"
    )
    assert output_path.exists()
    saved = torch.load(output_path, map_location="cpu", weights_only=False)
    heatmap = saved["heatmaps"][case.expected_camera]
    assert heatmap.shape == (
        batch_size,
        case.observation[case.expected_camera].shape[1],
        case.observation[case.expected_camera].shape[-2],
        case.observation[case.expected_camera].shape[-1],
    )
    assert torch.isfinite(heatmap).all()
