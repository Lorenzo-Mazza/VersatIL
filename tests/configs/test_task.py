"""Tests for versatil.configs.data.task module."""

import dataclasses

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.data.metadata import CameraMetadataConfig
from versatil.configs.data.task import (
    ActionSpaceConfig,
    ObservationSpaceConfig,
    TaskSpaceConfig,
)


@pytest.mark.unit
class TestActionSpaceConfig:
    def test_target_points_to_action_space(self):
        config = ActionSpaceConfig()
        assert config._target_ == "versatil.data.task.ActionSpace"

    def test_actions_metadata_defaults_to_empty_dict(self):
        config = ActionSpaceConfig()
        assert config.actions_metadata == {}

    @pytest.mark.parametrize("use_gripper_class_weights", [True, False])
    @pytest.mark.parametrize("denoise_actions", [True, False])
    def test_stores_configuration(self, use_gripper_class_weights, denoise_actions):
        config = ActionSpaceConfig(
            use_gripper_class_weights=use_gripper_class_weights,
            denoise_actions=denoise_actions,
        )
        assert config.use_gripper_class_weights == use_gripper_class_weights
        assert config.denoise_actions == denoise_actions

    @pytest.mark.parametrize("denoising_percentile", [10.0, 20.0])
    def test_stores_denoising_percentile(self, denoising_percentile):
        config = ActionSpaceConfig(denoising_percentile=denoising_percentile)
        assert config.denoising_percentile == denoising_percentile


@pytest.mark.unit
class TestObservationSpaceConfig:
    def test_target_points_to_observation_space(self):
        config = ObservationSpaceConfig()
        assert config._target_ == "versatil.data.task.ObservationSpace"

    def test_observations_metadata_defaults_to_empty_dict(self):
        config = ObservationSpaceConfig()
        assert config.observations_metadata == {}


@pytest.mark.unit
class TestTaskSpaceConfig:
    def test_target_points_to_task_space(self):
        config = TaskSpaceConfig()
        assert config._target_ == "versatil.data.task.TaskSpace"

    def test_dataset_schema_defaults_to_missing(self):
        config = TaskSpaceConfig()
        assert config.dataset_schema == MISSING

    def test_dataloader_defaults_to_missing(self):
        config = TaskSpaceConfig()
        assert config.dataloader == MISSING

    def test_action_space_defaults_to_missing(self):
        config = TaskSpaceConfig()
        assert config.action_space == MISSING

    def test_observation_space_defaults_to_missing(self):
        config = TaskSpaceConfig()
        assert config.observation_space == MISSING

    @pytest.mark.parametrize("observation_horizon", [1, 4])
    @pytest.mark.parametrize("prediction_horizon", [8, 32])
    def test_stores_horizons(self, observation_horizon, prediction_horizon):
        config = TaskSpaceConfig(
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        assert config.observation_horizon == observation_horizon
        assert config.prediction_horizon == prediction_horizon

    def test_has_all_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(TaskSpaceConfig)}
        expected = {
            "_target_",
            "dataset_schema",
            "dataloader",
            "action_space",
            "observation_space",
            "observation_horizon",
            "prediction_horizon",
        }
        assert expected == field_names


@pytest.mark.unit
class TestTaskConfigInstantiation:
    def test_action_space_instantiates_with_empty_metadata(self):
        config = ActionSpaceConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "ActionSpace"
        assert instance.actions_metadata == {}

    def test_action_space_instantiates_with_parameter_passthrough(self):
        config = ActionSpaceConfig(
            use_gripper_class_weights=True,
            denoise_actions=False,
            denoising_percentile=25.0,
        )
        instance = instantiate(config)
        assert instance.use_gripper_class_weights is True
        assert instance.denoise_actions is False
        assert instance.denoising_percentile == 25.0

    def test_observation_space_instantiates_with_empty_metadata(self):
        config = ObservationSpaceConfig()
        instance = instantiate(config)
        assert type(instance).__name__ == "ObservationSpace"
        assert instance.observations_metadata == {}

    def test_observation_space_instantiates_with_camera(self):
        config = ObservationSpaceConfig(
            observations_metadata={
                "left": CameraMetadataConfig(
                    camera_key="left",
                    dtype="float32",
                    channels=3,
                ),
            },
        )
        instance = instantiate(config)
        assert "left" in instance.observations_metadata
        assert instance.has_cameras is True

    def test_observation_space_instantiates_with_multiple_cameras(self):
        config = ObservationSpaceConfig(
            observations_metadata={
                "left": CameraMetadataConfig(
                    camera_key="left",
                    dtype="float32",
                    channels=3,
                ),
                "right": CameraMetadataConfig(
                    camera_key="right",
                    dtype="float32",
                    channels=3,
                ),
            },
        )
        instance = instantiate(config)
        assert len(instance.cameras) == 2
