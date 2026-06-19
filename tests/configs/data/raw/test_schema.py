"""Tests for versatil.configs.data.raw.schema module."""

import pytest
from omegaconf import MISSING

from versatil.configs.data.raw.schema import (
    CsvDatasetSchemaConfig,
    DatasetSchemaConfig,
    Hdf5DatasetSchemaConfig,
    LeRobotDatasetSchemaConfig,
    SyntheticDatasetSchemaConfig,
)


@pytest.mark.unit
def test_base_schema_required_fields_default_to_missing():
    config = DatasetSchemaConfig()
    assert config._target_ == MISSING
    assert config.zarr_path == MISSING
    assert config.metadata == MISSING
    assert config.dataset_type == MISSING


@pytest.mark.unit
@pytest.mark.parametrize(
    "schema_class, extra_field",
    [
        (CsvDatasetSchemaConfig, "dataset_folders"),
        (Hdf5DatasetSchemaConfig, "hdf5_paths"),
        (LeRobotDatasetSchemaConfig, "dataset_path"),
    ],
)
def test_subclass_required_field_defaults_to_missing(schema_class, extra_field):
    config = schema_class()
    assert getattr(config, extra_field) == MISSING
    assert config.zarr_path == MISSING


@pytest.mark.unit
@pytest.mark.parametrize(
    "schema_class",
    [
        CsvDatasetSchemaConfig,
        Hdf5DatasetSchemaConfig,
        LeRobotDatasetSchemaConfig,
        SyntheticDatasetSchemaConfig,
    ],
)
def test_subclass_carries_base_schema_fields(schema_class):
    config = schema_class()
    base_fields = {"_target_", "zarr_path", "metadata", "dataset_type"}
    assert base_fields.issubset(vars(config).keys())


@pytest.mark.unit
@pytest.mark.parametrize("num_episodes", [500, 1000])
@pytest.mark.parametrize("num_modes", [2, 3])
@pytest.mark.parametrize("trajectory_length", [60, 120])
def test_synthetic_schema_stores_configuration(
    num_episodes, num_modes, trajectory_length
):
    config = SyntheticDatasetSchemaConfig(
        task_name="conditional_circle",
        num_episodes=num_episodes,
        seed=7,
        image_size=128,
        num_modes=num_modes,
        trajectory_length=trajectory_length,
        noise_std=0.05,
        num_styles=3,
        mode_weights=[0.6, 0.4],
        num_rollouts=50,
    )
    assert config.task_name == "conditional_circle"
    assert config.num_episodes == num_episodes
    assert config.seed == 7
    assert config.image_size == 128
    assert config.num_modes == num_modes
    assert config.trajectory_length == trajectory_length
    assert config.noise_std == 0.05
    assert config.num_styles == 3
    assert config.mode_weights == [0.6, 0.4]
    assert config.num_rollouts == 50
