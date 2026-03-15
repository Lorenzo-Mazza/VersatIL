"""Tests for versatil.configs.data.raw.zarr_meta module."""
import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.data.metadata import CameraMetadataConfig
from versatil.configs.data.raw.zarr_meta import DatasetMetadataConfig


@pytest.mark.unit
class TestDatasetMetadataConfig:

    def test_target_points_to_dataset_metadata(self):
        config = DatasetMetadataConfig()
        assert config._target_ == "versatil.data.raw.zarr_meta.DatasetMetadata"

    def test_observations_required(self):
        config = DatasetMetadataConfig()
        assert config.observations == MISSING

    def test_precomputed_actions_defaults_to_empty_dict(self):
        config = DatasetMetadataConfig()
        assert config.precomputed_actions == {}


@pytest.mark.unit
class TestDatasetMetadataInstantiation:

    def test_dataset_metadata_instantiates_with_observations(self):
        config = DatasetMetadataConfig(
            observations={
                "left": CameraMetadataConfig(
                    camera_key="left",
                    dtype="float32",
                    channels=3,
                ),
            },
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "DatasetMetadata"
        assert "left" in instance.observations

    def test_dataset_metadata_instantiates_with_empty_observations(self):
        config = DatasetMetadataConfig(observations={})
        instance = instantiate(config)
        assert type(instance).__name__ == "DatasetMetadata"
        assert len(instance.observations) == 0
