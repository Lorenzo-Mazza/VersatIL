"""Tests for versatil.configs.data.raw.schema module."""
import pytest
from omegaconf import MISSING

from versatil.configs.data.raw.schema import (
    CsvDatasetSchemaConfig,
    DatasetSchemaConfig,
    Hdf5DatasetSchemaConfig,
    LeRobotDatasetSchemaConfig,
)


@pytest.mark.unit
class TestDatasetSchemaConfig:

    def test_required_fields_default_to_missing(self):
        config = DatasetSchemaConfig()
        assert config._target_ == MISSING
        assert config.zarr_path == MISSING
        assert config.metadata == MISSING
        assert config.dataset_type == MISSING


@pytest.mark.unit
class TestCsvDatasetSchemaConfig:

    def test_dataset_folders_required(self):
        config = CsvDatasetSchemaConfig()
        assert config.dataset_folders == MISSING

    def test_inherits_from_dataset_schema_config(self):
        config = CsvDatasetSchemaConfig()
        assert isinstance(config, DatasetSchemaConfig)


@pytest.mark.unit
class TestHdf5DatasetSchemaConfig:

    def test_hdf5_paths_required(self):
        config = Hdf5DatasetSchemaConfig()
        assert config.hdf5_paths == MISSING

    def test_inherits_from_dataset_schema_config(self):
        config = Hdf5DatasetSchemaConfig()
        assert isinstance(config, DatasetSchemaConfig)


@pytest.mark.unit
class TestLeRobotDatasetSchemaConfig:

    def test_dataset_path_required(self):
        config = LeRobotDatasetSchemaConfig()
        assert config.dataset_path == MISSING

    def test_inherits_from_dataset_schema_config(self):
        config = LeRobotDatasetSchemaConfig()
        assert isinstance(config, DatasetSchemaConfig)
