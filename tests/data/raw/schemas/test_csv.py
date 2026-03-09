"""Tests for versatil.data.raw.schemas.csv module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import albumentations as A
import numpy as np
import pandas as pd
import pytest

from versatil.data.raw.schemas.csv import CsvDatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


class ConcreteCsvSchema(CsvDatasetSchema):
    """Minimal concrete subclass for testing the ABC."""

    def extract_episode(
        self,
        episode: pd.DataFrame,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        return {}


class TestCsvDatasetSchemaInit:

    @pytest.mark.parametrize(
        "dataset_folders",
        [
            ["/data/episode_001", "/data/episode_002"],
            ["/single/folder"],
        ],
        ids=["multiple_folders", "single_folder"],
    )
    def test_stores_dataset_folders(
        self,
        dataset_folders: list[str],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteCsvSchema(
            dataset_folders=dataset_folders,
            zarr_path="/tmp/test.zarr",
            episode_filename="data.csv",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.dataset_folders == dataset_folders

    @pytest.mark.parametrize(
        "episode_filename",
        ["data.csv", "episode_data.tsv"],
        ids=["csv_format", "tsv_format"],
    )
    def test_stores_dataset_filename_from_episode_filename_param(
        self,
        episode_filename: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteCsvSchema(
            dataset_folders=["/data/folder"],
            zarr_path="/tmp/test.zarr",
            episode_filename=episode_filename,
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.dataset_filename == episode_filename

    @pytest.mark.parametrize(
        "zarr_path",
        ["/tmp/test.zarr", "/data/output.zarr"],
        ids=["tmp_path", "data_path"],
    )
    def test_inherits_zarr_path_from_base(
        self,
        zarr_path: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteCsvSchema(
            dataset_folders=["/data/folder"],
            zarr_path=zarr_path,
            episode_filename="data.csv",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.zarr_path == zarr_path

    def test_inherits_metadata_from_base(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteCsvSchema(
            dataset_folders=["/data/folder"],
            zarr_path="/tmp/test.zarr",
            episode_filename="data.csv",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.metadata is metadata

    @pytest.mark.parametrize(
        "dataset_type",
        ["bowel_retraction", "custom_csv"],
        ids=["bowel_retraction", "custom_csv"],
    )
    def test_inherits_dataset_type_from_base(
        self,
        dataset_type: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteCsvSchema(
            dataset_folders=["/data/folder"],
            zarr_path="/tmp/test.zarr",
            episode_filename="data.csv",
            metadata=metadata,
            dataset_type=dataset_type,
        )

        assert schema.dataset_type == dataset_type


class TestCsvDatasetSchemaAbstract:

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError, match="abstract method"):
            CsvDatasetSchema(
                dataset_folders=["/data/folder"],
                zarr_path="/tmp/test.zarr",
                episode_filename="data.csv",
                metadata=MagicMock(),
                dataset_type="test",
            )

    def test_extract_episode_is_abstract(self):
        assert "extract_episode" in CsvDatasetSchema.__abstractmethods__