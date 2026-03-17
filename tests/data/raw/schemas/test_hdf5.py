"""Tests for versatil.data.raw.schemas.hdf5 module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import albumentations as A
import h5py
import numpy as np
import pytest

from versatil.data.raw.schemas.hdf5 import Hdf5DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


class ConcreteHdf5Schema(Hdf5DatasetSchema):
    """Minimal concrete subclass for testing the ABC."""

    def get_demo_names(self, hdf5_path: str) -> list[str]:
        return []

    def extract_episode(
        self,
        demo_group: h5py.Group,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        return {}


class TestHdf5DatasetSchemaInit:
    @pytest.mark.parametrize(
        "hdf5_paths",
        [
            ["/data/demo_01.hdf5", "/data/demo_02.hdf5"],
            ["/single/file.hdf5"],
        ],
        ids=["multiple_files", "single_file"],
    )
    def test_stores_hdf5_paths(
        self,
        hdf5_paths: list[str],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteHdf5Schema(
            hdf5_paths=hdf5_paths,
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.hdf5_paths == hdf5_paths

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

        schema = ConcreteHdf5Schema(
            hdf5_paths=["/data/file.hdf5"],
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.zarr_path == zarr_path

    def test_inherits_metadata_from_base(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteHdf5Schema(
            hdf5_paths=["/data/file.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.metadata is metadata

    @pytest.mark.parametrize(
        "dataset_type",
        ["libero", "robomimic"],
        ids=["libero", "robomimic"],
    )
    def test_inherits_dataset_type_from_base(
        self,
        dataset_type: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteHdf5Schema(
            hdf5_paths=["/data/file.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=dataset_type,
        )

        assert schema.dataset_type == dataset_type


class TestHdf5DatasetSchemaAbstract:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError, match="abstract method"):
            Hdf5DatasetSchema(
                hdf5_paths=["/data/file.hdf5"],
                zarr_path="/tmp/test.zarr",
                metadata=MagicMock(),
                dataset_type="test",
            )

    def test_get_demo_names_is_abstract(self):
        assert "get_demo_names" in Hdf5DatasetSchema.__abstractmethods__

    def test_extract_episode_is_abstract(self):
        assert "extract_episode" in Hdf5DatasetSchema.__abstractmethods__
