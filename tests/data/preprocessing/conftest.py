"""Preprocessing test fixtures shared across create_zarr test modules."""
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_camera_metadata_factory() -> Callable[..., MagicMock]:
    """Factory for creating mock CameraMetadata with configurable dimensions."""

    def factory(
        image_width: int = 256,
        image_height: int = 256,
    ) -> MagicMock:
        camera = MagicMock()
        camera.image_width = image_width
        camera.image_height = image_height
        return camera

    return factory


@pytest.fixture
def spec_factory() -> Callable[..., dict]:
    """Factory for creating zarr array spec dicts."""

    def factory(
        shape: tuple = (0, 7),
        chunks: tuple = (256, 7),
        dtype: str = "float32",
        needs_compressor: bool = True,
    ) -> dict:
        return {
            "shape": shape,
            "chunks": chunks,
            "dtype": dtype,
            "needs_compressor": needs_compressor,
        }

    return factory


@pytest.fixture
def mock_schema_factory() -> Callable[..., MagicMock]:
    """Factory for creating mock DatasetSchema with controlled specs and zarr_path."""

    def factory(
        specs: dict = None,
        zarr_path: str = "/tmp/test.zarr",
        cameras: dict = None,
        hdf5_paths: list[str] = None,
        demo_names_per_file: dict[str, list[str]] = None,
        total_episodes: int = 5,
        extract_return: dict = None,
    ) -> MagicMock:
        schema = MagicMock()
        schema.zarr_path = zarr_path
        schema.__class__.__name__ = "MockSchema"
        schema.get_zarr_array_specs.return_value = specs or {}
        schema.metadata = MagicMock()
        schema.metadata.cameras = cameras or {}
        schema.hdf5_paths = hdf5_paths or []
        schema.lerobot_metadata = MagicMock()
        schema.lerobot_metadata.get_total_episodes.return_value = total_episodes

        if demo_names_per_file is not None:
            schema.get_demo_names.side_effect = lambda path: demo_names_per_file[path]
        else:
            schema.get_demo_names.return_value = []

        if extract_return is not None:
            schema.extract_episode.return_value = extract_return

        return schema

    return factory