"""Tests for versatil.explainability.sources.schema_paths module."""

import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from versatil.explainability.sources.schema_paths import (
    OFFLINE_DATASET_ZARR_NAME,
    resolve_dataset_schema_for_explanation,
)


class TestResolveDatasetSchemaForExplanation:
    def test_none_override_returns_original_schema(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        schema = csv_schema_factory()

        result = resolve_dataset_schema_for_explanation(
            schema=schema,
            data_path_override=None,
            zarr_cache_directory=tmp_path,
        )

        assert result is schema

    def test_csv_raw_override_clones_schema_and_replaces_dataset_folders(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        raw_path = tmp_path / "raw_csv"
        raw_path.mkdir()
        cache_directory = tmp_path / "cache"
        schema = csv_schema_factory(dataset_folders=[str(tmp_path / "training_raw")])

        result = resolve_dataset_schema_for_explanation(
            schema=schema,
            data_path_override=str(raw_path),
            zarr_cache_directory=cache_directory,
        )

        assert result is not schema
        assert result.dataset_folders == [str(raw_path)]
        assert result.zarr_path == str(cache_directory / OFFLINE_DATASET_ZARR_NAME)
        assert schema.dataset_folders == [str(tmp_path / "training_raw")]

    def test_hdf5_raw_override_clones_schema_and_replaces_hdf5_paths(
        self,
        hdf5_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        raw_path = tmp_path / "episode.hdf5"
        raw_path.touch()
        cache_directory = tmp_path / "cache"
        schema = hdf5_schema_factory(hdf5_paths=[str(tmp_path / "training.hdf5")])

        result = resolve_dataset_schema_for_explanation(
            schema=schema,
            data_path_override=str(raw_path),
            zarr_cache_directory=cache_directory,
        )

        assert result is not schema
        assert result.hdf5_paths == [str(raw_path)]
        assert result.zarr_path == str(cache_directory / OFFLINE_DATASET_ZARR_NAME)
        assert schema.hdf5_paths == [str(tmp_path / "training.hdf5")]

    def test_lerobot_raw_override_replaces_dataset_path_and_metadata(
        self,
        lerobot_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        raw_path = tmp_path / "lerobot_raw"
        raw_path.mkdir()
        cache_directory = tmp_path / "cache"
        schema = lerobot_schema_factory(dataset_path=tmp_path / "training_raw")

        with patch(
            "versatil.explainability.sources.schema_paths.LeRobotDatasetMetadataV30"
        ) as metadata_class:
            result = resolve_dataset_schema_for_explanation(
                schema=schema,
                data_path_override=str(raw_path),
                zarr_cache_directory=cache_directory,
            )

        assert result is not schema
        assert result.dataset_path == raw_path
        assert result.zarr_path == str(cache_directory / OFFLINE_DATASET_ZARR_NAME)
        metadata_class.assert_called_once_with(dataset_path=raw_path)
        assert result.lerobot_metadata is metadata_class.return_value
        assert schema.dataset_path == tmp_path / "training_raw"

    def test_zarr_override_clones_schema_and_replaces_zarr_path(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        zarr_path = tmp_path / "offline.zarr"
        zarr_path.mkdir()
        schema = csv_schema_factory(
            dataset_folders=[str(tmp_path / "training_raw")],
            zarr_path=str(tmp_path / "training.zarr"),
        )

        result = resolve_dataset_schema_for_explanation(
            schema=schema,
            data_path_override=str(zarr_path),
            zarr_cache_directory=tmp_path / "cache",
        )

        assert result is not schema
        assert result.zarr_path == str(zarr_path)
        assert result.dataset_folders == [str(tmp_path / "training_raw")]
        assert schema.zarr_path == str(tmp_path / "training.zarr")

    @pytest.mark.parametrize(
        "data_path_override",
        [
            [],
            [""],
        ],
        ids=["empty_list", "blank_entry"],
    )
    def test_empty_override_raises(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        data_path_override: list[str],
        tmp_path: Path,
    ) -> None:
        expected_message = "data_path_override must contain at least one path."

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=csv_schema_factory(),
                data_path_override=data_path_override,
                zarr_cache_directory=tmp_path,
            )

    def test_missing_raw_path_raises(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        missing_path = tmp_path / "missing_raw"
        expected_message = f"data_path_override path does not exist: {missing_path}"

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=csv_schema_factory(),
                data_path_override=str(missing_path),
                zarr_cache_directory=tmp_path,
            )

    def test_missing_zarr_override_raises(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        zarr_path = tmp_path / "missing.zarr"
        expected_message = (
            "data_path_override points to a .zarr path that does not exist: "
            f"{zarr_path}"
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=csv_schema_factory(),
                data_path_override=str(zarr_path),
                zarr_cache_directory=tmp_path,
            )

    @pytest.mark.parametrize(
        "data_path_override",
        [
            ["raw", "offline.zarr"],
            ["first.zarr", "second.zarr"],
        ],
        ids=["mixed_raw_and_zarr", "multiple_zarr"],
    )
    def test_multiple_paths_with_zarr_raises(
        self,
        csv_schema_factory: Callable[..., MagicMock],
        data_path_override: list[str],
        tmp_path: Path,
    ) -> None:
        resolved_paths = []
        for path_name in data_path_override:
            path = tmp_path / path_name
            path.mkdir()
            resolved_paths.append(str(path))
        expected_message = (
            "data_path_override accepts a single .zarr path, or one or more raw "
            "paths for the checkpoint schema format."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=csv_schema_factory(),
                data_path_override=resolved_paths,
                zarr_cache_directory=tmp_path,
            )

    def test_lerobot_multiple_raw_paths_raises(
        self,
        lerobot_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        raw_paths = [tmp_path / "first", tmp_path / "second"]
        for raw_path in raw_paths:
            raw_path.mkdir()
        expected_message = (
            "data_path_override for LeRobotDatasetSchemaV30 must be a single "
            "dataset root."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=lerobot_schema_factory(),
                data_path_override=[str(raw_path) for raw_path in raw_paths],
                zarr_cache_directory=tmp_path,
            )

    def test_synthetic_raw_override_raises(
        self,
        synthetic_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        raw_path = tmp_path / "raw_synthetic"
        raw_path.mkdir()
        expected_message = (
            "data_path_override cannot point to raw files for SyntheticSchema. "
            "Pass an existing .zarr path instead."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=synthetic_schema_factory(),
                data_path_override=str(raw_path),
                zarr_cache_directory=tmp_path,
            )

    def test_unsupported_schema_type_raises(
        self,
        tmp_path: Path,
    ) -> None:
        raw_path = tmp_path / "raw"
        raw_path.mkdir()
        schema = MagicMock()
        schema.zarr_path = str(tmp_path / "training.zarr")
        expected_message = (
            "data_path_override is unsupported for schema type MagicMock."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_dataset_schema_for_explanation(
                schema=schema,
                data_path_override=str(raw_path),
                zarr_cache_directory=tmp_path,
            )
