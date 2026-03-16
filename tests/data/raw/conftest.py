"""Raw data package test fixtures: metadata factories and resizer helpers."""
from collections.abc import Callable

import albumentations as A
import pytest

from versatil.data.constants import (
    CoordinateSystem,
    OrientationRepresentation,
)
from versatil.data.metadata import (
    OrientationActionMetadata,
    PositionActionMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.raw.zarr_meta import DatasetMetadata


@pytest.fixture
def precomputed_action_metadata_factory() -> Callable[..., PrecomputedActionMetadata]:
    """Factory for creating PrecomputedActionMetadata instances."""

    def factory(
        raw_data_column_keys: list[str] = None,
        storage_dimension: int = 7,
        prediction_dimension: int = 3,
        is_numerical: bool = True,
        needs_normalization: bool = True,
        dtype: str = "float32",
        slice_start: int = None,
        slice_end: int = None,
        requires_prediction_head: bool = True,
    ) -> PrecomputedActionMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["action_col"]
        return PrecomputedActionMetadata(
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            is_numerical=is_numerical,
            needs_normalization=needs_normalization,
            dtype=dtype,
            slice_start=slice_start,
            slice_end=slice_end,
            requires_prediction_head=requires_prediction_head,
        )

    return factory


@pytest.fixture
def dataset_metadata_factory() -> Callable[..., DatasetMetadata]:
    """Factory for creating DatasetMetadata instances."""

    def factory(
        observations: dict = None,
        precomputed_actions: dict = None,
    ) -> DatasetMetadata:
        if observations is None:
            observations = {}
        if precomputed_actions is None:
            precomputed_actions = {}
        return DatasetMetadata(
            observations=observations,
            precomputed_actions=precomputed_actions,
        )

    return factory


@pytest.fixture
def noop_resizer() -> A.NoOp:
    """No-op resizer for extract_episode calls that do not need resizing."""
    return A.NoOp()