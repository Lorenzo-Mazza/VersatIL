"""Tests for versatil.data.raw.schemas.base module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import albumentations as A
import numpy as np
import pytest

from versatil.data.constants import (
    Cameras,
    CoordinateSystem,
)
from versatil.data.metadata import (
    CameraMetadata,
    ObservationMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


class ConcreteDatasetSchema(DatasetSchema):
    """Minimal concrete subclass for testing the ABC."""

    def extract_episode(
        self,
        episode_source: object,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        return {}


class TestDatasetSchemaInit:

    @pytest.mark.parametrize(
        "zarr_path",
        ["/tmp/test.zarr", "/data/experiments/dataset.zarr"],
        ids=["tmp_path", "nested_path"],
    )
    def test_stores_zarr_path(
        self,
        zarr_path: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteDatasetSchema(
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.zarr_path == zarr_path

    def test_stores_metadata(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.metadata is metadata

    @pytest.mark.parametrize(
        "dataset_type",
        ["libero", "bowel_retraction", "metaworld"],
        ids=["libero", "bowel_retraction", "metaworld"],
    )
    def test_stores_dataset_type(
        self,
        dataset_type: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=dataset_type,
        )

        assert schema.dataset_type == dataset_type


class TestDatasetSchemaAbstract:

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError, match="abstract method"):
            DatasetSchema(
                zarr_path="/tmp/test.zarr",
                metadata=MagicMock(),
                dataset_type="test",
            )

    def test_extract_episode_is_abstract(self):
        assert "extract_episode" in DatasetSchema.__abstractmethods__


class TestGetRequiredZarrKeys:

    def test_delegates_to_metadata_get_all_keys(
        self,
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            "position": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            )
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        keys = schema.get_required_zarr_keys()

        assert set(keys) == {"position", "action"}

    def test_empty_metadata_returns_empty_list(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.get_required_zarr_keys() == []


class TestGetZarrArraySpecs:

    @pytest.mark.parametrize(
        "camera_key, image_height, image_width, channels",
        [
            (Cameras.LEFT.value, 480, 640, 3),
            (Cameras.RIGHT.value, 256, 256, 3),
            (Cameras.DEPTH.value, 128, 128, 1),
        ],
        ids=["left_640x480_rgb", "right_256x256_rgb", "depth_128x128_mono"],
    )
    def test_camera_observation_spec(
        self,
        camera_key: str,
        image_height: int,
        image_width: int,
        channels: int,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            "cam": camera_metadata_factory(
                camera_key=camera_key,
                image_height=image_height,
                image_width=image_width,
                channels=channels,
            )
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["cam"]["shape"] == (0, image_height, image_width, channels)
        assert specs["cam"]["chunks"] == (16, image_height, image_width, channels)
        assert specs["cam"]["needs_compressor"] is True

    @pytest.mark.parametrize(
        "dimension, frame",
        [
            (3, CoordinateSystem.ROBOT_BASE.value),
            (6, CoordinateSystem.CAMERA.value),
        ],
        ids=["3d_robot_base", "6d_camera"],
    )
    def test_numeric_observation_spec(
        self,
        dimension: int,
        frame: str,
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            "position": position_observation_metadata_factory(
                dimension=dimension, frame=frame
            )
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["position"]["shape"] == (0, dimension)
        assert specs["position"]["chunks"] == (256, dimension)
        assert specs["position"]["dtype"] == "float32"
        assert specs["position"]["needs_compressor"] is True

    def test_string_observation_needs_compressor_false(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        string_obs = ObservationMetadata(
            raw_data_column_keys=["language"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        metadata = dataset_metadata_factory(
            observations={"language": string_obs}, precomputed_actions={}
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["language"]["dtype"] == "str"
        assert specs["language"]["needs_compressor"] is False

    @pytest.mark.parametrize(
        "storage_dimension",
        [7, 14],
        ids=["7d_storage", "14d_storage"],
    )
    def test_precomputed_action_without_slice_uses_storage_dimension(
        self,
        storage_dimension: int,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            "action": precomputed_action_metadata_factory(
                storage_dimension=storage_dimension,
                prediction_dimension=3,
                slice_start=None,
                slice_end=None,
            )
        }
        metadata = dataset_metadata_factory(
            observations={}, precomputed_actions=actions
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["action"]["shape"] == (0, storage_dimension)
        assert specs["action"]["chunks"] == (256, storage_dimension)
        assert specs["action"]["needs_compressor"] is True

    @pytest.mark.parametrize(
        "slice_start, slice_end, expected_dim",
        [
            (0, 3, 3),
            (3, 7, 4),
            (0, 7, 7),
        ],
        ids=["first_3_elements", "middle_4_elements", "full_range"],
    )
    def test_precomputed_action_with_slice_uses_slice_range(
        self,
        slice_start: int,
        slice_end: int,
        expected_dim: int,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            "action": precomputed_action_metadata_factory(
                storage_dimension=expected_dim,
                prediction_dimension=expected_dim,
                slice_start=slice_start,
                slice_end=slice_end,
            )
        }
        metadata = dataset_metadata_factory(
            observations={}, precomputed_actions=actions
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["action"]["shape"] == (0, expected_dim)
        assert specs["action"]["chunks"] == (256, expected_dim)

    def test_mixed_observations_and_actions_returns_all_specs(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            "camera": camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
                channels=3,
            ),
            "position": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert set(specs.keys()) == {"camera", "position", "action"}

    def test_empty_metadata_returns_empty_specs(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        assert schema.get_zarr_array_specs() == {}

    @pytest.mark.parametrize(
        "dtype",
        ["float32", "float64", "int32"],
        ids=["float32", "float64", "int32"],
    )
    def test_precomputed_action_preserves_dtype(
        self,
        dtype: str,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            "action": precomputed_action_metadata_factory(
                storage_dimension=3,
                prediction_dimension=3,
                dtype=dtype,
            )
        }
        metadata = dataset_metadata_factory(
            observations={}, precomputed_actions=actions
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["action"]["dtype"] == dtype

    def test_camera_observation_preserves_dtype(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            "cam": camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
                dtype="uint8",
            )
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )
        schema = ConcreteDatasetSchema(
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type="test",
        )

        specs = schema.get_zarr_array_specs()

        assert specs["cam"]["dtype"] == "uint8"