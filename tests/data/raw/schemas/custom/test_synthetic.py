"""Tests for versatil.data.raw.schemas.custom.synthetic module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from versatil.data.constants import (
    Cameras,
    DatasetType,
)
from versatil.data.metadata import CameraMetadata
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.schemas.custom.synthetic import (
    ALLOWED_ACTION_KEYS,
    ALLOWED_CAMERAS,
    ALLOWED_OBSERVATION_KEYS,
    ALLOWED_POSITION_KEYS,
    SyntheticSchema,
)
from versatil.data.raw.zarr_meta import DatasetMetadata
from versatil.data.synthetic.constants import SyntheticTaskName


@pytest.fixture
def valid_synthetic_metadata(
    camera_metadata_factory: Callable[..., CameraMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
) -> DatasetMetadata:
    observations = {
        Cameras.AGENTVIEW.value: camera_metadata_factory(
            camera_key=Cameras.AGENTVIEW.value,
            image_height=64,
            image_width=64,
        ),
    }
    return dataset_metadata_factory(
        observations=observations,
        precomputed_actions={},
    )


@pytest.fixture
def metadata_mock_factory() -> Callable[..., MagicMock]:

    def factory(
        camera_keys: list[str] | None = None,
        position_observations: dict | None = None,
        precomputed_actions: dict | None = None,
        custom_observations: dict | None = None,
    ) -> MagicMock:
        mock_metadata = MagicMock(spec=DatasetMetadata)
        mock_metadata.get_camera_keys.return_value = (
            camera_keys if camera_keys is not None else []
        )
        mock_metadata.position_observations = (
            position_observations if position_observations is not None else {}
        )
        mock_metadata.precomputed_actions = (
            precomputed_actions if precomputed_actions is not None else {}
        )
        mock_metadata.custom_observations = (
            custom_observations if custom_observations is not None else {}
        )
        return mock_metadata

    return factory


@pytest.fixture
def synthetic_schema_factory(
    valid_synthetic_metadata: DatasetMetadata,
    tmp_path: Path,
) -> Callable[..., SyntheticSchema]:

    def factory(
        zarr_path: str | None = None,
        metadata: DatasetMetadata | None = None,
        dataset_type: str = DatasetType.SYNTHETIC.value,
        task_name: str = SyntheticTaskName.CORRIDOR_NAVIGATION.value,
        num_episodes: int = 50,
        seed: int = 7,
        image_size: int = 32,
        num_modes: int = 2,
        trajectory_length: int = 25,
        noise_std: float = 0.05,
        num_styles: int = 6,
        mode_weights: list[float] | None = None,
    ) -> SyntheticSchema:
        if zarr_path is None:
            zarr_path = str(tmp_path / "test.zarr")
        if metadata is None:
            metadata = valid_synthetic_metadata
        return SyntheticSchema(
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type=dataset_type,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            image_size=image_size,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=noise_std,
            num_styles=num_styles,
            mode_weights=mode_weights,
        )

    return factory


@pytest.mark.unit
class TestSyntheticSchemaInit:
    @pytest.mark.parametrize("num_episodes", [50, 200])
    @pytest.mark.parametrize("image_size", [32, 128])
    @pytest.mark.parametrize(
        "mode_weights",
        [None, [0.7, 0.1, 0.05, 0.05, 0.1]],
    )
    def test_stores_configuration(
        self,
        synthetic_schema_factory: Callable[..., SyntheticSchema],
        num_episodes: int,
        image_size: int,
        mode_weights: list[float] | None,
    ):
        task_name = SyntheticTaskName.CIRCLE.value
        seed = 123
        num_modes = 5
        trajectory_length = 40
        noise_std = 0.2
        num_styles = 8
        schema = synthetic_schema_factory(
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            image_size=image_size,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=noise_std,
            num_styles=num_styles,
            mode_weights=mode_weights,
        )
        assert schema.task_name == task_name
        assert schema.num_episodes == num_episodes
        assert schema.seed == seed
        assert schema.image_size == image_size
        assert schema.num_modes == num_modes
        assert schema.trajectory_length == trajectory_length
        assert schema.noise_std == noise_std
        assert schema.num_styles == num_styles
        assert schema.mode_weights == mode_weights

    @pytest.mark.parametrize(
        "dataset_type, expectation",
        [
            (DatasetType.SYNTHETIC.value, does_not_raise()),
            (
                DatasetType.TSO.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"SyntheticSchema only supports dataset_type='{DatasetType.SYNTHETIC.value}', "
                        f"got '{DatasetType.TSO.value}'"
                    ),
                ),
            ),
        ],
    )
    def test_dataset_type_validation(
        self,
        synthetic_schema_factory: Callable[..., SyntheticSchema],
        dataset_type: str,
        expectation: AbstractContextManager,
    ):
        with expectation:
            synthetic_schema_factory(dataset_type=dataset_type)


@pytest.mark.unit
class TestSyntheticSchemaValidation:
    def test_valid_metadata_passes(
        self,
        valid_synthetic_metadata: DatasetMetadata,
    ):
        SyntheticSchema._validate_metadata(valid_synthetic_metadata)

    @pytest.mark.parametrize(
        "field_name, expected_message",
        [
            (
                "camera_keys",
                f"Invalid cameras for SyntheticSchema: {{'invalid_camera'}}. "
                f"Allowed cameras: {ALLOWED_CAMERAS}",
            ),
            (
                "position_observations",
                f"Invalid position observation keys: {{'bad_position'}}. "
                f"Allowed: {ALLOWED_POSITION_KEYS}",
            ),
            (
                "precomputed_actions",
                f"Invalid precomputed action keys: {{'wrong_action'}}. "
                f"Allowed: {ALLOWED_ACTION_KEYS}",
            ),
            (
                "custom_observations",
                f"Invalid custom observation keys: {{'unknown_obs'}}. "
                f"Allowed: {ALLOWED_OBSERVATION_KEYS}",
            ),
        ],
    )
    def test_invalid_metadata_fields_raise(
        self,
        metadata_mock_factory: Callable[..., MagicMock],
        field_name: str,
        expected_message: str,
    ):
        match field_name:
            case "camera_keys":
                mock_metadata = metadata_mock_factory(camera_keys=["invalid_camera"])
            case "position_observations":
                mock_metadata = metadata_mock_factory(
                    position_observations={"bad_position": MagicMock()},
                )
            case "precomputed_actions":
                mock_metadata = metadata_mock_factory(
                    precomputed_actions={"wrong_action": MagicMock()},
                )
            case "custom_observations":
                mock_metadata = metadata_mock_factory(
                    custom_observations={"unknown_obs": MagicMock()},
                )
            case _:
                raise ValueError(f"Unknown field_name: {field_name}")
        with pytest.raises(
            ValueError,
            match=re.escape(expected_message),
        ):
            SyntheticSchema._validate_metadata(mock_metadata)

    def test_multiple_validation_errors_collected(
        self,
        metadata_mock_factory: Callable[..., MagicMock],
    ):
        mock_metadata = metadata_mock_factory(
            camera_keys=["bad_cam"],
            position_observations={"bad_pos": MagicMock()},
        )
        expected_header = "SyntheticSchema metadata validation failed:"
        with pytest.raises(
            ValueError, match=re.escape(expected_header)
        ) as exception_info:
            SyntheticSchema._validate_metadata(mock_metadata)
        error_message = str(exception_info.value)
        assert "Invalid cameras" in error_message
        assert "Invalid position observation keys" in error_message


@pytest.mark.unit
class TestGetCallbacks:
    @pytest.mark.parametrize(
        "task_name, image_size",
        [
            (SyntheticTaskName.CIRCLE.value, 64),
            (SyntheticTaskName.CORRIDOR_NAVIGATION.value, 48),
        ],
    )
    def test_returns_callback_with_schema_params(
        self,
        synthetic_schema_factory: Callable[..., SyntheticSchema],
        task_name: str,
        image_size: int,
    ):
        schema = synthetic_schema_factory(task_name=task_name, image_size=image_size)
        callbacks = schema.get_callbacks(experiment_config=MagicMock())
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert callback.task_name == task_name
        assert callback.image_size == image_size
        assert callback.num_rollouts == 50

    def test_base_dataset_schema_has_no_get_callbacks(self):
        assert not hasattr(DatasetSchema, "get_callbacks")


@pytest.mark.unit
class TestExtractEpisode:
    def test_extract_episode_raises_not_implemented(
        self,
        synthetic_schema_factory: Callable[..., SyntheticSchema],
    ):
        schema = synthetic_schema_factory()
        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "SyntheticSchema does not support extract_episode(). "
                "Use create_zarr_from_synthetic.create_replay_buffer_from_synthetic() instead."
            ),
        ):
            schema.extract_episode(
                episode_source=None,
                resizer=MagicMock(),
                depth_resizer=MagicMock(),
            )
