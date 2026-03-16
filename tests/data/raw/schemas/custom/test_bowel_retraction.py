"""Tests for versatil.data.raw.schemas.custom.bowel_retraction module."""
from collections.abc import Callable
from unittest.mock import patch

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest

from versatil.data.constants import (
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    DatasetType,
    GripperType,
    ObsKey,
    ProprioKey,
)
from versatil.data.metadata import (
    CameraMetadata,
    GripperObservationMetadata,
    ObservationMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.raw.schemas.custom.bowel_retraction import (
    BOWEL_RETRACTION_EPISODE_FILENAME,
    BOWEL_RETRACTION_GRIPPER_COL,
    BOWEL_RETRACTION_LANGUAGE_COL,
    BOWEL_RETRACTION_LEFT_IMAGE_KEY,
    BOWEL_RETRACTION_PHASE_COL,
    BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY,
    BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY,
    BOWEL_RETRACTION_RIGHT_IMAGE_KEY,
    BowelRetractionSchema,
)
from versatil.data.raw.zarr_meta import DatasetMetadata


@pytest.fixture
def valid_bowel_retraction_metadata(
    camera_metadata_factory: Callable[..., CameraMetadata],
    position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
    gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
) -> DatasetMetadata:
    """Minimal valid metadata that passes BowelRetraction validation."""
    observations = {
        Cameras.LEFT.value: camera_metadata_factory(
            camera_key=Cameras.LEFT.value,
            image_height=480,
            image_width=640,
        ),
        Cameras.RIGHT.value: camera_metadata_factory(
            camera_key=Cameras.RIGHT.value,
            image_height=480,
            image_width=640,
        ),
        ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
            dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
        ),
        ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
            dimension=1,
            raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
        ),
    }
    return dataset_metadata_factory(observations=observations, precomputed_actions={})


class TestBowelRetractionSchemaInit:

    def test_wrong_dataset_type_raises(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        with pytest.raises(ValueError, match="only supports dataset_type"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=valid_bowel_retraction_metadata,
                dataset_type="wrong_type",
            )

    def test_valid_init_with_minimal_metadata(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.zarr_path == "/tmp/test.zarr"
        assert schema.metadata is valid_bowel_retraction_metadata
        assert schema.dataset_type == DatasetType.TSO.value

    def test_sets_episode_filename(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.dataset_filename == BOWEL_RETRACTION_EPISODE_FILENAME

    def test_sets_use_rectified_images_true(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.use_rectified_images is True

    def test_stores_dataset_folders(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        folders = ["/data/ep1", "/data/ep2"]

        schema = BowelRetractionSchema(
            dataset_folders=folders,
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.dataset_folders == folders


class TestBowelRetractionValidateMetadata:

    def test_invalid_camera_key_raises(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            "wrist": camera_metadata_factory(
                camera_key=Cameras.WRIST.value, image_height=480, image_width=640
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="Invalid cameras"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    @pytest.mark.parametrize(
        "present_camera, missing_camera",
        [
            (Cameras.RIGHT, Cameras.LEFT),
            (Cameras.LEFT, Cameras.RIGHT),
        ],
        ids=["missing_left", "missing_right"],
    )
    def test_missing_camera_raises(
        self,
        present_camera: Cameras,
        missing_camera: Cameras,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            present_camera.value: camera_metadata_factory(
                camera_key=present_camera.value, image_height=480, image_width=640
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="Missing"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    def test_invalid_position_observation_key_raises(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            "invalid_proprio": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="Invalid proprioceptive"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    @pytest.mark.parametrize(
        "proprio_key, wrong_frame",
        [
            (ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value, CoordinateSystem.CAMERA.value),
            (ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value, CoordinateSystem.ROBOT_BASE.value),
        ],
        ids=["robot_frame_with_camera_frame", "camera_frame_with_robot_frame"],
    )
    def test_position_with_wrong_frame_raises(
        self,
        proprio_key: str,
        wrong_frame: str,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            proprio_key: position_observation_metadata_factory(
                dimension=3, frame=wrong_frame
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="must have frame"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    @pytest.mark.parametrize(
        "key, frame",
        [
            (ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value, CoordinateSystem.ROBOT_BASE.value),
            (ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value, CoordinateSystem.CAMERA.value),
        ],
        ids=["robot_frame_correct", "camera_frame_correct"],
    )
    def test_position_with_correct_frame_succeeds(
        self,
        key: str,
        frame: str,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            key: position_observation_metadata_factory(dimension=3, frame=frame),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.metadata is metadata

    def test_orientation_observations_present_raises(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        orientation_observation_metadata_factory: Callable,
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            "orientation": orientation_observation_metadata_factory(
                dimension=1, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="does not support orientation"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    def test_gripper_not_binary_raises(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.CONTINUOUS.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="binary gripper"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    def test_gripper_wrong_column_keys_raises(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=480, image_width=640
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=["wrong_column"],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="gripper source column"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    def test_custom_observation_without_language_key_logs_warning(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        custom_observation = ObservationMetadata(
            raw_data_column_keys=["custom_col"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=False,
        )
        # Rebuild metadata adding a custom obs to the existing valid observations
        observations = dict(valid_bowel_retraction_metadata.observations)
        observations["some_custom"] = custom_observation
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with patch(
            "versatil.data.raw.schemas.custom.bowel_retraction.logging"
        ) as mock_logging:
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

        mock_logging.warning.assert_called_once()

    def test_language_observation_with_wrong_column_raises(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        language_observation = ObservationMetadata(
            raw_data_column_keys=["wrong_language_col"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = dict(valid_bowel_retraction_metadata.observations)
        observations[ObsKey.LANGUAGE.value] = language_observation
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError, match="language source column"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    def test_language_observation_with_correct_column_succeeds(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        language_observation = ObservationMetadata(
            raw_data_column_keys=[BOWEL_RETRACTION_LANGUAGE_COL],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = dict(valid_bowel_retraction_metadata.observations)
        observations[ObsKey.LANGUAGE.value] = language_observation
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.metadata is metadata

    def test_custom_action_without_phase_label_logs_warning(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            "some_custom_action": precomputed_action_metadata_factory(
                storage_dimension=1,
                prediction_dimension=1,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=dict(valid_bowel_retraction_metadata.observations),
            precomputed_actions=actions,
        )

        with patch(
            "versatil.data.raw.schemas.custom.bowel_retraction.logging"
        ) as mock_logging:
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

        mock_logging.warning.assert_called_once()

    def test_phase_label_action_with_wrong_column_raises(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            ObsKey.PHASE_LABEL.value: precomputed_action_metadata_factory(
                storage_dimension=1,
                prediction_dimension=1,
                raw_data_column_keys=["wrong_phase_col"],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=dict(valid_bowel_retraction_metadata.observations),
            precomputed_actions=actions,
        )

        with pytest.raises(ValueError, match="phase label source column"):
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

    def test_phase_label_action_with_correct_column_succeeds(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            ObsKey.PHASE_LABEL.value: precomputed_action_metadata_factory(
                storage_dimension=1,
                prediction_dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_PHASE_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=dict(valid_bowel_retraction_metadata.observations),
            precomputed_actions=actions,
        )

        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert schema.metadata is metadata

    def test_multiple_errors_accumulated_in_single_raise(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=480, image_width=640
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.CONTINUOUS.value,
                dimension=1,
                raw_data_column_keys=["wrong"],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )

        with pytest.raises(ValueError) as exc_info:
            BowelRetractionSchema(
                dataset_folders=["/data/ep1"],
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type=DatasetType.TSO.value,
            )

        error_message = str(exc_info.value)
        assert error_message.count("  - ") >= 2

    def test_no_custom_observations_skips_language_validation(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert valid_bowel_retraction_metadata.custom_observations == {}
        assert schema.metadata is valid_bowel_retraction_metadata

    def test_no_custom_actions_skips_phase_validation(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        assert valid_bowel_retraction_metadata.custom_actions == {}
        assert schema.metadata is valid_bowel_retraction_metadata


class TestBowelRetractionExtractEpisode:

    @pytest.fixture
    def bowel_retraction_episode_factory(
        self,
        rng: np.random.Generator,
    ) -> Callable:
        """Factory for creating test DataFrames for BowelRetraction extract_episode."""

        def factory(
            num_rows: int = 5,
            extra_columns: dict = None,
        ) -> pd.DataFrame:
            data = {
                "x": rng.standard_normal(num_rows).astype(np.float32),
                "y": rng.standard_normal(num_rows).astype(np.float32),
                "z": rng.standard_normal(num_rows).astype(np.float32),
                BOWEL_RETRACTION_GRIPPER_COL: rng.integers(0, 2, size=num_rows).astype(np.float32),
                BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY: [
                    f"/img/framesLeftRectified/frame_{i:04d}.png" for i in range(num_rows)
                ],
                BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY: [
                    f"/img/framesRightRectified/frame_{i:04d}.png" for i in range(num_rows)
                ],
            }
            if extra_columns:
                data.update(extra_columns)
            return pd.DataFrame(data)

        return factory

    @pytest.fixture
    def mock_cv2_factory(self, rng: np.random.Generator) -> Callable:
        """Factory that returns a context manager patching cv2 with a mock image."""

        def factory(image_height: int = 480, image_width: int = 640):
            mock_image = rng.integers(
                0, 255, size=(image_height, image_width, 3), dtype=np.uint8
            )

            class _Cv2Patcher:
                def __init__(self):
                    self.mock_image = mock_image

                def __enter__(self):
                    self._patcher = patch(
                        "versatil.data.raw.schemas.custom.bowel_retraction.cv2"
                    )
                    mock_cv2 = self._patcher.start()
                    mock_cv2.imread.return_value = self.mock_image
                    mock_cv2.cvtColor.return_value = self.mock_image
                    mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
                    return mock_cv2

                def __exit__(self, exception_type, exception_value, traceback):
                    self._patcher.stop()

            return _Cv2Patcher()

        return factory

    def test_extracts_non_camera_observation_from_dataframe(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        bowel_retraction_episode_factory: Callable,
        mock_cv2_factory: Callable,
        noop_resizer: A.NoOp,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )
        episode = bowel_retraction_episode_factory(num_rows=5)

        with mock_cv2_factory():
            data = schema.extract_episode(
                episode=episode, resizer=noop_resizer, depth_resizer=noop_resizer
            )

        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        assert proprio_key in data
        assert data[proprio_key].shape == (5, 3)
        expected_values = episode[["x", "y", "z"]].values.astype(np.float32)
        np.testing.assert_array_almost_equal(data[proprio_key], expected_values)

    def test_skips_camera_metadata_in_observation_loop(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
        bowel_retraction_episode_factory: Callable,
        mock_cv2_factory: Callable,
        noop_resizer: A.NoOp,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )
        episode = bowel_retraction_episode_factory(num_rows=3)

        with mock_cv2_factory():
            data = schema.extract_episode(
                episode=episode, resizer=noop_resizer, depth_resizer=noop_resizer
            )

        assert Cameras.LEFT.value in data
        assert Cameras.RIGHT.value in data

    def test_extracts_precomputed_actions_from_dataframe(
        self,
        rng: np.random.Generator,
        valid_bowel_retraction_metadata: DatasetMetadata,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        bowel_retraction_episode_factory: Callable,
        mock_cv2_factory: Callable,
        noop_resizer: A.NoOp,
    ):
        actions = {
            ObsKey.PHASE_LABEL.value: precomputed_action_metadata_factory(
                storage_dimension=1,
                prediction_dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_PHASE_COL],
                dtype="int32",
                is_numerical=True,
                needs_normalization=False,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=dict(valid_bowel_retraction_metadata.observations),
            precomputed_actions=actions,
        )
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.TSO.value,
        )
        episode = bowel_retraction_episode_factory(
            num_rows=4,
            extra_columns={BOWEL_RETRACTION_PHASE_COL: rng.integers(0, 3, size=4)},
        )

        with mock_cv2_factory():
            data = schema.extract_episode(
                episode=episode, resizer=noop_resizer, depth_resizer=noop_resizer
            )

        assert ObsKey.PHASE_LABEL.value in data
        assert data[ObsKey.PHASE_LABEL.value].dtype == np.int32
        assert data[ObsKey.PHASE_LABEL.value].shape == (4, 1)

    def test_extracts_depth_images_from_npy(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        noop_resizer: A.NoOp,
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value, image_height=64, image_width=64
            ),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value, image_height=64, image_width=64
            ),
            Cameras.DEPTH.value: camera_metadata_factory(
                camera_key=Cameras.DEPTH.value,
                image_height=64,
                image_width=64,
                channels=1,
                dtype="float32",
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
                raw_data_column_keys=[BOWEL_RETRACTION_GRIPPER_COL],
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions={}
        )
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.TSO.value,
        )
        episode = pd.DataFrame({
            "x": rng.standard_normal(2).astype(np.float32),
            "y": rng.standard_normal(2).astype(np.float32),
            "z": rng.standard_normal(2).astype(np.float32),
            BOWEL_RETRACTION_GRIPPER_COL: rng.integers(0, 2, size=2).astype(np.float32),
            BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY: [
                "/data/framesLeftRectified/frame_0001.png",
                "/data/framesLeftRectified/frame_0002.png",
            ],
            BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY: [
                "/data/framesRightRectified/frame_0001.png",
                "/data/framesRightRectified/frame_0002.png",
            ],
        })
        mock_rgb = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        mock_depth = rng.standard_normal((64, 64)).astype(np.float32)

        with patch(
            "versatil.data.raw.schemas.custom.bowel_retraction.cv2"
        ) as mock_cv2:
            mock_cv2.imread.return_value = mock_rgb
            mock_cv2.cvtColor.return_value = mock_rgb
            mock_cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
            with patch(
                "versatil.data.raw.schemas.custom.bowel_retraction.np.load",
                return_value=mock_depth,
            ):
                data = schema.extract_episode(
                    episode=episode, resizer=noop_resizer, depth_resizer=noop_resizer
                )

        assert Cameras.DEPTH.value in data
        assert data[Cameras.DEPTH.value].shape == (2, 64, 64, 1)


class TestBowelRetractionGetRgbColumn:

    @pytest.mark.parametrize(
        "use_rectified, camera, expected_key",
        [
            (True, Cameras.LEFT.value, BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY),
            (False, Cameras.LEFT.value, BOWEL_RETRACTION_LEFT_IMAGE_KEY),
            (True, Cameras.RIGHT.value, BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY),
            (False, Cameras.RIGHT.value, BOWEL_RETRACTION_RIGHT_IMAGE_KEY),
        ],
        ids=[
            "left_rectified",
            "left_non_rectified",
            "right_rectified",
            "right_non_rectified",
        ],
    )
    def test_returns_correct_column_key(
        self,
        use_rectified: bool,
        camera: str,
        expected_key: str,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )
        schema.use_rectified_images = use_rectified

        assert schema._get_rgb_column(camera) == expected_key

    def test_unknown_camera_raises(
        self,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )

        with pytest.raises(ValueError, match="Unknown RGB camera"):
            schema._get_rgb_column(Cameras.DEPTH.value)


class TestBowelRetractionComputeDepthPath:

    @pytest.mark.parametrize(
        "use_rectified, base_path, expected_path",
        [
            (
                True,
                "/data/framesLeftRectified/frame_0001.png",
                "/data/depth/frame_depth_0001.npy",
            ),
            (
                False,
                "/data/framesLeft/frame_0001.png",
                "/data/depth/frame_depth_0001.npy",
            ),
        ],
        ids=["rectified", "non_rectified"],
    )
    def test_replaces_directory_and_filename(
        self,
        use_rectified: bool,
        base_path: str,
        expected_path: str,
        valid_bowel_retraction_metadata: DatasetMetadata,
    ):
        schema = BowelRetractionSchema(
            dataset_folders=["/data/ep1"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_bowel_retraction_metadata,
            dataset_type=DatasetType.TSO.value,
        )
        schema.use_rectified_images = use_rectified

        assert schema._compute_depth_path(base_path) == expected_path