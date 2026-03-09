"""Tests for versatil.data.raw.schemas.custom.libero module."""
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from versatil.data.constants import (
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
from versatil.data.raw.schemas.custom.libero import LiberoSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


@pytest.fixture
def valid_libero_metadata(
    camera_metadata_factory: Callable[..., CameraMetadata],
    position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
    gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
) -> DatasetMetadata:
    """Minimal valid metadata for LiberoSchema."""
    observations = {
        Cameras.LEFT.value: camera_metadata_factory(
            camera_key=Cameras.LEFT.value,
            image_height=128,
            image_width=128,
        ),
        ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
            dimension=3,
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["ee_pos"],
        ),
        ProprioKey.GRIPPER_STATE.value: gripper_observation_metadata_factory(
            gripper_type=GripperType.CONTINUOUS.value,
            dimension=2,
            raw_data_column_keys=["gripper_states"],
        ),
    }
    actions = {
        "action": precomputed_action_metadata_factory(
            raw_data_column_keys=["actions"],
            storage_dimension=7,
            prediction_dimension=3,
            slice_start=0,
            slice_end=3,
        ),
    }
    return dataset_metadata_factory(
        observations=observations, precomputed_actions=actions
    )


@pytest.fixture
def libero_demo_group_factory(rng: np.random.Generator) -> Callable:
    """Factory for creating mock HDF5 demo groups for LiberoSchema tests.

    Args:
        obs_arrays: Dict mapping obs keys to numpy arrays.
        actions_array: Numpy array for actions data.
        filename: HDF5 filename for language extraction.
    """

    def factory(
        obs_arrays: dict[str, np.ndarray],
        actions_array: np.ndarray,
        filename: str = "/data/task_demo.hdf5",
    ) -> MagicMock:
        obs_group = {}
        for key, value in obs_arrays.items():
            arr_mock = MagicMock()
            arr_mock.__getitem__ = MagicMock(return_value=value)
            arr_mock.astype = MagicMock(
                return_value=value.astype(str) if value.dtype.kind in ("U", "S", "O") else value
            )
            obs_group[key] = arr_mock

        obs_mock = MagicMock()
        obs_mock.__getitem__ = MagicMock(side_effect=lambda k: obs_group[k])
        obs_mock.__contains__ = MagicMock(side_effect=lambda k: k in obs_group)
        obs_mock.keys = MagicMock(return_value=list(obs_group.keys()))
        obs_mock.__iter__ = MagicMock(return_value=iter(obs_group.keys()))

        actions_mock = MagicMock()
        actions_mock.__getitem__ = MagicMock(return_value=actions_array)
        actions_mock.shape = actions_array.shape

        demo_dict = {"obs": obs_mock, "actions": actions_mock}
        demo_group = MagicMock()
        demo_group.__getitem__ = MagicMock(side_effect=lambda k: demo_dict[k])
        demo_group.__contains__ = MagicMock(side_effect=lambda k: k in demo_dict)
        demo_group.file.filename = filename

        return demo_group

    return factory


class TestLiberoSchemaInit:

    def test_wrong_dataset_type_raises(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        with pytest.raises(ValueError, match="only supports dataset_type"):
            LiberoSchema(
                hdf5_paths=["/data/task.hdf5"],
                zarr_path="/tmp/test.zarr",
                metadata=valid_libero_metadata,
                dataset_type="wrong_type",
            )

    def test_valid_init(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        assert schema.zarr_path == "/tmp/test.zarr"
        assert schema.metadata is valid_libero_metadata
        assert schema.dataset_type == DatasetType.LIBERO.value

    def test_sets_obs_group_path(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        assert schema.obs_group_path == "obs"

    def test_sets_actions_key(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        assert schema.actions_key == "actions"

    def test_sets_extract_language_from_filename_true(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        assert schema.extract_language_from_filename is True

    @pytest.mark.parametrize(
        "hdf5_paths",
        [
            ["/data/task1.hdf5"],
            ["/data/task1.hdf5", "/data/task2.hdf5"],
        ],
        ids=["single_file", "multiple_files"],
    )
    def test_stores_hdf5_paths(
        self,
        hdf5_paths: list[str],
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=hdf5_paths,
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        assert schema.hdf5_paths == hdf5_paths


class TestLiberoSchemaGetDemoNames:

    def test_returns_list_of_demo_keys(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )
        mock_data_group = MagicMock()
        mock_data_group.keys.return_value = ["demo_0", "demo_1", "demo_2"]

        with patch(
            "versatil.data.raw.schemas.custom.libero.h5py.File"
        ) as mock_file:
            mock_file.return_value.__enter__ = MagicMock(return_value={"data": mock_data_group})
            mock_file.return_value.__exit__ = MagicMock(return_value=False)

            result = schema.get_demo_names("/data/task.hdf5")

        assert result == ["demo_0", "demo_1", "demo_2"]


class TestLiberoSchemaExtractEpisode:

    def test_skips_camera_metadata_in_obs_loop(
        self,
        rng: np.random.Generator,
        valid_libero_metadata: DatasetMetadata,
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )
        demo_group = libero_demo_group_factory(
            obs_arrays={
                Cameras.LEFT.value: rng.integers(0, 255, size=(5, 128, 128, 3), dtype=np.uint8),
                "ee_pos": rng.standard_normal((5, 3)).astype(np.float32),
                "gripper_states": rng.integers(0, 2, size=(5, 2)).astype(np.float32),
            },
            actions_array=rng.standard_normal((5, 7)).astype(np.float32),
            filename="/data/pick_up_the_bowl_demo.hdf5",
        )

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        assert Cameras.LEFT.value in data
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in data

    def test_skips_language_when_extract_from_filename(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        language_observation = ObservationMetadata(
            raw_data_column_keys=["language"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = {
            ObsKey.LANGUAGE.value: language_observation,
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=3,
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["ee_pos"],
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=3,
                slice_start=0,
                slice_end=3,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        demo_group = libero_demo_group_factory(
            obs_arrays={
                Cameras.LEFT.value: rng.integers(0, 255, size=(3, 64, 64, 3), dtype=np.uint8),
                "ee_pos": rng.standard_normal((3, 3)).astype(np.float32),
            },
            actions_array=rng.standard_normal((3, 7)).astype(np.float32),
            filename="/data/pick_up_the_bowl_demo.hdf5",
        )

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        assert ObsKey.LANGUAGE.value in data
        assert data[ObsKey.LANGUAGE.value][0, 0] == "pick up the bowl"
        assert data[ObsKey.LANGUAGE.value].shape == (3, 1)

    def test_extracts_numeric_observation_concatenating_columns(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=6,
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["ee_pos", "ee_ori"],
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=7,
                slice_start=0,
                slice_end=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        mock_position = rng.standard_normal((4, 3)).astype(np.float32)
        mock_orientation = rng.standard_normal((4, 3)).astype(np.float32)
        demo_group = libero_demo_group_factory(
            obs_arrays={
                "ee_pos": mock_position,
                "ee_ori": mock_orientation,
                Cameras.LEFT.value: rng.integers(0, 255, size=(4, 64, 64, 3), dtype=np.uint8),
            },
            actions_array=rng.standard_normal((4, 7)).astype(np.float32),
        )

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        assert data[proprio_key].shape == (4, 6)
        np.testing.assert_array_almost_equal(
            data[proprio_key],
            np.concatenate([mock_position, mock_orientation], axis=-1),
        )

    def test_extracts_precomputed_actions_with_slicing(
        self,
        rng: np.random.Generator,
        valid_libero_metadata: DatasetMetadata,
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        mock_actions = rng.standard_normal((5, 7)).astype(np.float32)
        demo_group = libero_demo_group_factory(
            obs_arrays={
                Cameras.LEFT.value: rng.integers(0, 255, size=(5, 128, 128, 3), dtype=np.uint8),
                "ee_pos": rng.standard_normal((5, 3)).astype(np.float32),
                "gripper_states": rng.integers(0, 2, size=(5, 2)).astype(np.float32),
            },
            actions_array=mock_actions,
        )

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        assert data["action"].shape == (5, 3)
        np.testing.assert_array_almost_equal(
            data["action"], mock_actions[:, 0:3]
        )

    @pytest.mark.parametrize(
        "camera_key, image_shape, expected_dtype",
        [
            (Cameras.LEFT.value, (3, 64, 64, 3), np.uint8),
            (Cameras.DEPTH.value, (3, 64, 64), np.float32),
        ],
        ids=["rgb_camera", "depth_camera"],
    )
    def test_extracts_camera_with_correct_dtype(
        self,
        camera_key: str,
        image_shape: tuple,
        expected_dtype: type,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        channels = 1 if camera_key == Cameras.DEPTH.value else 3
        dtype = "float32" if camera_key == Cameras.DEPTH.value else "uint8"
        observations = {
            camera_key: camera_metadata_factory(
                camera_key=camera_key,
                image_height=64,
                image_width=64,
                channels=channels,
                dtype=dtype,
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=7,
                slice_start=0,
                slice_end=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        if expected_dtype == np.uint8:
            mock_images = rng.integers(0, 255, size=image_shape, dtype=np.uint8)
        else:
            mock_images = rng.standard_normal(image_shape).astype(np.float32)

        demo_group = libero_demo_group_factory(
            obs_arrays={camera_key: mock_images},
            actions_array=rng.standard_normal((image_shape[0], 7)).astype(np.float32),
        )

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        assert camera_key in data
        assert data[camera_key].dtype == expected_dtype
        assert data[camera_key].shape[0] == image_shape[0]
        np.testing.assert_array_equal(data[camera_key][0], mock_images[0])

    def test_missing_camera_key_raises(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=7,
                slice_start=0,
                slice_end=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        # Empty obs — camera key missing
        demo_group = libero_demo_group_factory(
            obs_arrays={},
            actions_array=rng.standard_normal((3, 7)).astype(np.float32),
        )

        with pytest.raises(ValueError, match="not found in HDF5"):
            schema.extract_episode(
                demo_group=demo_group,
                resizer=noop_resizer,
                depth_resizer=noop_resizer,
            )

    def test_extracts_string_dtype_observation(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        string_obs = ObservationMetadata(
            raw_data_column_keys=["task_description"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = {
            "task_desc": string_obs,
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=7,
                slice_start=0,
                slice_end=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        mock_string_data = MagicMock()
        mock_string_data.astype.return_value.__getitem__ = MagicMock(
            return_value=np.array(["pick up bowl", "pick up bowl"])
        )

        # Build obs manually since string mock needs special handling
        obs_data = {
            "task_description": mock_string_data,
            Cameras.LEFT.value: MagicMock(
                __getitem__=MagicMock(
                    return_value=rng.integers(0, 255, size=(2, 64, 64, 3), dtype=np.uint8)
                )
            ),
        }
        obs_mock = MagicMock()
        obs_mock.__getitem__ = MagicMock(side_effect=lambda k: obs_data[k])
        obs_mock.__contains__ = MagicMock(side_effect=lambda k: k in obs_data)

        actions_arr = MagicMock()
        actions_arr.__getitem__ = MagicMock(
            return_value=rng.standard_normal((2, 7)).astype(np.float32)
        )
        actions_arr.shape = (2, 7)

        demo_dict = {"obs": obs_mock, "actions": actions_arr}
        demo_group = MagicMock()
        demo_group.__getitem__ = MagicMock(side_effect=lambda k: demo_dict[k])
        demo_group.__contains__ = MagicMock(side_effect=lambda k: k in demo_dict)
        demo_group.file.filename = "/data/task_demo.hdf5"

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        assert "task_desc" in data

    def test_language_from_filename_not_appended_when_disabled(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        libero_demo_group_factory: Callable,
        noop_resizer,
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(
                camera_key=Cameras.LEFT.value,
                image_height=64,
                image_width=64,
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=7,
                slice_start=0,
                slice_end=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )
        schema.extract_language_from_filename = False

        demo_group = libero_demo_group_factory(
            obs_arrays={
                Cameras.LEFT.value: rng.integers(0, 255, size=(3, 64, 64, 3), dtype=np.uint8),
            },
            actions_array=rng.standard_normal((3, 7)).astype(np.float32),
        )

        data = schema.extract_episode(
            demo_group=demo_group,
            resizer=noop_resizer,
            depth_resizer=noop_resizer,
        )

        assert ObsKey.LANGUAGE.value not in data


class TestLiberoSchemaGetEpisodeLength:

    def test_uses_actions_key_when_present(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        actions_arr = MagicMock()
        actions_arr.shape = (10, 7)

        demo_group = MagicMock()
        demo_group.__getitem__ = MagicMock(return_value=actions_arr)
        demo_group.__contains__ = MagicMock(return_value=True)

        result = schema._get_episode_length(demo_group)

        assert result == 10

    def test_uses_first_obs_key_when_no_actions(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )
        schema.actions_key = None

        obs_arr = MagicMock()
        obs_arr.shape = (8, 3)
        obs_group = MagicMock()
        obs_group.keys.return_value = ["ee_pos"]
        obs_group.__getitem__ = MagicMock(return_value=obs_arr)
        obs_group.__iter__ = MagicMock(return_value=iter(["ee_pos"]))

        demo_group = MagicMock()
        demo_group.__getitem__ = MagicMock(side_effect=lambda k: {
            "obs": obs_group,
        }[k])
        demo_group.__contains__ = MagicMock(return_value=False)

        result = schema._get_episode_length(demo_group)

        assert result == 8

    def test_falls_back_to_obs_when_actions_key_not_in_demo_group(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        obs_arr = MagicMock()
        obs_arr.shape = (12, 3)
        obs_group = MagicMock()
        obs_group.keys.return_value = ["ee_pos"]
        obs_group.__getitem__ = MagicMock(return_value=obs_arr)
        obs_group.__iter__ = MagicMock(return_value=iter(["ee_pos"]))

        demo_group = MagicMock()
        demo_group.__getitem__ = MagicMock(side_effect=lambda k: {
            "obs": obs_group,
        }[k])
        demo_group.__contains__ = MagicMock(return_value=False)

        result = schema._get_episode_length(demo_group)

        assert result == 12


class TestLiberoSchemaGetLanguageFromFilename:

    @pytest.mark.parametrize(
        "hdf5_path, expected",
        [
            ("/data/pick_up_the_black_bowl_demo.hdf5", "pick up the black bowl"),
            ("single_word_demo.hdf5", "single word"),
            ("/a/b/c/open_the_drawer_demo.hdf5", "open the drawer"),
        ],
        ids=["standard_path", "filename_only", "deep_nested_path"],
    )
    def test_extracts_task_name(self, hdf5_path: str, expected: str):
        result = LiberoSchema.get_language_from_filename(hdf5_path)

        assert result == expected


class TestLiberoSchemaGetRequiredZarrKeys:

    def test_includes_language_key_when_not_in_metadata(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        keys = schema.get_required_zarr_keys()

        assert ObsKey.LANGUAGE.value in keys

    def test_no_duplicate_when_already_in_metadata(
        self,
        valid_libero_metadata: DatasetMetadata,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        language_observation = ObservationMetadata(
            raw_data_column_keys=["language"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = dict(valid_libero_metadata.observations)
        observations[ObsKey.LANGUAGE.value] = language_observation
        metadata = dataset_metadata_factory(
            observations=observations,
            precomputed_actions=dict(valid_libero_metadata.precomputed_actions),
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        keys = schema.get_required_zarr_keys()

        language_count = keys.count(ObsKey.LANGUAGE.value)
        assert language_count == 1

    def test_no_language_key_when_extract_disabled(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )
        schema.extract_language_from_filename = False

        keys = schema.get_required_zarr_keys()

        assert ObsKey.LANGUAGE.value not in keys


class TestLiberoSchemaGetZarrArraySpecs:

    def test_includes_language_spec_when_not_in_base(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        specs = schema.get_zarr_array_specs()

        assert ObsKey.LANGUAGE.value in specs
        assert specs[ObsKey.LANGUAGE.value]["dtype"] == "str"
        assert specs[ObsKey.LANGUAGE.value]["needs_compressor"] is False
        assert specs[ObsKey.LANGUAGE.value]["shape"] == (0, 1)

    def test_no_duplicate_when_already_in_base(
        self,
        valid_libero_metadata: DatasetMetadata,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        language_observation = ObservationMetadata(
            raw_data_column_keys=["language"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = dict(valid_libero_metadata.observations)
        observations[ObsKey.LANGUAGE.value] = language_observation
        metadata = dataset_metadata_factory(
            observations=observations,
            precomputed_actions=dict(valid_libero_metadata.precomputed_actions),
        )
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=metadata,
            dataset_type=DatasetType.LIBERO.value,
        )

        specs = schema.get_zarr_array_specs()

        assert ObsKey.LANGUAGE.value in specs

    def test_no_language_spec_when_extract_disabled(
        self,
        valid_libero_metadata: DatasetMetadata,
    ):
        schema = LiberoSchema(
            hdf5_paths=["/data/task.hdf5"],
            zarr_path="/tmp/test.zarr",
            metadata=valid_libero_metadata,
            dataset_type=DatasetType.LIBERO.value,
        )
        schema.extract_language_from_filename = False

        specs = schema.get_zarr_array_specs()

        assert ObsKey.LANGUAGE.value not in specs