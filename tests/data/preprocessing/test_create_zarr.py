import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import cv2
import albumentations as A

from refactoring.data.raw.schemas.base import DatasetSchema
from refactoring.data.constants import (
    Cameras,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
)
from refactoring.data.preprocessing.create_zarr_from_csv import create_replay_buffer, _create_zarr_arrays, _append_observations, _append_images


class MockDatasetSchema(DatasetSchema):
    """Concrete mock implementation of DatasetSchema."""


    def get_image_path_column(self, camera: str) -> str:
        return f"{camera}_image_path"


    def compute_depth_path(self, base_image_path: str) -> str:
        return base_image_path.replace("left", "depth").replace(".png", ".npy")


@pytest.fixture
def mock_schema(tmp_path):
    """Fixture for a mock DatasetSchema."""
    observation_space = MagicMock()
    observation_space.image_width = 64
    observation_space.image_height = 64
    observation_space.robot_frame_proprio_keys = ["pos_x", "pos_y", "pos_z"]
    observation_space.camera_frame_proprio_keys = []
    observation_space.gripper_state_keys = []
    observation_space.custom_obs_keys = {}
    observation_space.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value]
    observation_space.language_key = None  # No language by default

    schema = MockDatasetSchema(
        dataset_folders=[],
        zarr_path=str(tmp_path / "test.zarr"),
        dataset_filename="data.csv",
        metadata=observation_space,
        image_path_config=MagicMock(),
        has_phase_labels=False,
    )

    return schema


@pytest.fixture
def mock_schema_with_language(tmp_path):
    """Fixture for a mock DatasetSchema with language support."""
    observation_space = MagicMock()
    observation_space.image_width = 64
    observation_space.image_height = 64
    observation_space.robot_frame_proprio_keys = ["pos_x", "pos_y", "pos_z"]
    observation_space.camera_frame_proprio_keys = []
    observation_space.gripper_state_keys = []
    observation_space.custom_obs_keys = {}
    observation_space.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value]
    observation_space.language_key = "instruction"  # Enable language

    schema = MockDatasetSchema(
        dataset_folders=[],
        zarr_path=str(tmp_path / "test.zarr"),
        dataset_filename="data.csv",
        metadata=observation_space,
        image_path_config=MagicMock(),
        has_phase_labels=False,
    )

    return schema


@pytest.fixture
def sample_episode_df():
    """Fixture for a sample episode DataFrame."""
    data = {
        "left_image_path": ["1/left_0001.png", "1/left_0002.png"],
        "right_image_path": ["1/right_0001.png", "1/right_0002.png"],
        "pos_x": [0.1, 0.2],
        "pos_y": [0.3, 0.4],
        "pos_z": [0.5, 0.6],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_episode_df_with_language():
    """Fixture for a sample episode DataFrame with language instructions."""
    data = {
        "left_image_path": ["1/left_0001.png", "1/left_0002.png"],
        "right_image_path": ["1/right_0001.png", "1/right_0002.png"],
        "pos_x": [0.1, 0.2],
        "pos_y": [0.3, 0.4],
        "pos_z": [0.5, 0.6],
        "instruction": ["pick up the red cube", "place the cube on the table"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_data_dir(tmp_path, sample_episode_df, mock_schema):
    """Fixture to create synthetic episode data with images and CSV."""
    ep_dir = tmp_path / "1"
    ep_dir.mkdir()

    # Create dummy RGB images
    dummy_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    for path in sample_episode_df["left_image_path"]:
        full_path = tmp_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full_path), dummy_rgb)

    for path in sample_episode_df["right_image_path"]:
        full_path = tmp_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full_path), dummy_rgb)

    # Create dummy depth images
    dummy_depth = np.random.rand(64, 64).astype(np.float32)
    for base in sample_episode_df["left_image_path"]:
        depth_path = mock_schema.compute_depth_path(base)
        full_depth_path = tmp_path / depth_path
        full_depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(full_depth_path), dummy_depth)

    # Save CSV
    csv_path = ep_dir / "data.csv"
    sample_episode_df.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def mocked_data_group():
    class DataGroupMock:

        def __init__(self):
            self.arrays = {}


        def __getitem__(self, key):
            if key not in self.arrays:
                self.arrays[key] = MagicMock()
            return self.arrays[key]


    return DataGroupMock()


class TestCreateZarrArrays:
    """Test Zarr array creation."""


    def test_create_zarr_arrays(self, mock_schema):
        data_group = MagicMock()
        compressor = MagicMock()

        _create_zarr_arrays(data_group=data_group, schema=mock_schema, compressor=compressor)

        specs = mock_schema.get_zarr_array_specs()
        assert data_group.create_array.call_count == len(specs)

        for key, spec in specs.items():
            expected_compressor = [compressor] if spec["needs_compressor"] else None
            expected_dtype = str if spec["dtype"] == 'str' else getattr(np, spec["dtype"])
            data_group.create_array.assert_any_call(
                key,
                shape=spec["shape"],
                chunks=spec["chunks"],
                dtype=expected_dtype,
                compressors=expected_compressor,
            )


    def test_create_zarr_arrays_with_language(self, mock_schema_with_language):
        """Test that language arrays are created with str dtype."""
        data_group = MagicMock()
        compressor = MagicMock()

        _create_zarr_arrays(data_group=data_group, schema=mock_schema_with_language, compressor=compressor)

        specs = mock_schema_with_language.get_zarr_array_specs()
        assert data_group.create_array.call_count == len(specs)

        # Verify language array was created with str dtype and no compressor
        language_key = mock_schema_with_language.metadata.language_key
        data_group.create_array.assert_any_call(
            language_key,
            shape=(0,),
            chunks=(100,),
            dtype=str,
            compressors=None,
        )


class TestAppendObservations:
    """Test observation appending."""


    def test_append_observations_basic(self, sample_episode_df, mock_schema):
        data_group = MagicMock()
        _append_observations(episode=sample_episode_df, data_group=data_group, schema=mock_schema)

        calls = data_group[PROPRIO_OBS_ROBOT_FRAME_KEY].append.call_args_list
        assert len(calls) == 1
        appended = calls[0][0][0]
        expected = sample_episode_df[
            mock_schema.metadata.robot_frame_proprio_keys
        ].values.astype(np.float32)
        np.testing.assert_allclose(appended, expected)


    def test_append_observations_with_language(self, sample_episode_df_with_language, mock_schema_with_language, mocked_data_group):
        """Test that language instructions are properly appended."""
        _append_observations(
            episode=sample_episode_df_with_language,
            data_group=mocked_data_group,
            schema=mock_schema_with_language
        )

        # Check robot frame observations
        calls = mocked_data_group[PROPRIO_OBS_ROBOT_FRAME_KEY].append.call_args_list
        assert len(calls) == 1

        # Check language instructions
        language_key = mock_schema_with_language.metadata.language_key
        calls = mocked_data_group[language_key].append.call_args_list
        assert len(calls) == 1

        appended_language = calls[0][0][0]
        expected_language = sample_episode_df_with_language[language_key].astype(str).values

        assert isinstance(appended_language, np.ndarray)
        assert len(appended_language) == 2
        assert appended_language[0] == "pick up the red cube"
        assert appended_language[1] == "place the cube on the table"
        np.testing.assert_array_equal(appended_language, expected_language)


class TestAppendImages:
    """Test image appending."""


    @patch("refactoring.data.preprocessing.create_zarr.cv2.imread")
    @patch("refactoring.data.preprocessing.create_zarr.cv2.cvtColor")
    @patch("refactoring.data.preprocessing.create_zarr.np.load")
    def test_append_images(self, mock_np_load, mock_cvtColor, mock_imread, sample_episode_df, mock_schema, mocked_data_group):
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

        dummy_rgb = np.ones((64, 64, 3), dtype=np.uint8) * 100
        dummy_depth = np.ones((64, 64), dtype=np.float32) * 0.5

        mock_imread.side_effect = lambda p, f: dummy_rgb
        mock_cvtColor.side_effect = lambda img, code: img
        mock_np_load.side_effect = lambda p: dummy_depth

        _append_images(episode=sample_episode_df, data_group=mocked_data_group, schema=mock_schema, resizer=resizer, depth_resizer=depth_resizer)

        # Assert for left
        calls = mocked_data_group[Cameras.LEFT.value].append.call_args_list
        assert len(calls) == 1
        appended = calls[0][0][0]
        expected = np.stack([dummy_rgb] * len(sample_episode_df))
        np.testing.assert_allclose(appended, expected)

        # Assert for right
        calls = mocked_data_group[Cameras.RIGHT.value].append.call_args_list
        assert len(calls) == 1
        appended = calls[0][0][0]
        expected = np.stack([dummy_rgb] * len(sample_episode_df))
        np.testing.assert_allclose(appended, expected)

        # Assert for depth
        calls = mocked_data_group[Cameras.DEPTH.value].append.call_args_list
        assert len(calls) == 1
        appended = calls[0][0][0]
        expected = np.stack([dummy_depth] * len(sample_episode_df))
        np.testing.assert_allclose(appended, expected)


class TestIntegration:
    """Integration tests for replay buffer creation."""


    @patch("refactoring.data.preprocessing.create_zarr._create_zarr_arrays")
    @patch("refactoring.data.preprocessing.create_zarr._append_observations")
    @patch("refactoring.data.preprocessing.create_zarr._append_images")
    def test_create_replay_buffer(self, mock_append_images, mock_append_observations, mock_create_arrays, mock_schema, synthetic_data_dir):
        datasets_paths = [synthetic_data_dir]

        with patch("zarr.open_group") as mock_open_group, \
                patch("threadpoolctl.threadpool_limits"):
            mock_root = MagicMock()
            mock_data = MagicMock()
            mock_meta = MagicMock()
            mock_root.create_group.side_effect = [mock_data, mock_meta]
            mock_open_group.return_value = mock_root

            create_replay_buffer(schema=mock_schema, datasets_paths=datasets_paths)

            mock_create_arrays.assert_called_once()
            mock_append_observations.assert_called_once()
            mock_append_images.assert_called_once()

            mock_meta.create_array.assert_called_once_with(
                'episode_ends',
                data=np.array([2]),
                chunks=(1,),
                compressors=None,
            )


class TestLanguageInstructions:
    """Dedicated tests for language instruction handling."""


    def test_language_spec_in_schema(self, mock_schema_with_language):
        """Test that language spec is correctly generated in schema."""
        specs = mock_schema_with_language.get_zarr_array_specs()
        language_key = mock_schema_with_language.metadata.language_key

        assert language_key in specs
        assert specs[language_key]['dtype'] == 'str'
        assert specs[language_key]['shape'] == (0,)
        assert specs[language_key]['chunks'] == (100,)
        assert specs[language_key]['needs_compressor'] is False


    def test_language_with_various_lengths(self, mock_schema_with_language, mocked_data_group):
        """Test that variable-length strings are handled correctly."""
        # Create episode with varying instruction lengths
        data = {
            "left_image_path": ["1/left_0001.png", "1/left_0002.png", "1/left_0003.png"],
            "right_image_path": ["1/right_0001.png", "1/right_0002.png", "1/right_0003.png"],
            "pos_x": [0.1, 0.2, 0.3],
            "pos_y": [0.3, 0.4, 0.5],
            "pos_z": [0.5, 0.6, 0.7],
            "instruction": [
                "pick",  # Short
                "pick up the red cube and place it on the table",  # Long
                "grasp the object",  # Medium
            ],
        }
        episode_df = pd.DataFrame(data)

        _append_observations(
            episode=episode_df,
            data_group=mocked_data_group,
            schema=mock_schema_with_language
        )

        language_key = mock_schema_with_language.metadata.language_key
        calls = mocked_data_group[language_key].append.call_args_list
        assert len(calls) == 1

        appended_language = calls[0][0][0]
        assert len(appended_language) == 3
        assert appended_language[0] == "pick"
        assert appended_language[1] == "pick up the red cube and place it on the table"
        assert appended_language[2] == "grasp the object"


    def test_language_with_special_characters(self, mock_schema_with_language, mocked_data_group):
        """Test that special characters in language instructions are preserved."""
        data = {
            "left_image_path": ["1/left_0001.png"],
            "right_image_path": ["1/right_0001.png"],
            "pos_x": [0.1],
            "pos_y": [0.3],
            "pos_z": [0.5],
            "instruction": ["pick up the object & place it (carefully!)"],
        }
        episode_df = pd.DataFrame(data)

        _append_observations(
            episode=episode_df,
            data_group=mocked_data_group,
            schema=mock_schema_with_language
        )

        language_key = mock_schema_with_language.metadata.language_key
        calls = mocked_data_group[language_key].append.call_args_list

        appended_language = calls[0][0][0]
        assert appended_language[0] == "pick up the object & place it (carefully!)"