"""Tests for versatil.data.dataloader module."""

import shutil
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.constants import ActionComputationMethod, ProprioKey
from versatil.data.dataloader import (
    _collect_dataset_paths,
    _ensure_zarr_exists,
    _log_phase_distributions,
    get_dataloaders,
    validate_dataloader_config,
)
from versatil.data.raw.schemas import CsvDatasetSchema
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.schemas.hdf5 import Hdf5DatasetSchema
from versatil.data.raw.schemas.lerobot import LeRobotDatasetSchemaV30
from versatil.data.task import ActionSpace, ObservationSpace


@pytest.fixture
def dataloader_config_factory() -> Callable[..., DataLoaderConfig]:
    def factory(
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        total_ratio: float = 1.0,
        skip_initial_episode_steps: int = 0,
        downsample_factor: int = 1,
        action_backward_shift: int = 0,
    ) -> DataLoaderConfig:
        return DataLoaderConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            val_ratio=val_ratio,
            total_ratio=total_ratio,
            skip_initial_episode_steps=skip_initial_episode_steps,
            downsample_factor=downsample_factor,
            action_backward_shift=action_backward_shift,
        )

    return factory


@pytest.fixture
def episode_directory_factory(
    tmp_path: Path,
) -> Callable[..., tuple[Path, list[Path]]]:
    def factory(
        folder_name: str = "dataset",
        num_episodes: int = 2,
        episode_filename: str = "episode.csv",
    ) -> tuple[Path, list[Path]]:
        folder_path = tmp_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        episode_paths = []
        for index in range(num_episodes):
            episode_directory = folder_path / f"episode_{index}"
            episode_directory.mkdir(exist_ok=True)
            episode_file = episode_directory / episode_filename
            episode_file.write_text("column_a,column_b\n1,2\n")
            episode_paths.append(episode_file)

        return folder_path, episode_paths

    return factory


@pytest.fixture
def mock_hydra_config_factory() -> Callable[..., MagicMock]:
    def factory(
        val_ratio: float = 0.2,
        batch_size: int = 4,
        num_workers: int = 2,
        shuffle: bool = True,
        denoise_actions: bool = False,
        has_gripper_actions: bool = False,
        use_gripper_class_weights: bool = False,
    ) -> MagicMock:
        dataloader_config = DataLoaderConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            val_ratio=val_ratio,
        )
        schema = MagicMock(spec=DatasetSchema)
        schema.zarr_path = "/tmp/test.zarr"

        action_space = MagicMock()
        action_space.denoise_actions = denoise_actions
        action_space.has_gripper_actions = has_gripper_actions
        action_space.use_gripper_class_weights = use_gripper_class_weights

        observation_space = MagicMock()

        config = MagicMock()
        config.task.dataset_schema = schema
        config.task.action_space = action_space
        config.task.observation_space = observation_space
        config.task.dataloader = dataloader_config
        config.task.prediction_horizon = 4
        config.task.observation_horizon = 2
        config.experiment.seed = 42
        return config

    return factory


@pytest.fixture
def mock_dataset_schema_factory() -> Callable[..., MagicMock]:
    def factory(
        zarr_path: str = "/tmp/test.zarr",
        schema_type: type = DatasetSchema,
        required_keys: list[str] | None = None,
        dataset_folders: list[str] | None = None,
        dataset_filename: str = "episode.csv",
    ) -> MagicMock:
        schema = MagicMock(spec=schema_type)
        schema.zarr_path = zarr_path
        schema.get_required_zarr_keys.return_value = (
            required_keys if required_keys is not None else ["left", "right"]
        )
        schema.dataset_folders = dataset_folders or []
        schema.dataset_filename = dataset_filename
        schema.hdf5_paths = []
        return schema

    return factory


@pytest.mark.unit
class TestValidateDataloaderConfig:
    @pytest.mark.parametrize(
        "batch_size, expectation",
        [
            (1, does_not_raise()),
            (64, does_not_raise()),
            (0, pytest.raises(ValueError, match="batch_size must be positive")),
            (-1, pytest.raises(ValueError, match="batch_size must be positive")),
        ],
    )
    def test_batch_size_validation(
        self, dataloader_config_factory, batch_size, expectation
    ):
        config = dataloader_config_factory(batch_size=batch_size)

        with expectation:
            validate_dataloader_config(config)

    @pytest.mark.parametrize(
        "num_workers, expectation",
        [
            (0, does_not_raise()),
            (4, does_not_raise()),
            (-1, pytest.raises(ValueError, match="num_workers cannot be negative")),
        ],
    )
    def test_num_workers_validation(
        self, dataloader_config_factory, num_workers, expectation
    ):
        config = dataloader_config_factory(num_workers=num_workers)

        with expectation:
            validate_dataloader_config(config)

    @pytest.mark.parametrize(
        "val_ratio, expectation",
        [
            (0.0, does_not_raise()),
            (0.1, does_not_raise()),
            (0.5, does_not_raise()),
            (0.99, does_not_raise()),
            (-0.1, pytest.raises(ValueError, match="val_ratio must be in range")),
            (1.0, pytest.raises(ValueError, match="val_ratio must be in range")),
            (1.5, pytest.raises(ValueError, match="val_ratio must be in range")),
        ],
    )
    def test_val_ratio_validation(
        self, dataloader_config_factory, val_ratio, expectation
    ):
        config = dataloader_config_factory(val_ratio=val_ratio)

        with expectation:
            validate_dataloader_config(config)

    @pytest.mark.parametrize(
        "total_ratio, expectation",
        [
            (0.01, does_not_raise()),
            (0.5, does_not_raise()),
            (1.0, does_not_raise()),
            (0.0, pytest.raises(ValueError, match="total_ratio must be in range")),
            (-0.1, pytest.raises(ValueError, match="total_ratio must be in range")),
            (1.1, pytest.raises(ValueError, match="total_ratio must be in range")),
        ],
    )
    def test_total_ratio_validation(
        self, dataloader_config_factory, total_ratio, expectation
    ):
        config = dataloader_config_factory(total_ratio=total_ratio)

        with expectation:
            validate_dataloader_config(config)

    @pytest.mark.parametrize(
        "skip_steps, expectation",
        [
            (0, does_not_raise()),
            (10, does_not_raise()),
            (
                -1,
                pytest.raises(
                    ValueError, match="skip_initial_episode_steps cannot be negative"
                ),
            ),
        ],
    )
    def test_skip_initial_episode_steps_validation(
        self, dataloader_config_factory, skip_steps, expectation
    ):
        config = dataloader_config_factory(
            skip_initial_episode_steps=skip_steps,
        )

        with expectation:
            validate_dataloader_config(config)

    @pytest.mark.parametrize(
        "downsample_factor, expectation",
        [
            (1, does_not_raise()),
            (5, does_not_raise()),
            (0, pytest.raises(ValueError, match="downsample_factor must be")),
            (-1, pytest.raises(ValueError, match="downsample_factor must be")),
        ],
    )
    def test_downsample_factor_validation(
        self, dataloader_config_factory, downsample_factor, expectation
    ):
        config = dataloader_config_factory(
            downsample_factor=downsample_factor,
        )

        with expectation:
            validate_dataloader_config(config)

    @pytest.mark.parametrize(
        "shift, expectation",
        [
            (0, does_not_raise()),
            (1, does_not_raise()),
            (5, does_not_raise()),
            (
                -1,
                pytest.raises(
                    ValueError, match="action_backward_shift cannot be negative"
                ),
            ),
        ],
    )
    def test_action_backward_shift_validation(
        self, dataloader_config_factory, shift, expectation
    ):
        config = dataloader_config_factory(action_backward_shift=shift)

        with expectation:
            validate_dataloader_config(config)


@pytest.mark.unit
class TestCollectDatasetPaths:
    def test_collects_from_single_folder(self, episode_directory_factory):
        folder_path, expected_paths = episode_directory_factory(
            num_episodes=2,
        )

        result = _collect_dataset_paths(
            dataset_folders=[str(folder_path)],
            episode_filename="episode.csv",
        )

        assert len(result) == 2
        for path in expected_paths:
            assert str(path) in result

    def test_collects_from_multiple_folders(self, episode_directory_factory):
        folder_one, _ = episode_directory_factory(
            folder_name="folder_one",
            num_episodes=1,
        )
        folder_two, _ = episode_directory_factory(
            folder_name="folder_two",
            num_episodes=2,
        )

        result = _collect_dataset_paths(
            dataset_folders=[str(folder_one), str(folder_two)],
            episode_filename="episode.csv",
        )

        assert len(result) == 3

    def test_empty_folder_returns_empty_list(self, tmp_path):
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        result = _collect_dataset_paths(
            dataset_folders=[str(empty_folder)],
            episode_filename="episode.csv",
        )

        assert result == []

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            _collect_dataset_paths(
                dataset_folders=["/nonexistent/path"],
                episode_filename="episode.csv",
            )

    def test_skips_subdirectories_without_episode_file(self, tmp_path):
        folder = tmp_path / "mixed"
        folder.mkdir()

        valid_directory = folder / "episode_0"
        valid_directory.mkdir()
        (valid_directory / "episode.csv").write_text("a,b\n1,2\n")

        # Directory without the expected episode file
        missing_file_directory = folder / "episode_1"
        missing_file_directory.mkdir()

        result = _collect_dataset_paths(
            dataset_folders=[str(folder)],
            episode_filename="episode.csv",
        )

        assert len(result) == 1
        assert str(valid_directory / "episode.csv") in result

    def test_uses_custom_episode_filename(self, episode_directory_factory):
        folder_path, _ = episode_directory_factory(
            num_episodes=2,
            episode_filename="data.hdf5",
        )

        result = _collect_dataset_paths(
            dataset_folders=[str(folder_path)],
            episode_filename="data.hdf5",
        )

        assert len(result) == 2
        assert all(path.endswith("data.hdf5") for path in result)

    def test_ignores_files_at_root_level(self, tmp_path):
        folder = tmp_path / "with_file"
        folder.mkdir()

        (folder / "stray_file.txt").write_text("not a directory")

        episode_directory = folder / "episode_0"
        episode_directory.mkdir()
        (episode_directory / "episode.csv").write_text("a,b\n1,2\n")

        result = _collect_dataset_paths(
            dataset_folders=[str(folder)],
            episode_filename="episode.csv",
        )

        assert len(result) == 1


@pytest.mark.unit
class TestEnsureZarrExists:
    def test_existing_zarr_loads_without_preload(self, mock_dataset_schema_factory):
        required_keys = ["left", "right"]
        schema = mock_dataset_schema_factory(required_keys=required_keys)
        mock_buffer = MagicMock()
        mock_buffer.keys.return_value = required_keys

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch("versatil.data.dataloader.ReplayBuffer") as mock_replay_buffer,
        ):
            mock_path_class.return_value.exists.return_value = True
            mock_replay_buffer.create_from_path.return_value = mock_buffer

            _ensure_zarr_exists(schema=schema, preload_in_memory=False)

            mock_replay_buffer.create_from_path.assert_called_once()

    def test_existing_zarr_preloads_into_memory(self, mock_dataset_schema_factory):
        required_keys = ["left", "right"]
        schema = mock_dataset_schema_factory(required_keys=required_keys)

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch("versatil.data.dataloader.ReplayBuffer") as mock_replay_buffer,
        ):
            mock_path_class.return_value.exists.return_value = True

            _ensure_zarr_exists(schema=schema, preload_in_memory=True)

            mock_replay_buffer.copy_from_path.assert_called_once()

    def test_missing_keys_triggers_recreation(self, mock_dataset_schema_factory):
        schema = mock_dataset_schema_factory(
            schema_type=Hdf5DatasetSchema,
            required_keys=["left", "right", "depth"],
        )
        mock_buffer = MagicMock()
        mock_buffer.keys.return_value = ["left", "right"]

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch("versatil.data.dataloader.ReplayBuffer") as mock_replay_buffer,
            patch("versatil.data.dataloader.shutil") as mock_shutil,
            patch(
                "versatil.data.dataloader.create_replay_buffer_from_hdf5"
            ) as mock_create,
        ):
            mock_path_class.return_value.exists.return_value = True
            mock_replay_buffer.create_from_path.return_value = mock_buffer

            _ensure_zarr_exists(schema=schema, preload_in_memory=False)

            mock_shutil.rmtree.assert_called_once()
            mock_create.assert_called_once_with(schema=schema)

    def test_preload_failure_triggers_recreation(self, mock_dataset_schema_factory):
        schema = mock_dataset_schema_factory(schema_type=Hdf5DatasetSchema)

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch("versatil.data.dataloader.ReplayBuffer") as mock_replay_buffer,
            patch("versatil.data.dataloader.shutil") as mock_shutil,
            patch(
                "versatil.data.dataloader.create_replay_buffer_from_hdf5"
            ) as mock_create,
        ):
            mock_path_class.return_value.exists.return_value = True
            mock_replay_buffer.copy_from_path.side_effect = Exception("Memory error")

            _ensure_zarr_exists(schema=schema, preload_in_memory=True)

            mock_shutil.rmtree.assert_called_once()
            mock_create.assert_called_once_with(schema=schema)

    def test_load_failure_triggers_recreation(self, mock_dataset_schema_factory):
        schema = mock_dataset_schema_factory(schema_type=Hdf5DatasetSchema)

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch("versatil.data.dataloader.ReplayBuffer") as mock_replay_buffer,
            patch("versatil.data.dataloader.shutil") as mock_shutil,
            patch(
                "versatil.data.dataloader.create_replay_buffer_from_hdf5"
            ) as mock_create,
        ):
            mock_path_class.return_value.exists.return_value = True
            mock_replay_buffer.create_from_path.side_effect = Exception("Corrupt zarr")

            _ensure_zarr_exists(schema=schema, preload_in_memory=False)

            mock_shutil.rmtree.assert_called_once()
            mock_create.assert_called_once_with(schema=schema)

    def test_nonexistent_zarr_creates_from_hdf5(self, mock_dataset_schema_factory):
        schema = mock_dataset_schema_factory(schema_type=Hdf5DatasetSchema)

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch(
                "versatil.data.dataloader.create_replay_buffer_from_hdf5"
            ) as mock_create,
        ):
            mock_path_class.return_value.exists.return_value = False

            _ensure_zarr_exists(schema=schema, preload_in_memory=False)

            mock_create.assert_called_once_with(schema=schema)

    def test_nonexistent_zarr_creates_from_csv(self, mock_dataset_schema_factory):
        schema = mock_dataset_schema_factory(
            schema_type=CsvDatasetSchema,
            dataset_folders=["/path/to/datasets"],
            dataset_filename="episode.csv",
        )

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch("versatil.data.dataloader.create_replay_buffer") as mock_create,
            patch(
                "versatil.data.dataloader._collect_dataset_paths",
                return_value=["/ep1.csv", "/ep2.csv"],
            ) as mock_collect,
        ):
            mock_path_class.return_value.exists.return_value = False

            _ensure_zarr_exists(schema=schema, preload_in_memory=False)

            mock_collect.assert_called_once_with(
                dataset_folders=schema.dataset_folders,
                episode_filename=schema.dataset_filename,
            )
            mock_create.assert_called_once_with(
                schema=schema,
                datasets_paths=["/ep1.csv", "/ep2.csv"],
            )

    def test_nonexistent_zarr_creates_from_lerobot(self, mock_dataset_schema_factory):
        schema = mock_dataset_schema_factory(
            schema_type=LeRobotDatasetSchemaV30,
        )

        with (
            patch("versatil.data.dataloader.Path") as mock_path_class,
            patch(
                "versatil.data.dataloader.create_replay_buffer_from_lerobot"
            ) as mock_create,
        ):
            mock_path_class.return_value.exists.return_value = False

            _ensure_zarr_exists(schema=schema, preload_in_memory=False)

            mock_create.assert_called_once_with(schema=schema)

    def test_unknown_schema_type_raises_not_implemented(
        self, mock_dataset_schema_factory
    ):
        schema = mock_dataset_schema_factory(schema_type=DatasetSchema)

        with patch("versatil.data.dataloader.Path") as mock_path_class:
            mock_path_class.return_value.exists.return_value = False

            with pytest.raises(
                NotImplementedError,
                match="Zarr creation not implemented",
            ):
                _ensure_zarr_exists(
                    schema=schema,
                    preload_in_memory=False,
                )


@pytest.mark.unit
class TestLogPhaseDistributions:
    def test_logs_warning_and_returns(self):
        train_dataset = MagicMock()
        val_dataset = MagicMock()

        with patch("versatil.data.dataloader.logging") as mock_logging:
            _log_phase_distributions(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            mock_logging.warning.assert_called_once()


@pytest.mark.unit
class TestGetDataloaders:
    @pytest.fixture(autouse=True)
    def _patch_dependencies(self):
        mock_normalizer = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_train_dataset.get_normalizer_and_tokenizer.return_value = (
            mock_normalizer,
            mock_tokenizer,
        )
        mock_train_dataset.action_processor.denoising_thresholds = {"key": 0.1}
        mock_train_dataset.get_gripper_positive_class_imbalance_weight.return_value = (
            0.75
        )

        mock_val_dataset = MagicMock()
        mock_val_dataset.action_processor = MagicMock()

        self.mock_normalizer = mock_normalizer
        self.mock_tokenizer = mock_tokenizer
        self.mock_train_dataset = mock_train_dataset
        self.mock_val_dataset = mock_val_dataset

        with (
            patch("versatil.data.dataloader.validate_dataloader_config"),
            patch("versatil.data.dataloader.validate_tokenizer_config"),
            patch("versatil.data.dataloader._ensure_zarr_exists"),
            patch(
                "versatil.data.dataloader.EpisodicDataset",
                side_effect=[mock_train_dataset, mock_val_dataset],
            ) as self.mock_episodic_dataset,
            patch(
                "versatil.data.dataloader.data.DataLoader",
                side_effect=lambda dataset, **kwargs: MagicMock(
                    dataset=dataset,
                    **kwargs,
                ),
            ) as self.mock_dataloader_class,
        ):
            yield

    def test_returns_train_loader_and_normalizer(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(val_ratio=0.2)

        train_loader, val_loader, normalizer, tokenizer, weights = get_dataloaders(
            config=config,
        )

        assert train_loader.dataset is self.mock_train_dataset
        assert normalizer.normalize == self.mock_normalizer.normalize
        assert tokenizer.encode == self.mock_tokenizer.encode

    def test_creates_val_loader_when_val_ratio_positive(
        self, mock_hydra_config_factory
    ):
        config = mock_hydra_config_factory(val_ratio=0.2)

        _, val_loader, _, _, _ = get_dataloaders(config=config)

        assert val_loader.dataset is self.mock_val_dataset

    def test_skips_validation_when_val_ratio_zero(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(val_ratio=0.0)

        _, val_loader, _, _, _ = get_dataloaders(config=config)

        assert val_loader is None

    def test_normalizer_and_tokenizer_set_on_train_dataset(
        self, mock_hydra_config_factory
    ):
        config = mock_hydra_config_factory(val_ratio=0.0)

        get_dataloaders(config=config)

        self.mock_train_dataset.set_normalizer.assert_called_once_with(
            self.mock_normalizer,
        )
        self.mock_train_dataset.set_tokenizer.assert_called_once_with(
            self.mock_tokenizer,
        )

    def test_normalizer_and_tokenizer_shared_with_val_dataset(
        self, mock_hydra_config_factory
    ):
        config = mock_hydra_config_factory(val_ratio=0.2)

        get_dataloaders(config=config)

        self.mock_val_dataset.set_normalizer.assert_called_once_with(
            self.mock_normalizer,
        )
        self.mock_val_dataset.set_tokenizer.assert_called_once_with(
            self.mock_tokenizer,
        )

    def test_denoising_thresholds_propagated_to_val_dataset(
        self, mock_hydra_config_factory
    ):
        config = mock_hydra_config_factory(
            val_ratio=0.2,
            denoise_actions=True,
        )

        get_dataloaders(config=config)

        assert self.mock_val_dataset.action_processor.denoising_thresholds == {
            "key": 0.1
        }
        assert (
            self.mock_val_dataset.action_processor._denoising_thresholds_computed
            is True
        )

    def test_denoising_thresholds_not_propagated_when_disabled(
        self, mock_hydra_config_factory
    ):
        config = mock_hydra_config_factory(
            val_ratio=0.2,
            denoise_actions=False,
        )

        get_dataloaders(config=config)

        assert not hasattr(
            self.mock_val_dataset.action_processor, "denoising_thresholds"
        ) or self.mock_val_dataset.action_processor.denoising_thresholds != {"key": 0.1}

    def test_gripper_weights_computed_when_enabled(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(
            has_gripper_actions=True,
            use_gripper_class_weights=True,
        )

        _, _, _, _, weights = get_dataloaders(config=config)

        assert weights == 0.75
        self.mock_train_dataset.get_gripper_positive_class_imbalance_weight.assert_called_once()

    def test_gripper_weights_none_when_disabled(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(
            has_gripper_actions=False,
            use_gripper_class_weights=False,
        )

        _, _, _, _, weights = get_dataloaders(config=config)

        assert weights is None

    def test_val_loader_uses_correct_config(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(
            val_ratio=0.2,
            batch_size=8,
            num_workers=6,
        )

        get_dataloaders(config=config)

        val_call = self.mock_dataloader_class.call_args_list[1]
        assert val_call.kwargs["shuffle"] is False
        assert val_call.kwargs["num_workers"] == min(4, 6)

    def test_train_dataset_created_in_train_mode(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(val_ratio=0.0)

        get_dataloaders(config=config)

        train_call = self.mock_episodic_dataset.call_args_list[0]
        assert train_call.kwargs["train"] is True

    def test_val_dataset_created_in_eval_mode(self, mock_hydra_config_factory):
        config = mock_hydra_config_factory(val_ratio=0.2)

        get_dataloaders(config=config)

        val_call = self.mock_episodic_dataset.call_args_list[1]
        assert val_call.kwargs["train"] is False


@pytest.mark.integration
class TestGetDataloadersIntegration:
    def test_creates_working_dataloaders_from_synthetic_data(
        self,
        synthetic_replay_buffer,
        position_observation_metadata_factory,
        gripper_observation_metadata_factory,
        on_the_fly_action_metadata_factory,
    ):
        zarr_path, _ = synthetic_replay_buffer(
            num_episodes=5,
            num_timesteps_per_episode=20,
            position_dim=3,
            orientation_dim=4,
            has_gripper=True,
            cameras=[],
        )

        position_metadata = position_observation_metadata_factory(dimension=3)
        gripper_metadata = gripper_observation_metadata_factory()

        observations_metadata = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_metadata,
            ProprioKey.GRIPPER_STATE.value: gripper_metadata,
        }
        actions_metadata = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: on_the_fly_action_metadata_factory(
                source_metadata=position_metadata,
            ),
            ProprioKey.GRIPPER_STATE.value: on_the_fly_action_metadata_factory(
                source_metadata=gripper_metadata,
                computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
            ),
        }

        action_space = ActionSpace(
            actions_metadata=actions_metadata,
            denoise_actions=False,
        )
        observation_space = ObservationSpace(
            observations_metadata=observations_metadata,
        )

        dataloader_config = DataLoaderConfig(
            batch_size=4,
            num_workers=1,
            val_ratio=0.2,
        )
        dataloader_config.color_augmentation = None
        dataloader_config.spatial_augmentation = None

        schema = MagicMock(spec=DatasetSchema)
        schema.zarr_path = zarr_path

        config = MagicMock()
        config.task.dataset_schema = schema
        config.task.action_space = action_space
        config.task.observation_space = observation_space
        config.task.dataloader = dataloader_config
        config.task.prediction_horizon = 4
        config.task.observation_horizon = 2
        config.experiment.seed = 42

        result = get_dataloaders(config=config)

        train_loader, val_loader, normalizer, tokenizer, gripper_weights = result

        assert train_loader is not None
        assert val_loader is not None
        assert normalizer is not None
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0

        shutil.rmtree(Path(zarr_path).parent, ignore_errors=True)
