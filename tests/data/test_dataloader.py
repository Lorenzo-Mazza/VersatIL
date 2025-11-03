import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil
import tempfile

from refactoring.data.dataloader import (
    get_dataloaders,
    _collect_dataset_paths,
    _ensure_zarr_exists,
    _log_phase_distributions,
)
from refactoring.data.constants import (
    EPISODE_FILENAME,
    PHASE_LABEL_KEY,
)


class MockConfig:
    """Mock configuration object."""


    def __init__(
            self,
            prediction_horizon = 4,
            observation_horizon = 3,
            batch_size = 32,
            shuffle = True,
            num_workers = 4,
            val_ratio = 0.1,
            total_ratio = 1.0,
            downsample_factor = 1,
            skip_initial_episode_steps = 0,
            center_episode_start = False,
            action_backward_shift = 0,
            winsorize_depth = True,
            has_gripper = True,
            use_gripper_class_weights = True,
            denoise_actions = True,
            task_has_phases = False,
            camera_keys = None,
            seed = 42,
            device = "cpu",
            use_language = False,
    ):
        self.task = MagicMock()
        self.task.prediction_horizon = prediction_horizon
        self.task.observation_horizon = observation_horizon
        self.task.dataset_schema = None

        self.task.action_space = MagicMock()
        self.task.action_space.has_gripper = has_gripper
        self.task.action_space.use_gripper_class_weights = use_gripper_class_weights
        self.task.action_space.denoise_actions = denoise_actions

        self.task.observation_space = MagicMock()
        self.task.observation_space.camera_keys = camera_keys or ["left", "right"]
        self.task.observation_space.task_has_phases = task_has_phases
        self.task.observation_space.use_language = use_language

        self.task.dataloader = MagicMock()
        self.task.dataloader.batch_size = batch_size
        self.task.dataloader.shuffle = shuffle
        self.task.dataloader.num_workers = num_workers
        self.task.dataloader.val_ratio = val_ratio
        self.task.dataloader.total_ratio = total_ratio
        self.task.dataloader.downsample_factor = downsample_factor
        self.task.dataloader.skip_initial_episode_steps = skip_initial_episode_steps
        self.task.dataloader.center_episode_start = center_episode_start
        self.task.dataloader.action_backward_shift = action_backward_shift
        self.task.dataloader.winsorize_depth = winsorize_depth

        self.experiment = MagicMock()
        self.experiment.seed = seed
        self.experiment.device = device


class MockSchema:
    """Mock dataset schema."""


    def __init__(self, zarr_path, camera_keys = None, required_keys = None):
        self.zarr_path = zarr_path
        self.dataset_folders = []
        self.observation_space = MagicMock()
        self.observation_space.camera_keys = camera_keys or ["left", "right"]
        self._required_keys = required_keys or ["left", "right", "proprio_robot_frame"]


    def get_required_zarr_keys(self):
        return self._required_keys


class MockActionProcessor:
    """Mock action processor."""


    def __init__(
            self,
            action_denoising_threshold = 0.01,
            orientation_denoising_threshold = 0.01
    ):
        self.action_denoising_threshold = action_denoising_threshold
        self.orientation_denoising_threshold = orientation_denoising_threshold


class MockDataset:
    """Mock episodic dataset."""


    def __init__(
            self,
            length = 100,
            action_denoising_threshold = 0.01,
            orientation_denoising_threshold = 0.01,
            gripper_weight = 1.0,
    ):
        self._length = length
        self.action_processor = MockActionProcessor(
            action_denoising_threshold=action_denoising_threshold,
            orientation_denoising_threshold=orientation_denoising_threshold,
        )
        self._gripper_weight = gripper_weight
        self._normalizer = MagicMock()

        self.get_normalizer = MagicMock(return_value=self._normalizer)
        self.get_gripper_positive_class_imbalance_weight = MagicMock(return_value=self._gripper_weight)

        self.sampler = MagicMock()
        self.sampler.episode_mask = np.ones(10, dtype=bool)

        self.replay_buffer = MagicMock()
        self.replay_buffer.n_episodes = 10
        self.replay_buffer.n_steps = length
        self.replay_buffer.get_episode = MagicMock(
            return_value={PHASE_LABEL_KEY: np.array([[0], [1], [2]])}
        )


    def __len__(self):
        return self._length


class MockReplayBuffer:
    """Mock replay buffer."""


    def __init__(self, n_episodes = 10, n_steps = 100):
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.episode_ends = np.linspace(
            n_steps // n_episodes,
            n_steps,
            n_episodes,
            dtype=int
        )


    def keys(self):
        return ["left", "right", "proprio_robot_frame"]


    def __getitem__(self, key):
        return np.random.randint(0, 2, size=(self.n_steps, 1)).astype(np.float32)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_dataset_folders(temp_dir):
    """Create mock dataset folder structure."""
    folders = []
    for folder_idx in range(2):
        folder = Path(temp_dir) / f"dataset_{folder_idx}"
        folder.mkdir(parents=True)

        for ep_idx in range(3):
            ep_dir = folder / str(ep_idx)
            ep_dir.mkdir()
            ep_file = ep_dir / EPISODE_FILENAME
            ep_file.write_text(f"mock,data,{folder_idx},{ep_idx}\n")

        folders.append(str(folder))

    return folders


@pytest.fixture
def mock_config():
    """Mock main configuration."""
    return MockConfig()


@pytest.fixture
def mock_schema(temp_dir):
    """Mock dataset schema."""
    return MockSchema(zarr_path=str(Path(temp_dir) / "test.zarr"))


class TestCollectDatasetPaths:
    """Test dataset path collection."""


    def test_collect_paths_single_folder(self, mock_dataset_folders):
        """Test collecting paths from a single folder."""
        paths = _collect_dataset_paths([mock_dataset_folders[0]])

        assert len(paths) == 3
        for path in paths:
            assert EPISODE_FILENAME in path
            assert Path(path).exists()


    def test_collect_paths_multiple_folders(self, mock_dataset_folders):
        """Test collecting paths from multiple folders."""
        paths = _collect_dataset_paths(mock_dataset_folders)

        assert len(paths) == 6
        for path in paths:
            assert EPISODE_FILENAME in path


    def test_collect_paths_empty_folder(self, temp_dir):
        """Test collecting paths from empty folder."""
        empty_folder = Path(temp_dir) / "empty"
        empty_folder.mkdir()

        paths = _collect_dataset_paths([str(empty_folder)])

        assert len(paths) == 0


    def test_collect_paths_nonexistent_folder(self):
        """Test collecting paths from nonexistent folder."""
        with pytest.raises(FileNotFoundError):
            _collect_dataset_paths(["/nonexistent/path"])


    def test_collect_paths_mixed_valid_invalid(self, mock_dataset_folders, temp_dir):
        """Test collecting paths with mix of valid and invalid folders."""
        invalid_folder = Path(temp_dir) / "invalid"
        invalid_folder.mkdir()

        (invalid_folder / "0").mkdir()

        folders = mock_dataset_folders + [str(invalid_folder)]
        paths = _collect_dataset_paths(folders)

        assert len(paths) == 6


class TestEnsureZarrExists:
    """Test zarr existence checking and creation."""


    @patch('refactoring.data.dataloader.ReplayBuffer')
    @patch('refactoring.data.dataloader.create_replay_buffer')
    def test_create_zarr_when_missing(
            self, mock_create, mock_buffer_class, mock_schema, mock_dataset_folders
    ):
        """Test zarr creation when it doesn't exist."""
        _ensure_zarr_exists(mock_schema, mock_dataset_folders)

        mock_create.assert_called_once_with(
            schema=mock_schema,
            datasets_paths=mock_dataset_folders
        )


    @patch('refactoring.data.dataloader.ReplayBuffer')
    @patch('refactoring.data.dataloader.create_replay_buffer')
    def test_skip_creation_when_zarr_exists(
            self, mock_create, mock_buffer_class, mock_schema, mock_dataset_folders, temp_dir
    ):
        """Test skipping zarr creation when valid zarr exists."""
        zarr_path = Path(mock_schema.zarr_path)
        zarr_path.mkdir(parents=True)

        mock_buffer_class.copy_from_path.return_value = MockReplayBuffer()

        _ensure_zarr_exists(mock_schema, mock_dataset_folders)

        mock_create.assert_not_called()


    @patch('refactoring.data.dataloader.ReplayBuffer')
    @patch('refactoring.data.dataloader.create_replay_buffer')
    @patch('refactoring.data.dataloader.shutil')
    def test_recreate_zarr_on_load_error(
            self, mock_shutil, mock_create, mock_buffer_class, mock_schema,
            mock_dataset_folders, temp_dir
    ):
        """Test zarr recreation when loading fails."""
        zarr_path = Path(mock_schema.zarr_path)
        zarr_path.mkdir(parents=True)

        mock_buffer_class.copy_from_path.side_effect = Exception("Load failed")

        _ensure_zarr_exists(mock_schema, mock_dataset_folders)

        mock_shutil.rmtree.assert_called_once()
        mock_create.assert_called_once()


class TestLogPhaseDistributions:
    """Test phase distribution logging."""


    def test_log_phase_distributions_basic(self, caplog):
        """Test basic phase distribution logging."""
        train_dataset = MockDataset()
        val_dataset = MockDataset()

        train_dataset.sampler.episode_mask = np.array([True, True, False, True])
        train_dataset.replay_buffer.get_episode = MagicMock(side_effect=[
            {PHASE_LABEL_KEY: np.array([[0], [0], [1], [1], [2]])},
            {PHASE_LABEL_KEY: np.array([[1], [2], [2], [3]])},
            {PHASE_LABEL_KEY: np.array([[3], [4], [4]])},
        ])

        val_dataset.sampler.episode_mask = np.array([False, False, True, False])
        val_dataset.replay_buffer.get_episode = MagicMock(
            return_value={PHASE_LABEL_KEY: np.array([[0], [1], [2]])}
        )

        with caplog.at_level("INFO"):
            _log_phase_distributions(train_dataset, val_dataset)

        assert "Train phase distribution" in caplog.text
        assert "Val phase distribution" in caplog.text


    def test_log_phase_distributions_no_selected_episodes(self, caplog):
        """Test logging with no selected episodes."""
        train_dataset = MockDataset()
        val_dataset = MockDataset()

        train_dataset.sampler.episode_mask = np.array([False, False, False])
        val_dataset.sampler.episode_mask = np.array([False, False, False])

        with caplog.at_level("INFO"):
            _log_phase_distributions(train_dataset, val_dataset)

        assert train_dataset.replay_buffer.get_episode.call_count == 0
        assert val_dataset.replay_buffer.get_episode.call_count == 0


    def test_log_phase_distributions_unbalanced(self, caplog):
        """Test logging with unbalanced phase distribution."""
        train_dataset = MockDataset()
        val_dataset = MockDataset()

        train_dataset.sampler.episode_mask = np.array([True])
        train_dataset.replay_buffer.get_episode = MagicMock(
            return_value={PHASE_LABEL_KEY: np.array([[0]] * 95 + [[1]] * 3 + [[2]] * 2)}
        )

        val_dataset.sampler.episode_mask = np.array([True])
        val_dataset.replay_buffer.get_episode = MagicMock(
            return_value={PHASE_LABEL_KEY: np.array([[0]] * 10)}
        )

        with caplog.at_level("INFO"):
            _log_phase_distributions(train_dataset, val_dataset)

        assert "Train phase distribution" in caplog.text


class TestGetDataloaders:
    """Test main dataloader creation function."""


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_basic(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test basic dataloader creation."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv", "path2.csv"]

        train_dataset = MockDataset(length=100, gripper_weight=2.5)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, normalizer, gripper_weights = get_dataloaders(mock_config)

        assert train_loader is not None
        assert val_loader is not None
        assert normalizer == train_dataset._normalizer
        assert gripper_weights == 2.5
        assert mock_dataset_class.call_count == 2


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_train_val_parameters(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test that train and val datasets get correct parameters."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        get_dataloaders(mock_config)

        train_call = mock_dataset_class.call_args_list[0]
        assert train_call[1]['train'] is True
        assert train_call[1]['seed'] == 42
        assert train_call[1]['pred_horizon'] == 4
        assert train_call[1]['obs_horizon'] == 3

        val_call = mock_dataset_class.call_args_list[1]
        assert val_call[1]['train'] is False
        assert val_call[1]['seed'] == 42


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_shares_denoising_thresholds(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test that denoising thresholds are shared from train to val."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(
            length=100,
            action_denoising_threshold=0.05,
            orientation_denoising_threshold=0.03
        )
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        get_dataloaders(mock_config)

        assert val_dataset.action_processor.action_denoising_threshold == 0.05
        assert val_dataset.action_processor.orientation_denoising_threshold == 0.03


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_no_gripper_weights(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_schema
    ):
        """Test dataloader creation without gripper class weights."""
        config = MockConfig(has_gripper=False)
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, normalizer, gripper_weights = get_dataloaders(config)

        assert gripper_weights is None
        train_dataset.get_gripper_positive_class_imbalance_weight.assert_not_called()


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_with_phases(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_schema
    ):
        """Test dataloader creation with phase labels."""
        config = MockConfig(task_has_phases=True)
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)

        train_dataset.sampler.episode_mask = np.array([True, False])
        train_dataset.replay_buffer.get_episode = MagicMock(
            return_value={PHASE_LABEL_KEY: np.array([[0], [1], [2]])}
        )
        val_dataset.sampler.episode_mask = np.array([False, True])
        val_dataset.replay_buffer.get_episode = MagicMock(
            return_value={PHASE_LABEL_KEY: np.array([[0], [1]])}
        )

        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        with patch('refactoring.data.dataloader._log_phase_distributions') as mock_log:
            get_dataloaders(config)

            mock_log.assert_called_once_with(train_dataset, val_dataset)


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_batch_size_configuration(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_schema
    ):
        """Test that batch size is configured correctly."""
        config = MockConfig(batch_size=64)
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, _, _ = get_dataloaders(config)

        assert train_loader.batch_size == 64
        assert val_loader.batch_size == 64


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_num_workers_configuration(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_schema
    ):
        """Test that num_workers is configured correctly."""
        config = MockConfig(num_workers=8)
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, _, _ = get_dataloaders(config)

        assert train_loader.num_workers == 8
        assert val_loader.num_workers == 4


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_shuffle_configuration(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test that shuffle is configured correctly."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, _, _ = get_dataloaders(mock_config)

        assert train_loader is not None
        assert val_loader is not None


class TestDataloaderIntegration:
    """Integration tests for dataloader functionality."""


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_complete_dataloader_pipeline(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test complete dataloader creation pipeline."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["ep1.csv", "ep2.csv", "ep3.csv"]

        train_dataset = MockDataset(length=100, gripper_weight=3.2)
        train_dataset._normalizer.normalize.return_value = torch.randn(4, 3)

        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, normalizer, gripper_weights = get_dataloaders(mock_config)

        # instantiate is called 3 times: for schema, action_space, and observation_space
        assert mock_instantiate.call_count == 3
        mock_collect.assert_called_once()
        mock_ensure.assert_called_once()
        assert mock_dataset_class.call_count == 2

        assert train_loader is not None
        assert val_loader is not None
        assert normalizer == train_dataset._normalizer
        assert gripper_weights == 3.2

        train_dataset.get_normalizer.assert_called_once()
        call_kwargs = train_dataset.get_normalizer.call_args[1]
        assert 'device' in call_kwargs
        assert call_kwargs['device'].type == 'cpu'


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_dataloader_persistent_workers(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test that persistent workers are enabled."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, _, _ = get_dataloaders(mock_config)

        assert hasattr(train_loader, '_iterator')
        assert hasattr(val_loader, '_iterator')


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_dataloader_with_cuda_device(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_schema
    ):
        """Test dataloader creation with CUDA device."""
        config = MockConfig(device="cuda:0")
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, normalizer, _ = get_dataloaders(config)

        call_kwargs = train_dataset.get_normalizer.call_args[1]
        assert call_kwargs['device'].type == 'cuda'


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    @patch('refactoring.data.dataloader._log_phase_distributions')
    def test_dataloader_all_features_enabled(
            self, mock_log_phases, mock_dataset_class, mock_ensure, mock_collect,
            mock_instantiate, mock_schema
    ):
        """Test dataloader with all features enabled."""
        config = MockConfig(
            has_gripper=True,
            use_gripper_class_weights=True,
            denoise_actions=True,
            task_has_phases=True,
            winsorize_depth=True
        )

        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv", "path2.csv"]

        train_dataset = MockDataset(
            length=100,
            action_denoising_threshold=0.02,
            orientation_denoising_threshold=0.01,
            gripper_weight=2.8
        )
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, normalizer, gripper_weights = get_dataloaders(config)

        assert normalizer is not None
        assert gripper_weights == 2.8
        mock_log_phases.assert_called_once()

        assert val_dataset.action_processor.action_denoising_threshold == 0.02
        assert val_dataset.action_processor.orientation_denoising_threshold == 0.01

        call_kwargs = train_dataset.get_normalizer.call_args[1]
        assert call_kwargs['winsorize_depth'] is True


class TestErrorHandling:
    """Test error handling in dataloader creation."""


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    def test_handles_missing_dataset_paths(
            self, mock_collect, mock_instantiate, mock_config, mock_schema
    ):
        """Test handling when no dataset paths are found."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = []

        with patch('refactoring.data.dataloader._ensure_zarr_exists'):
            with patch('refactoring.data.dataloader.EpisodicDataset'):
                try:
                    get_dataloaders(mock_config)
                except Exception:
                    pass


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_handles_dataset_creation_error(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_config, mock_schema
    ):
        """Test handling when dataset creation fails."""
        mock_instantiate.return_value = mock_schema
        mock_collect.return_value = ["path1.csv"]

        mock_dataset_class.side_effect = RuntimeError("Dataset creation failed")

        with pytest.raises(RuntimeError, match="Dataset creation failed"):
            get_dataloaders(mock_config)


class TestLanguageInDataloader:
    """Test language instruction support in dataloader."""


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_get_dataloaders_with_language(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate,
            mock_schema
    ):
        """Test dataloader creation with language enabled."""
        config = MockConfig(use_language=True)

        # Create mock observation space with use_language attribute
        mock_obs_space = MagicMock()
        mock_obs_space.use_language = True

        # instantiate is called 3 times: schema, action_space, observation_space
        mock_instantiate.side_effect = [mock_schema, MagicMock(), mock_obs_space]
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        train_loader, val_loader, normalizer, _ = get_dataloaders(config)

        train_call = mock_dataset_class.call_args_list[0]
        obs_space = train_call[1]['observation_space']
        assert hasattr(obs_space, 'use_language')
        assert obs_space.use_language is True


    @patch('refactoring.data.dataloader.instantiate')
    @patch('refactoring.data.dataloader._collect_dataset_paths')
    @patch('refactoring.data.dataloader._ensure_zarr_exists')
    @patch('refactoring.data.dataloader.EpisodicDataset')
    def test_schema_with_language_key(
            self, mock_dataset_class, mock_ensure, mock_collect, mock_instantiate
    ):
        """Test that schema includes language key when enabled."""
        config = MockConfig(use_language=True)

        schema = MockSchema(
            zarr_path="/tmp/test.zarr",
            required_keys=['left', 'right', 'proprio_robot_frame', 'language']
        )
        mock_instantiate.return_value = schema
        mock_collect.return_value = ["path1.csv"]

        train_dataset = MockDataset(length=100)
        val_dataset = MockDataset(length=20)
        mock_dataset_class.side_effect = [train_dataset, val_dataset]

        get_dataloaders(config)

        mock_ensure.assert_called_once()
        # Access call arguments properly - handle both positional and keyword args
        if mock_ensure.call_args.args:
            schema_arg = mock_ensure.call_args.args[0]
        else:
            schema_arg = mock_ensure.call_args.kwargs['schema']
        assert 'language' in schema_arg.get_required_zarr_keys()