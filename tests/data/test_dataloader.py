import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from refactoring.data.dataloader import (
    validate_dataloader_config,
    _collect_dataset_paths,
    _ensure_zarr_exists,
    _log_phase_distributions,
)
from refactoring.configs.data.dataloader import DataLoaderConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def dataloader_config_factory():
    """Factory for creating DataLoaderConfig instances."""
    def factory(**kwargs):
        defaults = {
            'batch_size': 32,
            'num_workers': 4,
            'shuffle': True,
            'image_height': 224,
            'image_width': 224,
            'val_ratio': 0.1,
            'total_ratio': 1.0,
            'skip_initial_episode_steps': 0,
            'downsample_factor': 1,
            'action_backward_shift': 0,
            'center_episode_start': False,
            'winsorize_depth': True,
            'depth_winsorize_quantiles': (0.01, 0.99),
            'winsorize_kinematics': False,
            'kinematics_winsorize_quantiles': (0.01, 0.99),
        }
        defaults.update(kwargs)
        return DataLoaderConfig(**defaults)
    return factory


@pytest.fixture
def mock_schema_factory():
    """Factory for creating mock dataset schemas."""
    def factory(zarr_path, **kwargs):
        schema = MagicMock()
        schema.zarr_path = zarr_path
        schema.dataset_folders = kwargs.get('dataset_folders', [])
        schema.dataset_filename = kwargs.get('episode_filename', 'episode.csv')
        schema.get_required_zarr_keys.return_value = kwargs.get(
            'required_keys', ['left', 'right', 'proprio_robot_frame']
        )
        return schema
    return factory


@pytest.fixture
def episode_dir_factory(temp_dir):
    """Factory for creating episode directories with CSV files."""
    def factory(folder_name, num_episodes=2, episode_filename='episode.csv'):
        folder_path = Path(temp_dir) / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        episode_paths = []
        for i in range(num_episodes):
            episode_dir = folder_path / f'episode_{i}'
            episode_dir.mkdir(exist_ok=True)
            episode_file = episode_dir / episode_filename
            episode_file.write_text('dummy,data\n1,2\n')
            episode_paths.append(str(episode_file))

        return str(folder_path), episode_paths
    return factory


@pytest.mark.unit
class TestValidateDataloaderConfig:
    """Test validate_dataloader_config function."""

    def test_valid_config_passes(self, dataloader_config_factory):
        """Test that valid configuration passes validation."""
        config = dataloader_config_factory()
        validate_dataloader_config(config)

    @pytest.mark.parametrize("batch_size,should_raise", [
        (1, False),
        (32, False),
        (0, True),
        (-1, True),
    ])
    def test_batch_size_validation(self, dataloader_config_factory, batch_size, should_raise):
        """Test batch_size validation."""
        config = dataloader_config_factory(batch_size=batch_size)
        if should_raise:
            with pytest.raises(ValueError, match="batch_size must be positive"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("num_workers,should_raise", [
        (0, False),
        (4, False),
        (-1, True),
    ])
    def test_num_workers_validation(self, dataloader_config_factory, num_workers, should_raise):
        """Test num_workers validation."""
        config = dataloader_config_factory(num_workers=num_workers)
        if should_raise:
            with pytest.raises(ValueError, match="num_workers cannot be negative"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("height,should_raise", [
        (224, False),
        (1, False),
        (0, True),
        (-10, True),
    ])
    def test_image_height_validation(self, dataloader_config_factory, height, should_raise):
        """Test image_height validation."""
        config = dataloader_config_factory(image_height=height)
        if should_raise:
            with pytest.raises(ValueError, match="image_height must be positive"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("width,should_raise", [
        (224, False),
        (1, False),
        (0, True),
        (-10, True),
    ])
    def test_image_width_validation(self, dataloader_config_factory, width, should_raise):
        """Test image_width validation."""
        config = dataloader_config_factory(image_width=width)
        if should_raise:
            with pytest.raises(ValueError, match="image_width must be positive"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("val_ratio,should_raise", [
        (0.1, False),
        (0.5, False),
        (0.99, False),
        (0.0, True),
        (1.0, True),
        (-0.1, True),
        (1.5, True),
    ])
    def test_val_ratio_validation(self, dataloader_config_factory, val_ratio, should_raise):
        """Test val_ratio validation."""
        config = dataloader_config_factory(val_ratio=val_ratio)
        if should_raise:
            with pytest.raises(ValueError, match="val_ratio must be in range"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("total_ratio,should_raise", [
        (1.0, False),
        (0.5, False),
        (0.01, False),
        (0.0, True),
        (1.1, True),
        (-0.1, True),
    ])
    def test_total_ratio_validation(self, dataloader_config_factory, total_ratio, should_raise):
        """Test total_ratio validation."""
        config = dataloader_config_factory(total_ratio=total_ratio)
        if should_raise:
            with pytest.raises(ValueError, match="total_ratio must be in range"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("skip_steps,should_raise", [
        (0, False),
        (10, False),
        (-1, True),
    ])
    def test_skip_initial_episode_steps_validation(self, dataloader_config_factory, skip_steps, should_raise):
        """Test skip_initial_episode_steps validation."""
        config = dataloader_config_factory(skip_initial_episode_steps=skip_steps)
        if should_raise:
            with pytest.raises(ValueError, match="skip_initial_episode_steps cannot be negative"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("downsample,should_raise", [
        (1, False),
        (2, False),
        (10, False),
        (0, True),
        (-1, True),
    ])
    def test_downsample_factor_validation(self, dataloader_config_factory, downsample, should_raise):
        """Test downsample_factor validation."""
        config = dataloader_config_factory(downsample_factor=downsample)
        if should_raise:
            with pytest.raises(ValueError, match="downsample_factor must be"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)

    @pytest.mark.parametrize("shift,should_raise", [
        (0, False),
        (1, False),
        (5, False),
        (-1, True),
    ])
    def test_action_backward_shift_validation(self, dataloader_config_factory, shift, should_raise):
        """Test action_backward_shift validation."""
        config = dataloader_config_factory(action_backward_shift=shift)
        if should_raise:
            with pytest.raises(ValueError, match="action_backward_shift cannot be negative"):
                validate_dataloader_config(config)
        else:
            validate_dataloader_config(config)


@pytest.mark.unit
class TestCollectDatasetPaths:
    """Test _collect_dataset_paths function."""

    def test_collect_paths_single_folder(self, episode_dir_factory):
        """Test collecting paths from a single folder."""
        folder_path, expected_paths = episode_dir_factory('test_folder', num_episodes=2)

        result = _collect_dataset_paths([folder_path], 'episode.csv')

        assert len(result) == 2
        assert all(path in result for path in expected_paths)

    def test_collect_paths_multiple_folders(self, episode_dir_factory):
        """Test collecting paths from multiple folders."""
        folder1_path, paths1 = episode_dir_factory('folder1', num_episodes=2)
        folder2_path, paths2 = episode_dir_factory('folder2', num_episodes=2)

        result = _collect_dataset_paths([folder1_path, folder2_path], 'episode.csv')

        assert len(result) == 4

    def test_collect_paths_empty_folder(self, temp_dir):
        """Test collecting from empty folder."""
        empty_folder = Path(temp_dir) / 'empty'
        empty_folder.mkdir()

        result = _collect_dataset_paths([str(empty_folder)], 'episode.csv')

        assert len(result) == 0

    def test_collect_paths_nonexistent_folder(self):
        """Test collecting from nonexistent folder raises error."""
        with pytest.raises(FileNotFoundError):
            _collect_dataset_paths(['/nonexistent/path'], 'episode.csv')

    def test_collect_paths_mixed_valid_invalid(self, episode_dir_factory, temp_dir):
        """Test collecting from mix of valid and invalid folders raises error."""
        valid_folder, _ = episode_dir_factory('valid', num_episodes=2)
        invalid_folder = '/nonexistent/path'

        with pytest.raises(FileNotFoundError):
            _collect_dataset_paths([valid_folder, invalid_folder], 'episode.csv')


@pytest.mark.unit
class TestEnsureZarrExists:
    """Test _ensure_zarr_exists function."""

    def test_skip_creation_when_zarr_exists(self, temp_dir, mock_schema_factory):
        """Test that zarr creation is skipped when valid zarr exists."""
        zarr_path = Path(temp_dir) / 'test.zarr'
        schema = mock_schema_factory(str(zarr_path))

        with patch('refactoring.data.dataloader.ReplayBuffer.copy_from_path') as mock_copy, \
             patch('refactoring.data.dataloader.create_replay_buffer') as mock_create:
            zarr_path.mkdir()
            _ensure_zarr_exists(schema, [])

            mock_copy.assert_called_once()
            mock_create.assert_not_called()

    def test_recreate_zarr_on_load_error(self, temp_dir, mock_schema_factory):
        """Test that zarr is recreated when loading fails."""
        zarr_path = Path(temp_dir) / 'test.zarr'
        schema = mock_schema_factory(str(zarr_path))

        with patch('refactoring.data.dataloader.ReplayBuffer.copy_from_path', side_effect=Exception("Load error")), \
             patch('refactoring.data.dataloader.create_replay_buffer') as mock_create:
            zarr_path.mkdir()
            _ensure_zarr_exists(schema, ['/path/to/episode.csv'])

            mock_create.assert_called_once()


@pytest.mark.unit
class TestLogPhaseDistributions:
    """Test _log_phase_distributions function."""

    def test_log_phase_distributions_basic(self):
        """Test logging phase distributions."""
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        train_dataset.replay_buffer = {'phase': [0, 0, 1, 1, 2]}
        val_dataset.replay_buffer = {'phase': [0, 1, 2]}

        _log_phase_distributions(train_dataset, val_dataset)

    def test_log_phase_distributions_no_selected_episodes(self):
        """Test when no episodes selected."""
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        train_dataset.replay_buffer = {}
        val_dataset.replay_buffer = {}
        train_dataset.selected_episodes_indices = []
        val_dataset.selected_episodes_indices = []

        _log_phase_distributions(train_dataset, val_dataset)

    def test_log_phase_distributions_unbalanced(self):
        """Test with unbalanced phase distribution."""
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        train_dataset.replay_buffer = {'phase': [0] * 100 + [1] * 10}
        val_dataset.replay_buffer = {'phase': [0] * 20}

        _log_phase_distributions(train_dataset, val_dataset)