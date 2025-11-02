"""End-to-end tests for the training endpoint."""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.main import MainConfig
from refactoring.configs.task.task import TaskConfig, ActionSpace, ObservationSpace
from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.configs.training import TrainingConfig, OptimizerConfig, AdamWConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.workspace import Workspace
from refactoring.data.constants import Cameras, OrientationRepresentation


@pytest.mark.unit
class TestTrainingEndpointConfigSaving:
    """Test that training endpoint saves config properly."""

    def test_workspace_saves_config_on_init(self, tmp_path):
        """Test that workspace automatically saves config.yaml during initialization."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="train_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        # Verify config.yaml exists
        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists(), "config.yaml should be created during workspace init"

    def test_saved_config_contains_all_sections(self, tmp_path):
        """Test that saved config contains all required sections."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="sections_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                seed=42,
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                    has_orientation=True,
                    orientation_dim=4,
                    orientation_repr=OrientationRepresentation.QUATERNION.value,
                ),
                observation_horizon=2,
                prediction_horizon=16,
                dataloader=DataloaderConfig(batch_size=32),
            ),
            training=TrainingConfig(
                num_epochs=100,
                use_ema=True,
                optimizer=AdamWConfig(lr=1e-4),
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        config_path = workspace.output_dir / "config.yaml"

        # Load and verify all sections
        loaded = OmegaConf.load(config_path)

        assert "experiment" in loaded
        assert "task" in loaded
        assert "training" in loaded
        assert "policy" in loaded
        assert "inference" in loaded

        # Verify some key values
        assert loaded.experiment.seed == 42
        assert loaded.task.observation_horizon == 2
        assert loaded.task.prediction_horizon == 16
        assert loaded.training.num_epochs == 100
        assert loaded.training.use_ema is True

    def test_config_matches_original_after_save_load(self, tmp_path):
        """Test that config round-trips correctly through save/load cycle."""
        original_config = MainConfig(
            experiment=ExperimentConfig(
                name="roundtrip_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                seed=123,
                val_every=5,
                checkpoint_every=10,
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                    use_proprio_camera_frame=False,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=8,
                dataloader=DataloaderConfig(
                    batch_size=16,
                    num_workers=4,
                ),
            ),
            training=TrainingConfig(
                num_epochs=50,
                gradient_accumulate_every=2,
                use_ema=False,
                optimizer=AdamWConfig(
                    lr=5e-5,
                    weight_decay=1e-5,
                ),
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(original_config)
        config_path = workspace.output_dir / "config.yaml"

        # Reload config
        loaded_config = OmegaConf.load(config_path)

        # Verify key fields match
        assert loaded_config.experiment.name == original_config.experiment.name
        assert loaded_config.experiment.seed == original_config.experiment.seed
        assert loaded_config.experiment.val_every == original_config.experiment.val_every
        assert loaded_config.task.observation_horizon == original_config.task.observation_horizon
        assert loaded_config.task.prediction_horizon == original_config.task.prediction_horizon
        assert loaded_config.task.dataloader.batch_size == original_config.task.dataloader.batch_size
        assert loaded_config.training.num_epochs == original_config.training.num_epochs
        assert loaded_config.training.gradient_accumulate_every == original_config.training.gradient_accumulate_every


@pytest.mark.unit
class TestTrainingEndpointCheckpointing:
    """Test that training endpoint handles checkpointing correctly."""

    def test_checkpoint_directory_created(self, tmp_path):
        """Test that checkpoint directory is created."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="ckpt_dir_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        # Verify output directory exists
        assert workspace.output_dir.exists()
        assert workspace.output_dir.is_dir()

        # Verify it's in the right location
        expected_path = Path(tmp_path) / "ckpt_dir_test"
        assert workspace.output_dir == expected_path

    def test_config_saved_before_training(self, tmp_path):
        """Test that config.yaml is saved before any training begins."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="save_before_train_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        # Config should be saved during __init__, before any other setup
        workspace = Workspace(config)

        # Verify config exists immediately
        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists()
        assert config_path.parent == workspace.output_dir

        # Verify policy/trainer haven't been initialized yet
        assert workspace.policy is None
        assert workspace.trainer is None
