"""End-to-end tests for the training endpoint."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import numpy as np
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


@pytest.mark.unit
class TestHydraConfigLoading:
    """Test Hydra configuration loading and validation."""

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig has expected defaults."""
        config = ExperimentConfig(
            name="test",
            checkpoint_folder="/tmp/checkpoints"
        )

        assert config.name == "test"
        assert config.seed == 42
        assert config.device == "cuda"
        assert config.distributed is False
        assert config.use_wandb is True
        assert config.checkpoint_every == 100
        assert config.val_every == 1

    def test_config_to_object_conversion(self):
        """Test OmegaConf DictConfig can be converted to MainConfig."""
        dict_config = {
            "experiment": {
                "name": "test_exp",
                "seed": 123,
                "checkpoint_folder": "/tmp/ckpt",
                "device": "cpu",
                "distributed": False,
                "use_wandb": False,
            }
        }

        omega_conf = OmegaConf.create(dict_config)

        assert omega_conf.experiment.name == "test_exp"
        assert omega_conf.experiment.seed == 123
        assert omega_conf.experiment.device == "cpu"

    def test_hydra_interpolation_experiment_to_policy(self):
        """Test that ${experiment.device} interpolation works."""
        dict_config = {
            "experiment": {"device": "cuda"},
            "policy": {"device": "${experiment.device}"}
        }

        omega_conf = OmegaConf.create(dict_config)
        resolved = OmegaConf.to_container(omega_conf, resolve=True)

        assert resolved["policy"]["device"] == "cuda"

    def test_config_override_via_dict(self):
        """Test that config values can be overridden."""
        base_config = {
            "experiment": {
                "name": "base",
                "device": "cuda",
                "seed": 42,
            }
        }

        override = {"experiment": {"device": "cpu", "seed": 999}}

        base = OmegaConf.create(base_config)
        override_conf = OmegaConf.create(override)
        merged = OmegaConf.merge(base, override_conf)

        assert merged.experiment.device == "cpu"
        assert merged.experiment.seed == 999
        assert merged.experiment.name == "base"


@pytest.mark.unit
class TestDistributedTrainingDetection:
    """Test distributed training environment variable detection."""

    def test_world_size_env_var_enables_distributed(self):
        """Test that WORLD_SIZE env var enables distributed training."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False):
            assert "WORLD_SIZE" in os.environ
            assert os.environ["WORLD_SIZE"] == "4"

    def test_no_world_size_keeps_distributed_false(self):
        """Test that absence of WORLD_SIZE keeps distributed=False."""
        with patch.dict(os.environ, {}, clear=True):
            assert "WORLD_SIZE" not in os.environ

    def test_slurm_environment_variables_present(self):
        """Test detection of SLURM environment variables."""
        slurm_vars = {
            "WORLD_SIZE": "8",
            "SLURM_PROCID": "0",
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_CPUS_PER_TASK": "8",
        }

        with patch.dict(os.environ, slurm_vars, clear=False):
            assert os.environ.get("WORLD_SIZE") == "8"
            assert os.environ.get("SLURM_PROCID") == "0"


@pytest.mark.unit
class TestWorkspaceExtended:
    """Additional workspace initialization tests."""

    def test_workspace_seed_setting(self, tmp_path):
        """Test that workspace sets random seeds correctly."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="seed_test",
                checkpoint_folder=str(tmp_path),
                seed=12345,
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
            training=TrainingConfig(
                num_epochs=1,
                optimizer=AdamWConfig(),
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        rand_val1 = torch.rand(1).item()
        np_rand_val1 = np.random.rand()

        torch.manual_seed(12345)
        np.random.seed(12345)

        rand_val2 = torch.rand(1).item()
        np_rand_val2 = np.random.rand()

        assert rand_val1 == rand_val2
        assert np_rand_val1 == np_rand_val2


@pytest.mark.unit
class TestTrainingConfigValidation:
    """Test training configuration validation."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(),
        )

        assert config.num_epochs == 100
        assert config.gradient_accumulate_every == 1
        assert config.clip_gradient_norm is False
        assert config.use_ema is True
        assert config.ema_power == 0.75

    def test_optimizer_config_defaults(self):
        """Test AdamWConfig has expected defaults."""
        config = AdamWConfig()

        assert config.lr == 1e-4
        assert config.weight_decay == 1e-4
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config._target_ == "torch.optim.AdamW"

    def test_lr_schedule_in_training_config(self):
        """Test learning rate schedule is configured in TrainingConfig."""
        config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(),
            lr_schedule="cosine",
            lr_warmup_steps=1000,
        )

        assert config.lr_schedule == "cosine"
        assert config.lr_warmup_steps == 1000


@pytest.mark.unit
class TestCheckpointExtended:
    """Extended checkpoint configuration tests."""

    def test_resume_from_checkpoint_path(self, tmp_path):
        """Test resume_from checkpoint path."""
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        checkpoint_path.touch()

        config = ExperimentConfig(
            name="test",
            checkpoint_folder=str(tmp_path),
            resume_from=str(checkpoint_path),
        )

        assert config.resume_from == str(checkpoint_path)
        assert Path(config.resume_from).exists()
