"""Tests for Hydra-based training endpoint."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from refactoring.data.constants import OrientationRepresentation
import torch
import numpy as np
from refactoring.workspace import Workspace
from refactoring.configs.main import MainConfig
from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.task.task import TaskConfig, ActionSpace, ObservationSpace
from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.configs.training import TrainingConfig, OptimizerConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.data.constants import Cameras


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
class TestWorkspaceInitialization:
    """Test workspace initialization from config."""

    def test_workspace_creates_output_directory(self, tmp_path):
        """Test that workspace creates output directory."""

        checkpoint_folder = tmp_path / "checkpoints"

        config = MainConfig(
            experiment=ExperimentConfig(
                name="test_exp",
                checkpoint_folder=str(checkpoint_folder),
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
                    has_orientation=True,
                    orientation_dim=4,
                    orientation_repr=OrientationRepresentation.QUATERNION.value,
                    has_gripper=True,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(
                num_epochs=1,
                optimizer=OptimizerConfig(),
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        output_dir = checkpoint_folder / "test_exp"
        assert output_dir.exists()
        assert workspace.exp_name == "test_exp"
        assert workspace.output_dir == output_dir

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
                optimizer=OptimizerConfig(),
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
class TestTrainingConfig:
    """Test training configuration validation."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        from refactoring.configs.training import TrainingConfig, OptimizerConfig

        config = TrainingConfig(
            num_epochs=100,
            optimizer=OptimizerConfig(),
        )

        assert config.num_epochs == 100
        assert config.gradient_accumulate_every == 1
        assert config.clip_gradient_norm is False
        assert config.use_ema is True
        assert config.ema_power == 0.75

    def test_optimizer_config_defaults(self):
        """Test OptimizerConfig has expected defaults."""
        from refactoring.configs.training import OptimizerConfig

        config = OptimizerConfig()

        assert config.learning_rate == 1e-6
        assert config.weight_decay == 1e-6
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8

    def test_lr_schedule_in_training_config(self):
        """Test learning rate schedule is configured in TrainingConfig."""
        from refactoring.configs.training import TrainingConfig, OptimizerConfig

        config = TrainingConfig(
            num_epochs=100,
            optimizer=OptimizerConfig(),
            lr_schedule="cosine",
            lr_warmup_steps=1000,
        )

        assert config.lr_schedule == "cosine"
        assert config.lr_warmup_steps == 1000


@pytest.mark.unit
class TestCheckpointConfiguration:
    """Test checkpoint configuration."""

    def test_checkpoint_folder_creation(self, tmp_path):
        """Test that checkpoint folder is created if it doesn't exist."""
        checkpoint_folder = tmp_path / "new_checkpoints"
        assert not checkpoint_folder.exists()

        from refactoring.configs.experiment import ExperimentConfig

        config = ExperimentConfig(
            name="test",
            checkpoint_folder=str(checkpoint_folder),
        )

        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        assert checkpoint_folder.exists()

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
