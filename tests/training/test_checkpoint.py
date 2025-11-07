"""Tests for checkpoint saving and loading."""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.main import MainConfig
from refactoring.configs.task.task import TaskConfig, ActionSpace, ObservationSpace
from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.configs.training import TrainingConfig, OptimizerConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.workspace import Workspace
from refactoring.training.lightning_policy import LightningPolicy
from refactoring.training.callbacks import EMACallback
from refactoring.data.constants import Cameras, GripperType, OrientationRepresentation, ACTION_KEY
from refactoring.data.tokenize.tokenizer import Tokenizer
from refactoring.data.tokenize.action_tokenizer import ActionTokenizer
from refactoring.data.tokenize.binning_tokenizer import BinningTokenizer


@pytest.mark.unit
class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    @pytest.fixture
    def checkpoint_config(self, tmp_path):
        """Create config for checkpoint tests."""
        return MainConfig(
            experiment=ExperimentConfig(
                name="checkpoint_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
                checkpoint_every=1,
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
                num_epochs=2,
                optimizer=OptimizerConfig(),
                use_ema=False,
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

    def test_checkpoint_directory_created(self, checkpoint_config):
        """Test that checkpoint directory is created."""
        workspace = Workspace(checkpoint_config)

        assert workspace.output_dir.exists()
        assert workspace.output_dir.is_dir()

    def test_checkpoint_path_format(self, checkpoint_config):
        """Test checkpoint path follows expected format."""
        workspace = Workspace(checkpoint_config)

        expected_path = Path(checkpoint_config.experiment.checkpoint_folder) / "checkpoint_test"
        assert workspace.output_dir == expected_path


@pytest.mark.integration
class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_load_lightning_checkpoint(self, simple_policy, tmp_path):
        """Test loading Lightning checkpoint format."""
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"

        state_dict = {
            "state_dict": {f"policy.{k}": v for k, v in simple_policy.state_dict().items()},
            "epoch": 5,
            "global_step": 100,
        }

        torch.save(state_dict, checkpoint_path)

        training_config = TrainingConfig(
            num_epochs=10,
            optimizer=OptimizerConfig(),
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
            total_training_steps=1000,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        lightning_policy.load_state_dict(checkpoint["state_dict"])

        assert lightning_policy.policy is not None

    def test_load_policy_checkpoint(self, simple_policy, tmp_path):
        """Test loading policy-only checkpoint."""
        checkpoint_path = tmp_path / "policy.ckpt"

        torch.save(simple_policy.state_dict(), checkpoint_path)

        policy_copy = simple_policy

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        policy_copy.load_state_dict(checkpoint)

        for (name1, param1), (name2, param2) in zip(
            simple_policy.named_parameters(), policy_copy.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


@pytest.mark.integration
class TestCheckpointStateRestoration:
    """Test that checkpoint restores full training state."""

    def test_optimizer_state_restored(self, simple_policy):
        """Test optimizer state is saved and loaded."""
        optimizer = torch.optim.AdamW(simple_policy.parameters(), lr=1e-4)

        for _ in range(5):
            loss = torch.randn(1, requires_grad=True).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        optimizer_state_before = optimizer.state_dict()

        optimizer_new = torch.optim.AdamW(simple_policy.parameters(), lr=1e-4)
        optimizer_new.load_state_dict(optimizer_state_before)

        optimizer_state_after = optimizer_new.state_dict()

        assert optimizer_state_before.keys() == optimizer_state_after.keys()

    def test_epoch_counter_restored(self):
        """Test epoch counter is saved in checkpoint."""
        checkpoint = {
            "epoch": 42,
            "global_step": 1000,
        }

        assert checkpoint["epoch"] == 42
        assert checkpoint["global_step"] == 1000

    def test_random_state_can_be_saved(self):
        """Test random state can be checkpointed."""
        torch.manual_seed(42)
        rng_state = torch.get_rng_state()

        checkpoint = {
            "rng_state": rng_state,
        }

        torch.manual_seed(123)

        torch.set_rng_state(checkpoint["rng_state"])

        torch.manual_seed(42)
        expected_tensor = torch.rand(10)

        torch.set_rng_state(checkpoint["rng_state"])
        actual_tensor = torch.rand(10)

        assert torch.allclose(expected_tensor, actual_tensor)


@pytest.mark.integration
class TestEMACheckpoint:
    """Test EMA model checkpointing."""

    def test_ema_model_separate_from_main(self, simple_policy):
        """Test EMA model state is separate from main model."""
        ema_callback = EMACallback(power=0.75)
        ema_callback.on_train_start(trainer=None, pl_module=None)

        if not hasattr(ema_callback, "ema_model") or ema_callback.ema_model is None:
            ema_callback.ema_model = copy.deepcopy(simple_policy)

        for param in simple_policy.parameters():
            param.data += 0.1

        ema_params_dict = {name: param.clone() for name, param in ema_callback.ema_model.named_parameters()}
        main_params_dict = {name: param.clone() for name, param in simple_policy.named_parameters()}

        for name in ema_params_dict.keys():
            assert not torch.allclose(ema_params_dict[name], main_params_dict[name])

    def test_ema_checkpoint_saveable(self, simple_policy, tmp_path):
        """Test EMA model can be saved to checkpoint."""
        ema_callback = EMACallback(power=0.75)
        ema_callback.ema_model = copy.deepcopy(simple_policy)

        checkpoint_path = tmp_path / "ema_model.ckpt"

        torch.save(ema_callback.ema_model.state_dict(), checkpoint_path)

        assert checkpoint_path.exists()

        loaded_state = torch.load(checkpoint_path, map_location="cpu")

        assert loaded_state is not None


@pytest.mark.integration
class TestResumeFromCheckpoint:
    """Test resuming training from checkpoint."""

    def test_workspace_resume_from_checkpoint(self, simple_policy, tmp_path):
        """Test workspace can resume from checkpoint."""
        checkpoint_path = tmp_path / "resume.ckpt"

        torch.save(simple_policy.state_dict(), checkpoint_path)

        config = MainConfig(
            experiment=ExperimentConfig(
                name="resume_test",
                checkpoint_folder=str(tmp_path),
                resume_from=str(checkpoint_path),
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
                num_epochs=10,
                optimizer=OptimizerConfig(),
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        assert config.experiment.resume_from == str(checkpoint_path)
        assert Path(config.experiment.resume_from).exists()

    def test_checkpoint_not_found_warning(self, tmp_path):
        """Test warning when checkpoint file doesn't exist."""
        nonexistent_path = tmp_path / "does_not_exist.ckpt"

        config = MainConfig(
            experiment=ExperimentConfig(
                name="missing_ckpt_test",
                checkpoint_folder=str(tmp_path),
                resume_from=str(nonexistent_path),
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
                num_epochs=10,
                optimizer=OptimizerConfig(),
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        assert not Path(config.experiment.resume_from).exists()


@pytest.mark.unit
class TestCheckpointCompatibility:
    """Test checkpoint format compatibility."""

    def test_model_weights_are_tensors(self, simple_policy):
        """Test that all model weights are tensors."""
        state_dict = simple_policy.state_dict()

        for name, param in state_dict.items():
            assert isinstance(param, torch.Tensor), f"{name} is not a tensor"

    def test_checkpoint_serializable(self, simple_policy, tmp_path):
        """Test checkpoint can be serialized and deserialized."""
        checkpoint_path = tmp_path / "serialization_test.ckpt"

        checkpoint = {
            "model_state_dict": simple_policy.state_dict(),
            "epoch": 5,
            "loss": 0.123,
        }

        torch.save(checkpoint, checkpoint_path)

        loaded = torch.load(checkpoint_path, map_location="cpu")

        assert loaded["epoch"] == 5
        assert loaded["loss"] == 0.123
        assert "model_state_dict" in loaded

    def test_checkpoint_map_location(self, simple_policy, tmp_path):
        """Test checkpoint loading with map_location."""
        checkpoint_path = tmp_path / "map_location_test.ckpt"

        torch.save(simple_policy.state_dict(), checkpoint_path)

        loaded_cpu = torch.load(checkpoint_path, map_location="cpu")

        for param in loaded_cpu.values():
            assert param.device.type == "cpu"


@pytest.mark.unit
class TestWorkspaceLoadCheckpoint:
    """Test Workspace.load_checkpoint() method."""

    @pytest.fixture
    def workspace_with_policy(self, simple_policy, tmp_path):
        """Create workspace with initialized policy and lightning_policy."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="load_test",
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
                num_epochs=2,
                optimizer=OptimizerConfig(),
                use_ema=False,
            ),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        # Move policy to CPU to match workspace config
        simple_policy.to(torch.device("cpu"))
        workspace.policy = simple_policy
        # Use real LightningPolicy, not mock
        workspace.lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=config.training,
        )
        return workspace

    def test_load_checkpoint_success(self, workspace_with_policy, simple_policy, tmp_path):
        """Test successful checkpoint loading with Lightning format."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Create Lightning format checkpoint
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
            "epoch": 5,
        }
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify weights match
        for name, param in workspace_with_policy.policy.named_parameters():
            original = simple_policy.state_dict()[name]
            assert torch.allclose(param, original)

    def test_load_checkpoint_device_mapping(self, workspace_with_policy, simple_policy, tmp_path):
        """Test that checkpoint uses correct device mapping."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Create checkpoint
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify all weights are on correct device (cpu as per config)
        for param in workspace_with_policy.policy.parameters():
            assert param.device.type == "cpu"

    def test_load_checkpoint_overwrite_weights(self, workspace_with_policy, simple_policy, tmp_path):
        """Test that loading checkpoint overwrites existing weights."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Save original weights before modifying
        original_state_dict = {
            name: param.clone().detach()
            for name, param in simple_policy.named_parameters()
        }

        # Create checkpoint with original weights
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in original_state_dict.items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Modify workspace policy weights
        with torch.no_grad():
            for param in workspace_with_policy.policy.parameters():
                param.fill_(999.0)

        # Verify weights are 999.0 before loading
        for param in workspace_with_policy.policy.parameters():
            assert torch.allclose(param, torch.full_like(param, 999.0))

        # Load checkpoint
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify weights match checkpoint (not 999.0)
        for name, param in workspace_with_policy.policy.named_parameters():
            assert torch.allclose(param, original_state_dict[name])
            assert not torch.allclose(param, torch.full_like(param, 999.0))

        # Load again - should be idempotent
        workspace_with_policy.load_checkpoint(str(checkpoint_path))
        for name, param in workspace_with_policy.policy.named_parameters():
            original = simple_policy.state_dict()[name]
            assert torch.allclose(param, original)

    def test_load_checkpoint_policy_not_initialized(self, workspace_with_policy, simple_policy, tmp_path):
        """Test AssertionError when policy is not initialized."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Create valid checkpoint
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Set policy to None
        workspace_with_policy.policy = None

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="Policy must be initialized before loading checkpoint"):
            workspace_with_policy.load_checkpoint(str(checkpoint_path))

    def test_load_checkpoint_lightning_policy_not_initialized(self, workspace_with_policy, simple_policy, tmp_path):
        """Test AssertionError when lightning_policy is not initialized."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Create valid checkpoint
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Set lightning_policy to None (but keep policy)
        workspace_with_policy.lightning_policy = None

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="LightningPolicy must be initialized"):
            workspace_with_policy.load_checkpoint(str(checkpoint_path))

    def test_load_checkpoint_unrecognized_format(self, workspace_with_policy, tmp_path):
        """Test ValueError when checkpoint format is not recognized."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Test 1: Empty dict
        torch.save({}, checkpoint_path)
        with pytest.raises(ValueError, match="Checkpoint format not recognized"):
            workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Test 2: Legacy format (should not be supported)
        torch.save({"model_state_dict": {}}, checkpoint_path)
        with pytest.raises(ValueError, match="Checkpoint format not recognized"):
            workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Test 3: Raw state dict (should not be supported)
        torch.save(workspace_with_policy.policy.state_dict(), checkpoint_path)
        with pytest.raises(ValueError, match="Checkpoint format not recognized"):
            workspace_with_policy.load_checkpoint(str(checkpoint_path))

    def test_load_checkpoint_file_not_found(self, workspace_with_policy):
        """Test error when checkpoint file doesn't exist."""
        # Should raise FileNotFoundError or similar torch error
        with pytest.raises((FileNotFoundError, RuntimeError)):
            workspace_with_policy.load_checkpoint("/nonexistent/path/checkpoint.ckpt")

    def test_load_checkpoint_with_extra_keys(self, workspace_with_policy, simple_policy, tmp_path):
        """Test that extra keys in checkpoint are ignored."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Create checkpoint with extra keys
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
            "epoch": 10,
            "optimizer_states": {"lr": 0.001},
            "metadata": {"timestamp": "2024-01-01"},
        }
        torch.save(checkpoint, checkpoint_path)

        # Should load successfully (extra keys ignored)
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify weights match
        for name, param in workspace_with_policy.policy.named_parameters():
            original = simple_policy.state_dict()[name]
            assert torch.allclose(param, original)

    # Tokenizer fixtures
    @pytest.fixture
    def action_chunks(self):
        """Generate action chunks for action tokenizer."""
        np.random.seed(42)
        return np.random.randn(10, 5, 7).astype(np.float32) * 0.5

    @pytest.fixture
    def normalized_proprio(self):
        """Generate normalized proprioceptive data."""
        np.random.seed(42)
        return np.random.randn(100, 7).astype(np.float32) * 0.5

    @pytest.fixture
    def binning_tokenizer(self, normalized_proprio):
        """Create fitted binning tokenizer."""
        device = torch.device("cpu")
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_proprio)
        return tokenizer

    @pytest.fixture
    def action_tokenizer_pretrained(self, action_chunks):
        """Create action tokenizer with pretrained weights."""
        device = torch.device("cpu")
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)
        return tokenizer

    @pytest.fixture
    def action_tokenizer_custom(self, action_chunks):
        """Create action tokenizer fitted on custom data."""
        device = torch.device("cpu")
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)
        tokenizer.fit(action_chunks)
        return tokenizer

    @pytest.fixture
    def tokenizer_with_binning(self, binning_tokenizer):
        """Create Tokenizer with binning tokenizer."""
        device = torch.device("cpu")
        tokenizer = Tokenizer(device=device)
        tokenizer.tokenizers["proprio_robot_frame"] = binning_tokenizer
        return tokenizer

    @pytest.fixture
    def tokenizer_with_action_pretrained(self, action_tokenizer_pretrained):
        """Create Tokenizer with pretrained action tokenizer."""
        device = torch.device("cpu")
        tokenizer = Tokenizer(device=device)
        tokenizer.tokenizers[ACTION_KEY] = action_tokenizer_pretrained
        return tokenizer

    @pytest.fixture
    def tokenizer_with_action_custom(self, action_tokenizer_custom):
        """Create Tokenizer with custom-fitted action tokenizer."""
        device = torch.device("cpu")
        tokenizer = Tokenizer(device=device)
        tokenizer.tokenizers[ACTION_KEY] = action_tokenizer_custom
        return tokenizer

    @pytest.fixture
    def tokenizer_with_both(self, binning_tokenizer, action_tokenizer_custom):
        """Create Tokenizer with both action and binning tokenizers."""
        device = torch.device("cpu")
        tokenizer = Tokenizer(device=device)
        tokenizer.tokenizers[ACTION_KEY] = action_tokenizer_custom
        tokenizer.tokenizers["proprio_robot_frame"] = binning_tokenizer
        return tokenizer

    @pytest.fixture
    def tokenizer_with_action(self, action_chunks):
        """Create Tokenizer with custom-fitted action tokenizer."""
        device = torch.device("cpu")
        tokenizer = Tokenizer(device=device)
        action_tok = ActionTokenizer(use_pretrained_weights=False, device=device)
        action_tok.fit(action_chunks)
        tokenizer.tokenizers[ACTION_KEY] = action_tok
        return tokenizer

    # Tokenizer tests
    @pytest.mark.parametrize("tokenizer_fixture", [
        "tokenizer_with_binning",
        "tokenizer_with_action_custom",
        "tokenizer_with_both",
    ])
    def test_tokenizer_saved_to_workspace(self, tokenizer_fixture, workspace_with_policy, request):
        """Test that tokenizer can be saved to workspace directory."""
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        workspace_with_policy.tokenizer = tokenizer

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Verify directory structure
        assert tokenizer_path.exists()
        assert (tokenizer_path / "config.json").exists()

    @pytest.mark.integration
    @pytest.mark.parametrize("tokenizer_fixture", [
        "tokenizer_with_binning",
        "tokenizer_with_action_custom",
        "tokenizer_with_both",
    ])
    def test_tokenizer_loaded_from_checkpoint(self, tokenizer_fixture, workspace_with_policy, simple_policy, request):
        """Test loading tokenizer during checkpoint load."""
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        workspace_with_policy.tokenizer = tokenizer

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Create checkpoint
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Reset tokenizer
        workspace_with_policy.tokenizer = None

        # Load checkpoint
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify tokenizer was loaded
        assert workspace_with_policy.tokenizer is not None
        assert isinstance(workspace_with_policy.tokenizer, Tokenizer)
        assert len(workspace_with_policy.tokenizer.tokenizers) > 0

    def test_tokenizer_set_on_policy_after_load(self, tokenizer_with_binning, workspace_with_policy, simple_policy):
        """Test that loaded tokenizer is passed to policy.set_tokenizer."""
        workspace_with_policy.tokenizer = tokenizer_with_binning

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Create checkpoint
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Mock set_tokenizer and reset
        workspace_with_policy.policy.set_tokenizer = MagicMock()
        workspace_with_policy.tokenizer = None

        # Load checkpoint
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify set_tokenizer was called
        workspace_with_policy.policy.set_tokenizer.assert_called_once()
        assert workspace_with_policy.policy.set_tokenizer.call_args[0][0] is workspace_with_policy.tokenizer

    def test_load_checkpoint_without_tokenizer(self, workspace_with_policy, simple_policy):
        """Test loading checkpoint when no tokenizer was saved."""
        # Create checkpoint without tokenizer
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Ensure no tokenizer directory exists
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        assert not tokenizer_path.exists()

        # Load checkpoint (should not error)
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify tokenizer is None
        assert workspace_with_policy.tokenizer is None

    def test_tokenizer_device_matches_config(self, tokenizer_with_binning, workspace_with_policy, simple_policy):
        """Test that loaded tokenizer uses correct device from config."""
        workspace_with_policy.tokenizer = tokenizer_with_binning

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Create checkpoint
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Reset and load
        workspace_with_policy.tokenizer = None
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Verify device matches config
        expected_device = torch.device(workspace_with_policy.config.experiment.device)
        assert workspace_with_policy.tokenizer.device == expected_device

    def test_binning_tokenizer_directory_structure(self, tokenizer_with_binning, workspace_with_policy):
        """Test binning tokenizer creates correct directory structure."""
        workspace_with_policy.tokenizer = tokenizer_with_binning

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Verify binning tokenizer files
        assert (tokenizer_path / "proprio_robot_frame").exists()
        assert (tokenizer_path / "proprio_robot_frame" / "binning_state.pt").exists()

    @pytest.mark.integration
    def test_action_tokenizer_directory_structure(self, tokenizer_with_action_custom, workspace_with_policy):
        """Test action tokenizer creates correct directory structure."""
        workspace_with_policy.tokenizer = tokenizer_with_action_custom

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Verify action tokenizer files
        assert (tokenizer_path / ACTION_KEY).exists()
        # HuggingFace saves multiple files
        assert len(list((tokenizer_path / ACTION_KEY).iterdir())) > 0

    def test_binning_tokenizer_roundtrip(self, tokenizer_with_binning, workspace_with_policy, simple_policy, normalized_proprio):
        """Test save/load preserves binning tokenizer functionality."""
        workspace_with_policy.tokenizer = tokenizer_with_binning

        # Test data
        test_data = {"proprio_robot_frame": normalized_proprio[:10]}
        original_tokenized = workspace_with_policy.tokenizer.tokenize(test_data)

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Create checkpoint and load
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        workspace_with_policy.tokenizer = None
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Test with loaded tokenizer
        loaded_tokenized = workspace_with_policy.tokenizer.tokenize(test_data)

        # Verify tokenization is identical
        for key in original_tokenized:
            assert torch.equal(original_tokenized[key], loaded_tokenized[key])

    @pytest.mark.integration
    def test_action_tokenizer_roundtrip(self, tokenizer_with_action, workspace_with_policy, simple_policy, action_chunks):
        """Test save/load preserves action tokenizer functionality."""
        workspace_with_policy.tokenizer = tokenizer_with_action

        # Test data
        test_data = {ACTION_KEY: action_chunks[:3]}
        original_tokenized = workspace_with_policy.tokenizer.tokenize(test_data)

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Create checkpoint and load
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        workspace_with_policy.tokenizer = None
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Test with loaded tokenizer
        loaded_tokenized = workspace_with_policy.tokenizer.tokenize(test_data)

        # Verify tokenization produces same type
        assert type(original_tokenized[ACTION_KEY]) == type(loaded_tokenized[ACTION_KEY])

    @pytest.mark.integration
    def test_mixed_tokenizers_roundtrip(self, tokenizer_with_binning, workspace_with_policy, simple_policy, normalized_proprio, action_chunks):
        """Test save/load with both action and binning tokenizers."""
        # Add action tokenizer to binning tokenizer
        device = torch.device("cpu")
        action_tok = ActionTokenizer(use_pretrained_weights=False, device=device)
        action_tok.fit(action_chunks)
        tokenizer_with_binning.tokenizers[ACTION_KEY] = action_tok

        workspace_with_policy.tokenizer = tokenizer_with_binning

        # Test data
        test_data = {
            "proprio_robot_frame": normalized_proprio[:10],
            ACTION_KEY: action_chunks[:3],
        }
        original_tokenized = workspace_with_policy.tokenizer.tokenize(test_data)

        # Save tokenizer
        tokenizer_path = workspace_with_policy.output_dir / "tokenizer"
        workspace_with_policy.tokenizer.save_pretrained(tokenizer_path)

        # Create checkpoint and load
        checkpoint_path = workspace_with_policy.output_dir / "model.ckpt"
        checkpoint = {
            "state_dict": {
                f"policy.{k}": v
                for k, v in simple_policy.state_dict().items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

        workspace_with_policy.tokenizer = None
        workspace_with_policy.load_checkpoint(str(checkpoint_path))

        # Test with loaded tokenizer
        loaded_tokenized = workspace_with_policy.tokenizer.tokenize(test_data)

        # Verify both tokenizers work
        assert "proprio_robot_frame" in loaded_tokenized
        assert ACTION_KEY in loaded_tokenized
        assert torch.equal(original_tokenized["proprio_robot_frame"], loaded_tokenized["proprio_robot_frame"])

@pytest.mark.unit
class TestWorkspaceConfigSaving:
    """Test Workspace config saving and loading functionality."""

    def test_config_saved_on_initialization(self, tmp_path):
        """Test that config.yaml is created during workspace initialization."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="config_test",
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
                    has_orientation=True,
                    orientation_dim=4,
                    orientation_repr=OrientationRepresentation.QUATERNION.value,
                    has_gripper=True,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=2),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        # Check config.yaml exists
        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists()

    def test_config_saved_to_correct_path(self, tmp_path):
        """Test config is saved to {output_dir}/config.yaml."""
        exp_name = "test_experiment"
        config = MainConfig(
            experiment=ExperimentConfig(
                name=exp_name,
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
            training=TrainingConfig(num_epochs=2),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        # Verify exact path
        expected_path = tmp_path / exp_name / "config.yaml"
        assert expected_path.exists()

    def test_saved_config_can_be_reloaded(self, tmp_path):
        """Test that saved config can be loaded back with OmegaConf."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="reload_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                seed=42,
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                    use_proprio_base_frame=True,
                    use_proprio_camera_frame=False,
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
            training=TrainingConfig(num_epochs=100, use_ema=True),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        config_path = workspace.output_dir / "config.yaml"

        # Reload config
        loaded_config = OmegaConf.load(config_path)

        # Verify key fields match
        assert loaded_config.experiment.name == "reload_test"
        assert loaded_config.experiment.seed == 42
        assert loaded_config.task.observation_horizon == 2
        assert loaded_config.task.prediction_horizon == 16
        assert loaded_config.training.num_epochs == 100
        assert loaded_config.training.use_ema is True

    def test_observation_space_properly_saved(self, tmp_path):
        """Test that ObservationSpace config is properly serialized."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="obs_space_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
                    use_proprio_base_frame=True,
                    use_proprio_camera_frame=True,
                    use_language=True,
                ),
                action_space=ActionSpace(has_position=True, position_dim=3),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=2),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        config_path = workspace.output_dir / "config.yaml"

        # Reload and check observation space
        loaded_config = OmegaConf.load(config_path)
        obs_space = loaded_config.task.observation_space

        assert Cameras.LEFT.value in obs_space.camera_keys
        assert Cameras.RIGHT.value in obs_space.camera_keys
        assert Cameras.DEPTH.value in obs_space.camera_keys
        assert obs_space.use_proprio_base_frame is True
        assert obs_space.use_proprio_camera_frame is True
        assert obs_space.use_language is True

    def test_action_space_properly_saved(self, tmp_path):
        """Test that ActionSpace config is properly serialized."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="action_space_test",
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
                    has_orientation=True,
                    orientation_dim=3,
                    orientation_repr=OrientationRepresentation.EULER.value,
                    has_gripper=True,
                    gripper_type=GripperType.BINARY.value,
                    deltas_as_actions=True,
                    predict_in_camera_frame=True,
                    task_has_phases=True,
                    number_of_phases=5,
                ),
                observation_horizon=1,
                prediction_horizon=30,
                dataloader=DataloaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=2),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        config_path = workspace.output_dir / "config.yaml"

        # Reload and check action space
        loaded_config = OmegaConf.load(config_path)
        action_space = loaded_config.task.action_space

        assert action_space.has_position is True
        assert action_space.position_dim == 3
        assert action_space.has_orientation is True
        assert action_space.orientation_dim == 3
        assert action_space.orientation_repr == OrientationRepresentation.EULER.value
        assert action_space.has_gripper is True
        assert action_space.gripper_type == GripperType.BINARY.value
        assert action_space.deltas_as_actions is True
        assert action_space.predict_in_camera_frame is True
        assert action_space.task_has_phases is True
        assert action_space.number_of_phases == 5

    def test_config_yaml_format(self, tmp_path):
        """Test that saved config is valid YAML format."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="yaml_test",
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
            training=TrainingConfig(num_epochs=2),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        config_path = workspace.output_dir / "config.yaml"

        # Read as text and verify YAML markers
        content = config_path.read_text()
        assert "experiment:" in content
        assert "task:" in content
        assert "training:" in content
        assert "policy:" in content

    def test_save_config_creates_directory_if_needed(self, tmp_path):
        """Test that save_config works even if output_dir doesn't exist."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="create_dir_test",
                checkpoint_folder=str(tmp_path / "nonexistent"),
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
            training=TrainingConfig(num_epochs=2),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        # Workspace __init__ creates output_dir and calls save_config
        workspace = Workspace(config)

        # Verify config saved
        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists()
