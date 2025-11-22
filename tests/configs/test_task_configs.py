"""Tests for task configuration dataclasses."""
import dataclasses

import pytest

from refactoring.configs.data.task import TaskSpaceConfig
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.configs.data.tokenizer import ObservationTokenizationConfig, ActionTokenizationConfig, TokenizationConfig
from refactoring.data.constants import GripperType, OrientationRepresentation, TokenizerType


@pytest.mark.unit
class TestActionSpace:

    def test_config_can_be_instantiated(self):
        config = ActionSpace()
        assert isinstance(config, ActionSpace)
        assert config.has_position is True
        assert config.position_dim == 3
        assert config.has_gripper is True

    def test_get_total_action_dim(self):
        config = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=3,
            has_gripper=True,
            gripper_dim=1,
        )
        assert config.get_total_action_dim() == 7

    def test_get_total_action_dim_with_phases(self):
        config = ActionSpace(
            has_position=True,
            position_dim=3,
            has_gripper=True,
            gripper_dim=1,
            task_has_phases=True,
            number_of_phases=5,
        )
        assert config.get_total_action_dim() == 9

    def test_get_required_zarr_keys_camera_frame(self):
        config = ActionSpace(predict_in_camera_frame=True)
        keys = config.get_required_zarr_keys()
        assert "proprio_camera_frame" in keys
        assert "gripper_state_obs" in keys

    def test_get_required_zarr_keys_robot_frame(self):
        config = ActionSpace(predict_in_camera_frame=False)
        keys = config.get_required_zarr_keys()
        assert "proprio_robot_frame" in keys


@pytest.mark.unit
class TestObservationSpace:

    def test_config_can_be_instantiated(self):
        config = ObservationSpace()
        assert isinstance(config, ObservationSpace)
        assert config.use_proprioceptive_data is False
        assert config.use_language is False

    def test_get_required_zarr_keys_minimal(self):
        config = ObservationSpace(
            camera_keys=["left"],
            use_proprio_camera_frame=False,
            use_proprio_base_frame=False,
            use_language=False,
        )
        keys = config.get_required_zarr_keys()
        assert "left" in keys

    def test_get_required_zarr_keys_with_proprio(self):
        config = ObservationSpace(
            camera_keys=["left"],
            use_proprio_camera_frame=True,
        )
        keys = config.get_required_zarr_keys()
        assert "proprio_camera_frame" in keys

    def test_get_required_zarr_keys_with_language(self):
        config = ObservationSpace(
            camera_keys=[],
            use_language=True,
        )
        keys = config.get_required_zarr_keys()
        assert "language_instruction" in keys


@pytest.mark.unit
class TestDataloaderConfig:

    def test_config_can_be_instantiated(self):
        config = DataLoaderConfig()
        assert isinstance(config, DataLoaderConfig)
        assert config.batch_size == 64
        assert config.num_workers == 16
        assert config.image_height == 270
        assert config.image_width == 480

    def test_validation_accepts_valid_config(self):
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=8,
            val_ratio=0.2,
            total_ratio=0.8,
        )
        assert config.batch_size == 32
        assert config.val_ratio == 0.2

    def test_default_tokenization_config(self):
        """Test that default tokenization config has tokenization disabled."""
        config = DataLoaderConfig()
        assert config.tokenization is not None
        assert config.tokenization.tokenize_observations is False
        assert config.tokenization.tokenize_actions is False

    def test_tokenization_with_action_tokenization_enabled(self):
        """Test dataloader config with action tokenization enabled."""
        tokenization_config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=True,
            observation_tokenizer=ObservationTokenizationConfig(),
            action_tokenizer=ActionTokenizationConfig(),
        )
        config = DataLoaderConfig(tokenization=tokenization_config)
        assert config.tokenization.tokenize_actions is True
        assert config.tokenization.tokenize_observations is False
        assert config.tokenization.action_tokenizer is not None

    def test_tokenization_with_observation_tokenization_enabled(self):
        """Test dataloader config with observation tokenization enabled."""
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=False,
            observation_tokenizer=ObservationTokenizationConfig(
                bin_continuous_data=True,
                num_bins=256,
            ),
            action_tokenizer=ActionTokenizationConfig(),
        )
        config = DataLoaderConfig(tokenization=tokenization_config)
        assert config.tokenization.tokenize_observations is True
        assert config.tokenization.tokenize_actions is False
        assert config.tokenization.observation_tokenizer is not None
        assert config.tokenization.observation_tokenizer.bin_continuous_data is True

    def test_tokenization_with_both_tokenizers_enabled(self):
        """Test dataloader config with both tokenizers enabled."""
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=True,
            observation_tokenizer=ObservationTokenizationConfig(),
            action_tokenizer=ActionTokenizationConfig(),
        )
        config = DataLoaderConfig(tokenization=tokenization_config)
        assert config.tokenization.tokenize_observations is True
        assert config.tokenization.tokenize_actions is True


@pytest.mark.unit
class TestObservationTokenizationConfig:

    def test_config_can_be_instantiated(self):
        """Test basic instantiation with defaults."""
        config = ObservationTokenizationConfig()
        assert isinstance(config, ObservationTokenizationConfig)
        assert config.tokenizer_model == "google/gemma-2b"
        assert config.bin_continuous_data is True
        assert config.num_bins == 256
        assert config.max_token_len == 256

    def test_custom_tokenizer_model(self):
        """Test with custom tokenizer model."""
        config = ObservationTokenizationConfig(
            tokenizer_model="bert-base-uncased",
            observation_keys=["language", "proprio_robot_frame"],
        )
        assert config.tokenizer_model == "bert-base-uncased"
        assert "language" in config.observation_keys
        assert "proprio_robot_frame" in config.observation_keys

    def test_binning_disabled(self):
        """Test with continuous data binning disabled."""
        config = ObservationTokenizationConfig(bin_continuous_data=False)
        assert config.bin_continuous_data is False


@pytest.mark.unit
class TestActionTokenizationConfig:

    def test_config_can_be_instantiated(self):
        """Test basic instantiation with defaults."""
        config = ActionTokenizationConfig()
        assert isinstance(config, ActionTokenizationConfig)
        assert config.tokenizer_chain == [TokenizerType.FAST.value]
        assert config.use_pretrained_fast is True

    def test_fast_tokenizer_only(self):
        """Test with FAST tokenizer only."""
        config = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
        )
        assert config.tokenizer_chain == [TokenizerType.FAST.value]
        assert TokenizerType.LANGUAGE.value not in config.tokenizer_chain

    def test_chained_tokenizers(self):
        """Test with chained tokenizers (FAST -> language)."""
        config = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="google/gemma-2b",
        )
        assert config.tokenizer_chain == [TokenizerType.FAST.value, TokenizerType.LANGUAGE.value]
        assert config.language_tokenizer_model == "google/gemma-2b"


@pytest.mark.unit
class TestTokenizationConfig:

    def test_config_can_be_instantiated(self):
        """Test basic instantiation with defaults."""
        config = TokenizationConfig()
        assert isinstance(config, TokenizationConfig)
        assert config.tokenize_observations is False
        assert config.tokenize_actions is False

    def test_both_tokenizers_can_be_disabled(self):
        """Test that both tokenizers can be disabled simultaneously."""
        config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=False,
            observation_tokenizer=None,
            action_tokenizer=None,
        )
        assert config.tokenize_observations is False
        assert config.tokenize_actions is False
