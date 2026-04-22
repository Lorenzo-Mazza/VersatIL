"""Tests for versatil.configs.data.dataloader module."""

import pytest

from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.data.tokenizer import TokenizationConfig
from versatil.data.constants import ImageNormalizationType, KinematicsNormalizationType


@pytest.mark.unit
class TestDataLoaderConfig:
    @pytest.mark.parametrize("preload_data_in_memory", [True])
    @pytest.mark.parametrize("batch_size", [32])
    @pytest.mark.parametrize("num_workers", [4])
    @pytest.mark.parametrize("shuffle", [True])
    @pytest.mark.parametrize(
        "image_norm_type",
        [
            ImageNormalizationType.MINUS_ONE_TO_ONE.value,
            ImageNormalizationType.ZERO_TO_ONE.value,
        ],
    )
    @pytest.mark.parametrize(
        "depth_norm_type",
        [
            ImageNormalizationType.MINUS_ONE_TO_ONE.value,
            ImageNormalizationType.ZERO_TO_ONE.value,
        ],
    )
    @pytest.mark.parametrize(
        "kinematics_norm_type",
        [
            KinematicsNormalizationType.MIN_MAX.value,
            KinematicsNormalizationType.GAUSSIAN.value,
        ],
    )
    @pytest.mark.parametrize("winsorize_depth", [True])
    @pytest.mark.parametrize("winsorize_kinematics", [True])
    @pytest.mark.parametrize("clamp_kinematics_range", [True])
    @pytest.mark.parametrize("skip_initial_episode_steps", [0])
    @pytest.mark.parametrize(
        "downsample_factor",
        [
            1,
        ],
    )
    @pytest.mark.parametrize("action_backward_shift", [2])
    @pytest.mark.parametrize("trailing_padded_actions", [None, 5])
    @pytest.mark.parametrize("val_ratio", [0.2])
    @pytest.mark.parametrize("total_ratio", [0.5])
    @pytest.mark.parametrize("action_sample_size", [0, 100, 2048])
    def test_stores_configuration(
        self,
        preload_data_in_memory: bool,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        image_norm_type: str,
        depth_norm_type: str,
        kinematics_norm_type: str,
        winsorize_depth: bool,
        winsorize_kinematics: bool,
        clamp_kinematics_range: bool,
        skip_initial_episode_steps: int,
        downsample_factor: int,
        action_backward_shift: int,
        trailing_padded_actions: int | None,
        val_ratio: float,
        total_ratio: float,
        action_sample_size: int,
    ):
        config = DataLoaderConfig(
            preload_data_in_memory=preload_data_in_memory,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            image_norm_type=image_norm_type,
            depth_norm_type=depth_norm_type,
            kinematics_norm_type=kinematics_norm_type,
            winsorize_depth=winsorize_depth,
            winsorize_kinematics=winsorize_kinematics,
            clamp_kinematics_range=clamp_kinematics_range,
            skip_initial_episode_steps=skip_initial_episode_steps,
            downsample_factor=downsample_factor,
            action_backward_shift=action_backward_shift,
            trailing_padded_actions=trailing_padded_actions,
            val_ratio=val_ratio,
            total_ratio=total_ratio,
            action_sample_size=action_sample_size,
        )
        assert config.preload_data_in_memory == preload_data_in_memory
        assert config.batch_size == batch_size
        assert config.num_workers == num_workers
        assert config.shuffle == shuffle
        assert config.image_norm_type == image_norm_type
        assert config.depth_norm_type == depth_norm_type
        assert config.kinematics_norm_type == kinematics_norm_type
        assert config.winsorize_depth == winsorize_depth
        assert config.winsorize_kinematics == winsorize_kinematics
        assert config.clamp_kinematics_range == clamp_kinematics_range
        assert config.skip_initial_episode_steps == skip_initial_episode_steps
        assert config.downsample_factor == downsample_factor
        assert config.action_backward_shift == action_backward_shift
        assert config.trailing_padded_actions == trailing_padded_actions
        assert config.val_ratio == val_ratio
        assert config.total_ratio == total_ratio
        assert config.action_sample_size == action_sample_size

    def test_defaults(self):
        config = DataLoaderConfig()
        assert config.preload_data_in_memory is False
        assert config.batch_size == 64
        assert config.num_workers == 16
        assert config.shuffle is True
        assert config.image_norm_type == ImageNormalizationType.MINUS_ONE_TO_ONE.value
        assert config.depth_norm_type == ImageNormalizationType.MINUS_ONE_TO_ONE.value
        assert config.kinematics_norm_type == KinematicsNormalizationType.MIN_MAX.value
        assert config.winsorize_depth is True
        assert config.depth_winsorize_quantiles == (0.01, 0.99)
        assert config.winsorize_kinematics is True
        assert config.kinematics_winsorize_quantiles == (0.01, 0.99)
        assert config.clamp_kinematics_range is True
        assert config.min_kinematics_std == 1e-2
        assert config.min_kinematics_range == 1e-2
        assert isinstance(config.tokenization, TokenizationConfig)
        assert isinstance(config.color_augmentation, AugmentationPipelineConfig)
        assert isinstance(config.spatial_augmentation, AugmentationPipelineConfig)
        assert config.rotation_augmentation is None
        assert config.skip_initial_episode_steps == 0
        assert config.downsample_factor == 1
        assert config.action_backward_shift == 0
        assert config.trailing_padded_actions is None
        assert config.val_ratio == 0.1
        assert config.total_ratio == 1.0
        assert config.action_sample_size == 2048
