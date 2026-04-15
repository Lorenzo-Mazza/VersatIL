"""Tests for versatil.configs.data.dataloader module."""

import pytest

from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.data.tokenizer import TokenizationConfig
from versatil.data.constants import ImageNormalizationType, KinematicsNormalizationType


@pytest.mark.unit
class TestDataLoaderConfig:
    @pytest.mark.parametrize("batch_size", [32, 128])
    @pytest.mark.parametrize("num_workers", [4, 16])
    @pytest.mark.parametrize("shuffle", [True, False])
    def test_stores_batching_configuration(self, batch_size, num_workers, shuffle):
        config = DataLoaderConfig(
            batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )
        assert config.batch_size == batch_size
        assert config.num_workers == num_workers
        assert config.shuffle == shuffle

    def test_image_norm_type_default_is_minus_one_to_one_string(self):
        config = DataLoaderConfig()
        assert config.image_norm_type == ImageNormalizationType.MINUS_ONE_TO_ONE.value
        assert config.image_norm_type == "minus_one_to_one"

    def test_depth_norm_type_default_is_minus_one_to_one_string(self):
        config = DataLoaderConfig()
        assert config.depth_norm_type == ImageNormalizationType.MINUS_ONE_TO_ONE.value

    def test_kinematics_norm_type_default_is_min_max_string(self):
        config = DataLoaderConfig()
        assert config.kinematics_norm_type == KinematicsNormalizationType.MIN_MAX.value
        assert config.kinematics_norm_type == "min_max"
        # Verify it is a plain string, not an enum member
        assert type(config.kinematics_norm_type) is str

    @pytest.mark.parametrize("winsorize_depth", [True, False])
    @pytest.mark.parametrize("winsorize_kinematics", [True, False])
    def test_stores_winsorize_flags(self, winsorize_depth, winsorize_kinematics):
        config = DataLoaderConfig(
            winsorize_depth=winsorize_depth,
            winsorize_kinematics=winsorize_kinematics,
        )
        assert config.winsorize_depth == winsorize_depth
        assert config.winsorize_kinematics == winsorize_kinematics

    @pytest.mark.parametrize("clamp_kinematics_range", [True, False])
    def test_stores_clamping_configuration(self, clamp_kinematics_range):
        config = DataLoaderConfig(clamp_kinematics_range=clamp_kinematics_range)
        assert config.clamp_kinematics_range == clamp_kinematics_range

    def test_tokenization_default_is_tokenization_config(self):
        config = DataLoaderConfig()
        assert isinstance(config.tokenization, TokenizationConfig)

    def test_color_augmentation_default_is_pipeline_config(self):
        config = DataLoaderConfig()
        assert isinstance(config.color_augmentation, AugmentationPipelineConfig)

    def test_spatial_augmentation_default_is_pipeline_config(self):
        config = DataLoaderConfig()
        assert isinstance(config.spatial_augmentation, AugmentationPipelineConfig)

    def test_rotation_augmentation_default_is_none(self):
        config = DataLoaderConfig()
        assert config.rotation_augmentation is None

    @pytest.mark.parametrize("val_ratio", [0.1, 0.2])
    @pytest.mark.parametrize("total_ratio", [0.5, 1.0])
    def test_stores_dataset_ratios(self, val_ratio, total_ratio):
        config = DataLoaderConfig(val_ratio=val_ratio, total_ratio=total_ratio)
        assert config.val_ratio == val_ratio
        assert config.total_ratio == total_ratio

    @pytest.mark.parametrize("skip_initial_episode_steps", [0, 5])
    @pytest.mark.parametrize("downsample_factor", [1, 3])
    @pytest.mark.parametrize("action_backward_shift", [0, 2])
    def test_stores_episode_processing_options(
        self, skip_initial_episode_steps, downsample_factor, action_backward_shift
    ):
        config = DataLoaderConfig(
            skip_initial_episode_steps=skip_initial_episode_steps,
            downsample_factor=downsample_factor,
            action_backward_shift=action_backward_shift,
        )
        assert config.skip_initial_episode_steps == skip_initial_episode_steps
        assert config.downsample_factor == downsample_factor
        assert config.action_backward_shift == action_backward_shift

    @pytest.mark.parametrize("preload_data_in_memory", [True, False])
    def test_stores_preload_flag(self, preload_data_in_memory):
        config = DataLoaderConfig(preload_data_in_memory=preload_data_in_memory)
        assert config.preload_data_in_memory == preload_data_in_memory
