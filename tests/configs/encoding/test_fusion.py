"""Tests for versatil.configs.encoding.fusion module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.encoding.fusion import (
    AttentionFusionConfig,
    ConcatFusionConfig,
    FusionConfig,
    MLPFusionConfig,
    SpatialFusionConfig,
)
from versatil.models.encoding.fusion.constants import ConcatDimension
from versatil.models.layers.activation import ActivationFunction


@pytest.mark.unit
class TestFusionConfig:
    def test_all_required_fields_default_to_missing(self):
        config = FusionConfig()
        assert config._target_ == MISSING
        assert config.input_features == MISSING
        assert config.output_name == MISSING
        assert config.hidden_dimension == MISSING


@pytest.mark.unit
class TestConcatFusionConfig:
    def test_target_points_to_concat_fusion(self):
        config = ConcatFusionConfig(
            input_features=["a", "b"], output_name="fused", hidden_dimension=256
        )
        assert config._target_ == "versatil.models.encoding.fusion.concat.ConcatFusion"

    def test_inherits_from_fusion_config(self):
        config = ConcatFusionConfig(
            input_features=["a", "b"], output_name="fused", hidden_dimension=256
        )
        assert isinstance(config, FusionConfig)


@pytest.mark.unit
class TestAttentionFusionConfig:
    def test_target_points_to_attention_fusion(self):
        config = AttentionFusionConfig(
            input_features=["a", "b"], output_name="fused", hidden_dimension=256
        )
        assert (
            config._target_
            == "versatil.models.encoding.fusion.attention.AttentionFusion"
        )

    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_stores_configuration(self, number_of_heads, dropout):
        config = AttentionFusionConfig(
            input_features=["a", "b"],
            output_name="fused",
            hidden_dimension=256,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        assert config.number_of_heads == number_of_heads
        assert config.dropout == dropout


@pytest.mark.unit
class TestMLPFusionConfig:
    def test_target_points_to_mlp_fusion(self):
        config = MLPFusionConfig(
            input_features=["a", "b"],
            output_name="fused",
            hidden_dimension=256,
            mlp_hidden_dims=[128],
        )
        assert config._target_ == "versatil.models.encoding.fusion.mlp.MLPFusion"

    def test_activation_name_default_is_gelu_string(self):
        config = MLPFusionConfig(
            input_features=["a", "b"],
            output_name="fused",
            hidden_dimension=256,
            mlp_hidden_dims=[128],
        )
        assert config.activation_name == ActivationFunction.GELU.value


@pytest.mark.unit
class TestSpatialFusionConfig:
    def test_target_points_to_spatial_fusion(self):
        config = SpatialFusionConfig(
            input_features=["a", "b"], output_name="fused", hidden_dimension=256
        )
        assert (
            config._target_ == "versatil.models.encoding.fusion.spatial.SpatialFusion"
        )

    def test_concat_dim_default_is_width_string(self):
        config = SpatialFusionConfig(
            input_features=["a", "b"], output_name="fused", hidden_dimension=256
        )
        assert config.concat_dim == ConcatDimension.WIDTH.value


@pytest.mark.unit
class TestFusionInstantiation:
    def test_concat_fusion_instantiates(self):
        config = ConcatFusionConfig(
            input_features=["rgb", "depth"],
            output_name="fused",
            hidden_dimension=256,
        )
        instance = instantiate(config)
        assert instance.output_name == "fused"
        assert instance.hidden_dimension == 256

    def test_attention_fusion_instantiates(self):
        config = AttentionFusionConfig(
            input_features=["rgb", "depth"],
            output_name="fused",
            hidden_dimension=256,
            number_of_heads=8,
        )
        instance = instantiate(config)
        assert instance.hidden_dimension == 256

    def test_mlp_fusion_instantiates(self):
        config = MLPFusionConfig(
            input_features=["rgb", "depth"],
            output_name="fused",
            hidden_dimension=256,
            mlp_hidden_dims=[512, 256],
        )
        instance = instantiate(config)
        assert instance.hidden_dimension == 256
