"""Tests for versatil.configs.decoding.action_head module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.decoding.action_head import (
    ActionHeadBlockConfig,
    ActionHeadConfig,
    AttentionBlockConfig,
    GaussianHeadConfig,
    MixtureOfExpertsHeadConfig,
    MLPBlockConfig,
    ResidualBlockConfig,
)
from versatil.models.decoding.constants import MoERoutingType


@pytest.mark.unit
class TestActionHeadBlockConfig:
    def test_target_defaults_to_missing(self):
        config = ActionHeadBlockConfig()
        assert config._target_ == MISSING


@pytest.mark.unit
class TestMLPBlockConfig:
    def test_target_points_to_mlp_block(self):
        config = MLPBlockConfig(input_dim=256)
        assert config._target_ == "versatil.models.decoding.action_heads.MLPBlock"

    @pytest.mark.parametrize("input_dim", [128, 512])
    @pytest.mark.parametrize("dropout", [0.0, 0.2])
    @pytest.mark.parametrize("normalization", [True, False])
    def test_stores_configuration(self, input_dim, dropout, normalization):
        config = MLPBlockConfig(
            input_dim=input_dim, dropout=dropout, normalization=normalization
        )
        assert config.input_dim == input_dim
        assert config.dropout == dropout
        assert config.normalization == normalization

    def test_inherits_from_action_head_block_config(self):
        config = MLPBlockConfig(input_dim=256)
        assert isinstance(config, ActionHeadBlockConfig)


@pytest.mark.unit
class TestAttentionBlockConfig:
    def test_target_points_to_attention_block(self):
        config = AttentionBlockConfig(embedding_dimension=256)
        assert config._target_ == "versatil.models.decoding.action_heads.AttentionBlock"

    @pytest.mark.parametrize("embedding_dimension", [128, 512])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_stores_configuration(self, embedding_dimension, num_heads):
        config = AttentionBlockConfig(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )
        assert config.embedding_dimension == embedding_dimension
        assert config.num_heads == num_heads


@pytest.mark.unit
class TestResidualBlockConfig:
    def test_target_points_to_residual_block(self):
        config = ResidualBlockConfig(block=MLPBlockConfig(input_dim=256))
        assert config._target_ == "versatil.models.decoding.action_heads.ResidualBlock"

    def test_block_required(self):
        config = ResidualBlockConfig()
        assert config.block == MISSING


@pytest.mark.unit
class TestActionHeadConfig:
    def test_target_points_to_action_head(self):
        config = ActionHeadConfig(input_dim=256)
        assert config._target_ == "versatil.models.decoding.action_heads.ActionHead"

    def test_input_dim_required(self):
        config = ActionHeadConfig()
        assert config.input_dim == MISSING

    def test_blocks_default_to_none(self):
        config = ActionHeadConfig(input_dim=256)
        assert config.blocks is None


@pytest.mark.unit
class TestGaussianHeadConfig:
    def test_target_points_to_gaussian_head(self):
        config = GaussianHeadConfig(input_dim=256)
        assert config._target_ == "versatil.models.decoding.action_heads.GaussianHead"

    @pytest.mark.parametrize("min_logvar", [-10.0, -5.0])
    @pytest.mark.parametrize("max_logvar", [4.0, 2.0])
    def test_stores_logvar_bounds(self, min_logvar, max_logvar):
        config = GaussianHeadConfig(
            input_dim=256, min_logvar=min_logvar, max_logvar=max_logvar
        )
        assert config.min_logvar == min_logvar
        assert config.max_logvar == max_logvar


@pytest.mark.unit
class TestMixtureOfExpertsHeadConfig:
    def test_target_points_to_moe_head(self):
        config = MixtureOfExpertsHeadConfig(num_experts=3)
        assert config._target_ == "versatil.models.decoding.action_heads.MoEHead"

    def test_num_experts_required(self):
        config = MixtureOfExpertsHeadConfig()
        assert config.num_experts == MISSING

    def test_routing_type_default_is_soft_string(self):
        config = MixtureOfExpertsHeadConfig(num_experts=3)
        assert config.routing_type == MoERoutingType.SOFT.value

    @pytest.mark.parametrize("num_experts", [2, 8])
    @pytest.mark.parametrize("top_k", [1, 3])
    @pytest.mark.parametrize("learnable_temperature", [True, False])
    def test_stores_configuration(self, num_experts, top_k, learnable_temperature):
        config = MixtureOfExpertsHeadConfig(
            num_experts=num_experts,
            top_k=top_k,
            learnable_temperature=learnable_temperature,
        )
        assert config.num_experts == num_experts
        assert config.top_k == top_k
        assert config.learnable_temperature == learnable_temperature

    def test_base_expert_and_experts_both_optional(self):
        config = MixtureOfExpertsHeadConfig(num_experts=3)
        assert config.base_expert is None
        assert config.experts is None


@pytest.mark.unit
class TestActionHeadInstantiation:
    def test_action_head_instantiates(self):
        config = ActionHeadConfig(input_dim=64)
        instance = instantiate(config)
        assert instance.input_dim == 64

    def test_gaussian_head_instantiates(self):
        config = GaussianHeadConfig(input_dim=64, min_logvar=-5.0, max_logvar=2.0)
        instance = instantiate(config)
        assert instance.min_logvar == -5.0
        assert instance.max_logvar == 2.0

    def test_mlp_block_instantiates(self):
        config = MLPBlockConfig(input_dim=64, hidden_dims=[128, 64])
        instance = instantiate(config)
        assert instance.input_dim == 64
