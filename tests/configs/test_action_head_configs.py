"""Tests for action head configuration dataclasses.

Tests ensure that:
1. Config dataclasses can be instantiated
2. Configs instantiate to correct target classes
3. Required fields are validated
4. No parameter name conflicts with target class
"""
import inspect

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from refactoring.configs.decoding.action_head import (
    ActionHeadConfig,
    MLPBlockConfig,
    AttentionBlockConfig,
    MixtureOfExpertsHeadConfig,
)
from refactoring.models.decoding.action_heads import (
    ActionHead,
    MLPBlock,
    AttentionBlock,
    MoEHead,
)


@pytest.mark.unit
class TestActionHeadConfig:

    def test_config_has_correct_target(self):
        config = ActionHeadConfig(input_dim=256, output_dim=7)
        assert config._target_ == "refactoring.models.decoding.action_heads.ActionHead"

    def test_config_instantiates_correctly(self):
        config = ActionHeadConfig(input_dim=256, output_dim=7, blocks=[])
        head = instantiate(config)
        assert isinstance(head, ActionHead)
        assert head.input_dim == 256
        assert head.output_dim == 7

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(ActionHead.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = ActionHeadConfig(input_dim=256, output_dim=7)
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestMLPBlockConfig:

    def test_config_has_correct_target(self):
        config = MLPBlockConfig(input_dim=256)
        assert config._target_ == "refactoring.models.decoding.action_heads.MLPBlock"

    def test_config_instantiates_correctly(self):
        config = MLPBlockConfig(input_dim=256, hidden_dims=[128, 64])
        block = instantiate(config)
        assert isinstance(block, MLPBlock)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(MLPBlock.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = MLPBlockConfig(input_dim=256)
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestAttentionBlockConfig:

    def test_config_has_correct_target(self):
        config = AttentionBlockConfig(embedding_dimension=256)
        assert config._target_ == "refactoring.models.decoding.action_heads.AttentionBlock"

    def test_config_instantiates_correctly(self):
        config = AttentionBlockConfig(embedding_dimension=256, num_heads=8)
        block = instantiate(config)
        assert isinstance(block, AttentionBlock)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(AttentionBlock.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = AttentionBlockConfig(embedding_dimension=256)
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestMixtureOfExpertsHeadConfig:

    def test_config_has_correct_target(self):
        config = MixtureOfExpertsHeadConfig(
            output_dim=7,
            base_expert=ActionHeadConfig(input_dim=256, output_dim=7, blocks=[]),
            num_experts=3,
        )
        assert config._target_ == "refactoring.models.decoding.action_heads.MoEHead"

    def test_config_instantiates_with_base_expert(self):
        config = MixtureOfExpertsHeadConfig(
            output_dim=7,
            base_expert=ActionHeadConfig(input_dim=256, output_dim=7, blocks=[]),
            num_experts=3,
            gating_input_dim=256,
            device="cpu",
        )
        moe = instantiate(config)
        assert isinstance(moe, MoEHead)
        assert moe.num_experts == 3
        assert len(moe.experts) == 3

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(MoEHead.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = MixtureOfExpertsHeadConfig(
            output_dim=7,
            base_expert=ActionHeadConfig(input_dim=256, output_dim=7, blocks=[]),
            num_experts=3,
        )
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_no_legacy_base_expert_config_field(self):
        config = MixtureOfExpertsHeadConfig(
            output_dim=7,
            base_expert=ActionHeadConfig(input_dim=256, output_dim=7, blocks=[]),
            num_experts=3,
        )
        config_dict = OmegaConf.structured(config)

        assert 'base_expert_config' not in config_dict
        assert 'base_expert' in config_dict