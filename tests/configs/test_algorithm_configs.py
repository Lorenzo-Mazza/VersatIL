"""Tests for algorithm configuration dataclasses."""
import dataclasses
import inspect

import pytest
from hydra.utils import instantiate

from refactoring.configs.decoding.algorithm import (
    BehavioralCloningConfig,
    DiffusionConfig,
    FlowMatchingConfig,
    VariationalAlgorithmConfig,
)
from refactoring.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from refactoring.models.decoding.algorithm.action_diffusion import Diffusion
from refactoring.models.decoding.algorithm.flow_matching import FlowMatching
from refactoring.models.decoding.algorithm.variational import VariationalAlgorithm


@pytest.mark.unit
class TestBehavioralCloningConfig:

    def test_config_has_correct_target(self):
        config = BehavioralCloningConfig()
        assert config._target_ == "refactoring.models.decoding.algorithm.behavior_cloning.BehavioralCloning"

    def test_config_instantiates_correctly(self):
        config = BehavioralCloningConfig()
        algorithm = instantiate(config)
        assert isinstance(algorithm, BehavioralCloning)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(BehavioralCloning.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = BehavioralCloningConfig()
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestDiffusionConfig:

    def test_config_has_correct_target(self):
        config = DiffusionConfig()
        assert config._target_ == "refactoring.models.decoding.algorithm.action_diffusion.Diffusion"

    def test_config_instantiates_correctly(self):
        config = DiffusionConfig()
        algorithm = instantiate(config)
        assert isinstance(algorithm, Diffusion)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(Diffusion.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = DiffusionConfig()
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestFlowMatchingConfig:

    def test_config_has_correct_target(self):
        config = FlowMatchingConfig()
        assert config._target_ == "refactoring.models.decoding.algorithm.flow_matching.FlowMatching"

    def test_config_instantiates_correctly(self):
        config = FlowMatchingConfig()
        algorithm = instantiate(config)
        assert isinstance(algorithm, FlowMatching)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(FlowMatching.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = FlowMatchingConfig()
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestVariationalAlgorithmConfig:

    def test_config_has_correct_target(self):
        from refactoring.configs.decoding.latent import VAETransformerEncoderConfig

        config = VariationalAlgorithmConfig(
            base_algorithm=BehavioralCloningConfig(),
            posterior_encoder=VAETransformerEncoderConfig(latent_dim=32),
        )
        assert config._target_ == "refactoring.models.decoding.algorithm.variational.VariationalAlgorithm"

    def test_config_params_match_class_signature(self):
        from refactoring.configs.decoding.latent import VAETransformerEncoderConfig

        sig = inspect.signature(VariationalAlgorithm.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = VariationalAlgorithmConfig(
            base_algorithm=BehavioralCloningConfig(),
            posterior_encoder=VAETransformerEncoderConfig(latent_dim=32),
        )
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"