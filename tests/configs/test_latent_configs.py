"""Tests for latent encoder and prior configuration dataclasses."""
import dataclasses
import inspect

import pytest
from hydra.utils import instantiate

from refactoring.configs.decoding.latent import (
    VAETransformerEncoderConfig,
    GaussianPriorConfig,
    DiffusionPriorConfig,
)
from refactoring.models.decoding.latent.vae_posterior import VAETransformerEncoder
from refactoring.models.decoding.latent.gaussian_prior import GaussianPrior
from refactoring.models.decoding.latent.diffusion_prior import DiffusionPrior


@pytest.mark.unit
class TestVAETransformerEncoderConfig:

    def test_config_has_correct_target(self):
        config = VAETransformerEncoderConfig(latent_dimension=32)
        assert config._target_ == "refactoring.models.decoding.latent.vae_posterior.VAETransformerEncoder"

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(VAETransformerEncoder.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = VAETransformerEncoderConfig(latent_dimension=32)
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestGaussianPriorConfig:

    def test_config_has_correct_target(self):
        config = GaussianPriorConfig(latent_dim=32)
        assert config._target_ == "refactoring.models.decoding.latent.gaussian_prior.GaussianPrior"

    def test_config_instantiates_correctly(self):
        config = GaussianPriorConfig(latent_dim=32, device="cpu")
        prior = instantiate(config)
        assert isinstance(prior, GaussianPrior)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(GaussianPrior.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = GaussianPriorConfig(latent_dim=32, )
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestDiffusionPriorConfig:

    def test_config_has_correct_target(self):
        config = DiffusionPriorConfig(latent_dim=32, conditioning_dim=128,)
        assert config._target_ == "refactoring.models.decoding.latent.diffusion_prior.DiffusionPrior"

    def test_config_instantiates_correctly(self):
        config = DiffusionPriorConfig(latent_dim=32, conditioning_dim=128, device="cpu")
        prior = instantiate(config)
        assert isinstance(prior, DiffusionPrior)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(DiffusionPrior.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = DiffusionPriorConfig(latent_dim=32, conditioning_dim=128,)
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"
