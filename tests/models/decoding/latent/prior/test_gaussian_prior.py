"""Tests for versatil.models.decoding.latent.prior.gaussian_prior module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.gaussian_prior import GaussianPrior


@pytest.fixture
def gaussian_prior_factory() -> Callable[..., GaussianPrior]:
    """Factory for GaussianPrior instances."""
    def factory(
        latent_dimension: int = 32,
        device: str = "cpu",
        infer_constant_prior: bool = False,
    ) -> GaussianPrior:
        return GaussianPrior(
            latent_dimension=latent_dimension,
            device=device,
            infer_constant_prior=infer_constant_prior,
        )
    return factory


class TestGaussianPriorInitialization:

    def test_inherits_from_prior_latent_encoder(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
    ):
        prior = gaussian_prior_factory()
        assert isinstance(prior, PriorLatentEncoder)

    @pytest.mark.parametrize("latent_dimension", [16, 64])
    @pytest.mark.parametrize("infer_constant_prior", [True, False])
    def test_stores_configuration(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        latent_dimension: int,
        infer_constant_prior: bool,
    ):
        prior = gaussian_prior_factory(
            latent_dimension=latent_dimension,
            infer_constant_prior=infer_constant_prior,
        )
        assert prior.latent_dimension == latent_dimension
        assert prior.infer_constant_prior is infer_constant_prior


class TestGaussianPriorForward:

    def test_returns_prior_keys(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        input_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        latent_dimension = 32
        prior = gaussian_prior_factory(latent_dimension=latent_dimension)
        target_latents = input_tensor_factory(
            batch_size=2, input_dim=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=2)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.PRIOR_MU.value,
            LatentKey.PRIOR_LOGVAR.value,
            LatentKey.PRIOR_LATENT.value,
        }
        assert isinstance(result[LatentKey.PRIOR_MU.value], torch.Tensor)
        assert isinstance(result[LatentKey.PRIOR_LOGVAR.value], torch.Tensor)
        assert isinstance(result[LatentKey.PRIOR_LATENT.value], torch.Tensor)

    def test_mu_is_zero(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        input_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        latent_dimension = 32
        prior = gaussian_prior_factory(latent_dimension=latent_dimension)
        target_latents = input_tensor_factory(
            batch_size=2, input_dim=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=2)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert torch.all(result[LatentKey.PRIOR_MU.value] == 0.0)

    def test_logvar_is_zero(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        input_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        latent_dimension = 32
        prior = gaussian_prior_factory(latent_dimension=latent_dimension)
        target_latents = input_tensor_factory(
            batch_size=2, input_dim=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=2)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert torch.all(result[LatentKey.PRIOR_LOGVAR.value] == 0.0)

    def test_output_shapes(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        input_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        latent_dimension = 16
        prior = gaussian_prior_factory(latent_dimension=latent_dimension)
        target_latents = input_tensor_factory(
            batch_size=batch_size, input_dim=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=batch_size)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert result[LatentKey.PRIOR_MU.value].shape == (batch_size, latent_dimension)
        assert result[LatentKey.PRIOR_LOGVAR.value].shape == (batch_size, latent_dimension)
        assert result[LatentKey.PRIOR_LATENT.value].shape == (batch_size, latent_dimension)


class TestGaussianPriorSamplePrior:

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dimension", [16, 64])
    def test_sample_shape(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        batch_size: int,
        latent_dimension: int,
    ):
        prior = gaussian_prior_factory(latent_dimension=latent_dimension)
        sample = prior.sample_prior(batch_size=batch_size)
        assert sample.shape == (batch_size, latent_dimension)

    def test_standard_normal_sampling(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
    ):
        prior = gaussian_prior_factory(
            latent_dimension=32,
            infer_constant_prior=False,
        )
        sample = prior.sample_prior(batch_size=4)
        assert not torch.all(sample == 0.0)

    def test_constant_prior_returns_zeros(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
    ):
        prior = gaussian_prior_factory(
            latent_dimension=32,
            infer_constant_prior=True,
        )
        sample = prior.sample_prior(batch_size=4)
        assert torch.all(sample == 0.0)

    def test_observations_ignored(
        self,
        gaussian_prior_factory: Callable[..., GaussianPrior],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = gaussian_prior_factory(latent_dimension=32, infer_constant_prior=True)
        observations = feature_dictionary_factory(batch_size=2)
        sample_with_obs = prior.sample_prior(batch_size=2, observations=observations)
        sample_without_obs = prior.sample_prior(batch_size=2)
        assert torch.all(sample_with_obs == 0.0)
        assert torch.all(sample_without_obs == 0.0)