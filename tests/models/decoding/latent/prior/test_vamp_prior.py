"""Tests for versatil.models.decoding.latent.prior.vamp_prior module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.task import ActionSpace
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.vamp_prior import VampPrior, log_normal_diag


@pytest.fixture
def mock_action_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ActionSpace with configurable total action dim."""
    def factory(total_action_dim: int = 7) -> MagicMock:
        action_space = MagicMock(spec=ActionSpace)
        action_space.get_total_action_dim.return_value = total_action_dim
        return action_space
    return factory


@pytest.fixture
def vamp_prior_factory(
    mock_action_space_factory: Callable[..., MagicMock],
) -> Callable[..., VampPrior]:
    """Factory for VampPrior instances with mocked ActionSpace."""
    def factory(
        latent_dimension: int = 16,
        num_components: int = 5,
        action_dim: int = 7,
        prediction_horizon: int = 8,
        device: str = "cpu",
        min_logvar: float | None = None,
    ) -> VampPrior:
        action_space = mock_action_space_factory(total_action_dim=action_dim)
        return VampPrior(
            latent_dimension=latent_dimension,
            num_components=num_components,
            action_space=action_space,
            prediction_horizon=prediction_horizon,
            device=device,
            min_logvar=min_logvar,
        )
    return factory


@pytest.fixture
def mock_encoder_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock PosteriorLatentEncoder returning mu and logvar."""
    def factory(
        latent_dimension: int = 16,
        num_components: int = 5,
    ) -> MagicMock:
        encoder = MagicMock(spec=PosteriorLatentEncoder)
        encoder.latent_dimension = latent_dimension
        mu = torch.from_numpy(
            rng.standard_normal((num_components, latent_dimension)).astype(np.float32)
        )
        logvar = torch.from_numpy(
            np.zeros((num_components, latent_dimension), dtype=np.float32)
        )
        latent = torch.from_numpy(
            rng.standard_normal((num_components, latent_dimension)).astype(np.float32)
        )
        encoder.encode.return_value = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: latent,
        }
        return encoder
    return factory


@pytest.fixture
def latent_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for latent tensors of shape (batch_size, latent_dimension)."""
    def factory(
        batch_size: int = 4,
        latent_dimension: int = 16,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
    return factory


class TestLogNormalDiag:

    def test_output_shape(self, rng: np.random.Generator):
        batch_size = 4
        latent_dimension = 8
        z = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        mu = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        logvar = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        result = log_normal_diag(z=z, mu=mu, logvar=logvar)
        assert result.shape == (batch_size, latent_dimension)

    def test_max_at_mean(self, rng: np.random.Generator):
        latent_dimension = 16
        mu = torch.from_numpy(
            rng.standard_normal((1, latent_dimension)).astype(np.float32)
        )
        logvar = torch.zeros(1, latent_dimension)
        z_at_mean = mu.clone()
        z_away = mu + 2.0
        log_prob_at_mean = log_normal_diag(z=z_at_mean, mu=mu, logvar=logvar).sum(dim=-1)
        log_prob_away = log_normal_diag(z=z_away, mu=mu, logvar=logvar).sum(dim=-1)
        assert log_prob_at_mean.item() > log_prob_away.item()

    def test_decreases_away_from_mean(self, rng: np.random.Generator):
        latent_dimension = 8
        mu = torch.from_numpy(
            rng.standard_normal((1, latent_dimension)).astype(np.float32)
        )
        logvar = torch.zeros(1, latent_dimension)
        z_near = mu + 0.5
        z_far = mu + 3.0
        log_prob_near = log_normal_diag(z=z_near, mu=mu, logvar=logvar).sum(dim=-1)
        log_prob_far = log_normal_diag(z=z_far, mu=mu, logvar=logvar).sum(dim=-1)
        assert log_prob_near.item() > log_prob_far.item()


class TestVampPriorInitialization:

    def test_inherits_from_prior_latent_encoder(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        prior = vamp_prior_factory()
        assert isinstance(prior, PriorLatentEncoder)

    @pytest.mark.parametrize("latent_dimension", [8, 32])
    @pytest.mark.parametrize("num_components", [3, 10])
    def test_stores_configuration(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        latent_dimension: int,
        num_components: int,
    ):
        action_dim = 7
        prediction_horizon = 8
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
        )
        assert prior.latent_dimension == latent_dimension
        assert prior.num_components == num_components
        assert prior.action_dim == action_dim
        assert prior.prediction_horizon == prediction_horizon

    @pytest.mark.parametrize("num_components, prediction_horizon, action_dim", [
        (3, 4, 7),
        (10, 16, 14),
    ])
    def test_pseudo_inputs_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        num_components: int,
        prediction_horizon: int,
        action_dim: int,
    ):
        prior = vamp_prior_factory(
            num_components=num_components,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
        )
        assert prior.pseudo_inputs.shape == (num_components, prediction_horizon, action_dim)
        assert prior.pseudo_inputs.requires_grad is True

    @pytest.mark.parametrize("num_components", [3, 10])
    def test_log_weights_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        num_components: int,
    ):
        prior = vamp_prior_factory(num_components=num_components)
        assert prior.log_weights.shape == (num_components, 1, 1)
        assert prior.log_weights.requires_grad is True


class TestVampPriorEncoder:

    def test_encoder_raises_when_not_set(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        prior = vamp_prior_factory()
        with pytest.raises(
            RuntimeError,
            match=(
                "VampPrior encoder not set. Call set_encoder\\(\\) first or ensure "
                "VariationalAlgorithm properly initializes the prior."
            ),
        ):
            _ = prior.encoder

    def test_set_encoder_stores_encoder(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.set_encoder(encoder=encoder)
        assert prior._encoder is encoder

    def test_encoder_returns_stored_encoder(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.set_encoder(encoder=encoder)
        assert prior.encoder is encoder


class TestVampPriorGetMixtureParams:

    def test_returns_mu_and_logvar(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        latent_dimension = 16
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        prior.set_encoder(encoder=encoder)
        mu, logvar = prior.get_mixture_params()
        assert mu.shape == (num_components, latent_dimension)
        assert logvar.shape == (num_components, latent_dimension)

    def test_calls_encoder_encode(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.set_encoder(encoder=encoder)
        prior.get_mixture_params()
        encoder.encode.assert_called_once()
        call_kwargs = encoder.encode.call_args
        assert call_kwargs.kwargs["observations"] is None

    def test_clamps_logvar_when_min_logvar_set(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        latent_dimension = 16
        num_components = 5
        min_logvar = -2.0
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
            min_logvar=min_logvar,
        )
        encoder = mock_encoder_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        # Set logvar to values below the clamp threshold
        encoder.encode.return_value[LatentKey.POSTERIOR_LOGVAR.value] = (
            torch.full((num_components, latent_dimension), -10.0)
        )
        prior.set_encoder(encoder=encoder)
        _, logvar = prior.get_mixture_params()
        assert torch.all(logvar >= min_logvar)


class TestVampPriorSamplePrior:

    @pytest.mark.parametrize("batch_size", [2, 8])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        batch_size: int,
        latent_dimension: int,
    ):
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        prior.set_encoder(encoder=encoder)
        sample = prior.sample_prior(batch_size=batch_size)
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (batch_size, latent_dimension)

    def test_calls_get_mixture_params(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.set_encoder(encoder=encoder)
        prior.sample_prior(batch_size=4)
        # get_mixture_params calls encoder.encode, so verify it was called
        encoder.encode.assert_called_once()


class TestVampPriorForward:

    def test_returns_expected_keys(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        latent_dimension = 16
        prior = vamp_prior_factory(latent_dimension=latent_dimension)
        encoder = mock_encoder_factory(latent_dimension=latent_dimension)
        prior.set_encoder(encoder=encoder)
        target_latents = latent_tensor_factory(
            batch_size=4,
            latent_dimension=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=4)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_LOG_PROB.value,
        }
        assert isinstance(result[LatentKey.PRIOR_LATENT.value], torch.Tensor)
        assert isinstance(result[LatentKey.PRIOR_LOG_PROB.value], torch.Tensor)

    @pytest.mark.parametrize("batch_size", [2, 6])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shapes(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        batch_size: int,
        latent_dimension: int,
    ):
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        prior.set_encoder(encoder=encoder)
        target_latents = latent_tensor_factory(
            batch_size=batch_size,
            latent_dimension=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=batch_size)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert result[LatentKey.PRIOR_LATENT.value].shape == (batch_size, latent_dimension)
        assert result[LatentKey.PRIOR_LOG_PROB.value].shape == (batch_size,)


class TestVampPriorLogProb:

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        latent_dimension: int,
    ):
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        prior.set_encoder(encoder=encoder)
        z = latent_tensor_factory(
            batch_size=batch_size,
            latent_dimension=latent_dimension,
        )
        log_prob = prior.log_prob(z=z)
        assert log_prob.shape == (batch_size,)

    def test_returns_finite_values(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
    ):
        latent_dimension = 16
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        prior.set_encoder(encoder=encoder)
        z = latent_tensor_factory(
            batch_size=4,
            latent_dimension=latent_dimension,
        )
        log_prob = prior.log_prob(z=z)
        assert torch.all(torch.isfinite(log_prob))
