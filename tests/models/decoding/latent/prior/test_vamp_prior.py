"""Tests for versatil.models.decoding.latent.prior.vamp_prior module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.vamp_prior import VampPrior, log_normal_diag


class _ModulePosterior(PosteriorLatentEncoder):
    """Concrete posterior module for ownership tests."""

    def __init__(self, latent_dimension: int = 16, device: str = "cpu") -> None:
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.projection = torch.nn.Linear(1, 1)

    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del observations
        first_action = next(
            tensor
            for key, tensor in actions.items()
            if key != SampleKey.IS_PAD_ACTION.value
        )
        batch_size = first_action.size(0)
        return {
            LatentKey.POSTERIOR_MU.value: torch.zeros(
                batch_size,
                self.latent_dimension,
                device=first_action.device,
            ),
            LatentKey.POSTERIOR_LOGVAR.value: torch.zeros(
                batch_size,
                self.latent_dimension,
                device=first_action.device,
            ),
            LatentKey.POSTERIOR_LATENT.value: torch.zeros(
                batch_size,
                self.latent_dimension,
                device=first_action.device,
            ),
        }


@pytest.fixture
def vamp_prior_factory(
    mock_action_space_factory: Callable[..., MagicMock],
) -> Callable[..., VampPrior]:
    """Factory for VampPrior instances with mocked ActionSpace."""

    def factory(
        latent_dimension: int = 16,
        num_components: int = 5,
        action_dim: int = 7,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        prediction_horizon: int = 8,
        device: str = "cpu",
        min_logvar: float | None = None,
    ) -> VampPrior:
        action_space = mock_action_space_factory(
            position_dim=action_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
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
        del num_components
        encoder = MagicMock(spec=PosteriorLatentEncoder)
        encoder.latent_dimension = latent_dimension

        def encode_side_effect(
            actions: dict[str, torch.Tensor],
            observations: dict[str, torch.Tensor] | None = None,
        ) -> dict[str, torch.Tensor]:
            del observations
            first_action = next(
                tensor
                for key, tensor in actions.items()
                if key != SampleKey.IS_PAD_ACTION.value
            )
            batch_size = first_action.size(0)
            mu = torch.from_numpy(
                rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
            )
            logvar = torch.zeros(batch_size, latent_dimension, dtype=torch.float32)
            latent = torch.from_numpy(
                rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
            )
            return {
                LatentKey.POSTERIOR_MU.value: mu,
                LatentKey.POSTERIOR_LOGVAR.value: logvar,
                LatentKey.POSTERIOR_LATENT.value: latent,
            }

        encoder.encode.side_effect = encode_side_effect
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


@pytest.fixture
def log_normal_inputs_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Factory for (z, mu, logvar) tuples for log_normal_diag tests."""

    def factory(
        batch_size: int = 4,
        latent_dimension: int = 8,
        logvar_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        mu = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        if logvar_value is not None:
            logvar = torch.full(
                (batch_size, latent_dimension), logvar_value, dtype=torch.float32
            )
        else:
            logvar = torch.from_numpy(
                rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
            )
        return z, mu, logvar

    return factory


class TestLogNormalDiag:
    @pytest.mark.unit
    def test_output_shape(
        self,
        log_normal_inputs_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        batch_size = 4
        latent_dimension = 8
        z, mu, logvar = log_normal_inputs_factory(
            batch_size=batch_size,
            latent_dimension=latent_dimension,
        )
        result = log_normal_diag(z=z, mu=mu, logvar=logvar)
        assert result.shape == (batch_size, latent_dimension)

    @pytest.mark.unit
    def test_max_at_mean(
        self,
        log_normal_inputs_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        latent_dimension = 16
        _, mu, logvar = log_normal_inputs_factory(
            batch_size=1,
            latent_dimension=latent_dimension,
            logvar_value=0.0,
        )
        z_at_mean = mu.clone()
        z_away = mu + 2.0
        log_prob_at_mean = log_normal_diag(z=z_at_mean, mu=mu, logvar=logvar).sum(
            dim=-1
        )
        log_prob_away = log_normal_diag(z=z_away, mu=mu, logvar=logvar).sum(dim=-1)
        assert log_prob_at_mean.item() > log_prob_away.item()

    @pytest.mark.unit
    def test_decreases_away_from_mean(
        self,
        log_normal_inputs_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        latent_dimension = 8
        _, mu, logvar = log_normal_inputs_factory(
            batch_size=1,
            latent_dimension=latent_dimension,
            logvar_value=0.0,
        )
        z_near = mu + 0.5
        z_far = mu + 3.0
        log_prob_near = log_normal_diag(z=z_near, mu=mu, logvar=logvar).sum(dim=-1)
        log_prob_far = log_normal_diag(z=z_far, mu=mu, logvar=logvar).sum(dim=-1)
        assert log_prob_near.item() > log_prob_far.item()


class TestVampPriorInitialization:
    @pytest.mark.unit
    def test_inherits_from_prior_latent_encoder(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        prior = vamp_prior_factory()
        assert isinstance(prior, PriorLatentEncoder)

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_stores_action_layout_from_action_space_metadata(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        prior = vamp_prior_factory(
            action_dim=3,
            has_orientation=True,
            orientation_dim=2,
            has_gripper=True,
            gripper_dim=1,
        )
        assert prior.action_keys == [
            "gripper_action",
            "orientation_action",
            "position_action",
        ]
        assert prior.action_dimensions == [1, 2, 3]
        assert prior.action_dim == 6

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_components, prediction_horizon, action_dim",
        [
            (3, 4, 7),
            (10, 16, 14),
        ],
    )
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
        assert prior.pseudo_inputs.shape == (
            num_components,
            prediction_horizon,
            action_dim,
        )
        assert prior.pseudo_inputs.requires_grad is True

    @pytest.mark.unit
    @pytest.mark.parametrize("num_components", [3, 10])
    def test_log_weights_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        num_components: int,
    ):
        prior = vamp_prior_factory(num_components=num_components)
        assert prior.log_weights.shape == (num_components, 1, 1)
        assert prior.log_weights.requires_grad is True


class TestVampPriorGetAuxiliaryOutputKeys:
    @pytest.mark.unit
    def test_returns_mixture_keys(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ) -> None:
        prior = vamp_prior_factory()
        keys = prior.get_auxiliary_output_keys()
        assert keys == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_LOG_PROB.value,
        }


class TestVampPriorEncoder:
    @pytest.mark.unit
    def test_encoder_raises_when_not_set(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        prior = vamp_prior_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "VampPrior encoder not set. Call wire_posterior() first or ensure "
                "VariationalAlgorithm properly initializes the prior."
            ),
        ):
            _ = prior.encoder

    @pytest.mark.unit
    def test_wire_posterior_stores_encoder(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.wire_posterior(posterior=encoder)
        assert prior.encoder is encoder

    @pytest.mark.unit
    def test_encoder_returns_stored_encoder(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.wire_posterior(posterior=encoder)
        assert prior.encoder is encoder

    @pytest.mark.unit
    def test_wire_posterior_does_not_register_encoder_module(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        prior = vamp_prior_factory()
        encoder = _ModulePosterior()
        prior.wire_posterior(posterior=encoder)

        module_names = {
            name for name, module in prior.named_modules() if module is encoder
        }
        assert module_names == set()

        encoder.train()
        prior.eval()
        assert encoder.training is True


class TestVampPriorGetMixtureParams:
    @pytest.mark.unit
    def test_raises_without_observations(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.wire_posterior(posterior=encoder)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "VampPrior requires observations to compute "
                "q_phi(z | pseudo_actions, observations)."
            ),
        ):
            prior.get_mixture_params()

    @pytest.mark.unit
    def test_returns_conditional_mu_and_logvar(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        latent_dimension = 16
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(latent_dimension=latent_dimension)
        observations = feature_dictionary_factory(batch_size=batch_size)
        prior.wire_posterior(posterior=encoder)
        mu, logvar = prior.get_mixture_params(observations=observations)
        assert mu.shape == (batch_size, num_components, latent_dimension)
        assert logvar.shape == (batch_size, num_components, latent_dimension)

    @pytest.mark.unit
    def test_calls_encoder_encode(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        observations = feature_dictionary_factory(batch_size=batch_size)
        prior.wire_posterior(posterior=encoder)
        prior.get_mixture_params(observations=observations)
        encoder.encode.assert_called_once()
        call_kwargs = encoder.encode.call_args
        for value in call_kwargs.kwargs["observations"].values():
            assert value.shape[0] == batch_size * prior.num_components

    @pytest.mark.unit
    def test_calls_encoder_with_real_action_keys(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        num_components = 5
        prediction_horizon = 8
        prior = vamp_prior_factory(
            num_components=num_components,
            prediction_horizon=prediction_horizon,
            action_dim=3,
            has_orientation=True,
            orientation_dim=2,
            has_gripper=True,
            gripper_dim=1,
        )
        encoder = mock_encoder_factory(num_components=num_components)
        observations = feature_dictionary_factory(batch_size=batch_size)
        prior.wire_posterior(posterior=encoder)
        prior.get_mixture_params(observations=observations)
        actions = encoder.encode.call_args.kwargs["actions"]
        assert set(actions.keys()) == {
            "position_action",
            "orientation_action",
            "gripper_action",
            SampleKey.IS_PAD_ACTION.value,
        }
        assert actions["position_action"].shape == (
            batch_size * num_components,
            prediction_horizon,
            3,
        )
        assert actions["orientation_action"].shape == (
            batch_size * num_components,
            prediction_horizon,
            2,
        )
        assert actions["gripper_action"].shape == (
            batch_size * num_components,
            prediction_horizon,
            1,
        )
        assert actions[SampleKey.IS_PAD_ACTION.value].shape == (
            batch_size * num_components,
            prediction_horizon,
        )

    @pytest.mark.unit
    def test_repeats_observations_for_conditional_components(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        num_components = 5
        prior = vamp_prior_factory(num_components=num_components)
        encoder = mock_encoder_factory(num_components=num_components)
        observations = feature_dictionary_factory(batch_size=batch_size)
        prior.wire_posterior(posterior=encoder)
        prior.get_mixture_params(observations=observations)
        encoded_actions = encoder.encode.call_args.kwargs["actions"]
        encoded_observations = encoder.encode.call_args.kwargs["observations"]
        expected_batch_size = batch_size * num_components
        assert encoded_actions["position_action"].shape[0] == expected_batch_size
        for value in encoded_observations.values():
            assert value.shape[0] == expected_batch_size

    @pytest.mark.unit
    def test_clamps_logvar_when_min_logvar_set(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
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

        def encode_side_effect(
            actions: dict[str, torch.Tensor],
            observations: dict[str, torch.Tensor] | None = None,
        ) -> dict[str, torch.Tensor]:
            del observations
            first_action = next(
                tensor
                for key, tensor in actions.items()
                if key != SampleKey.IS_PAD_ACTION.value
            )
            batch_size = first_action.size(0)
            return {
                LatentKey.POSTERIOR_MU.value: torch.zeros(
                    batch_size,
                    latent_dimension,
                ),
                LatentKey.POSTERIOR_LOGVAR.value: torch.full(
                    (batch_size, latent_dimension),
                    -10.0,
                ),
                LatentKey.POSTERIOR_LATENT.value: torch.zeros(
                    batch_size,
                    latent_dimension,
                ),
            }

        encoder.encode.side_effect = encode_side_effect
        observations = feature_dictionary_factory(batch_size=4)
        prior.wire_posterior(posterior=encoder)
        _, logvar = prior.get_mixture_params(observations=observations)
        assert torch.all(logvar >= min_logvar)


class TestVampPriorBuildTrainingTarget:
    @pytest.mark.unit
    def test_preserves_posterior_latent_gradient(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
    ):
        latent = torch.ones(2, 4, requires_grad=True)
        prior = vamp_prior_factory(latent_dimension=4)
        target = prior.build_training_target({LatentKey.POSTERIOR_LATENT.value: latent})
        loss = target.sum()
        loss.backward()
        assert latent.grad is not None
        assert torch.all(latent.grad == 1.0)


class TestVampPriorSamplePrior:
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [2, 8])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
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
        prior.wire_posterior(posterior=encoder)
        observations = feature_dictionary_factory(batch_size=batch_size)
        sample = prior.sample_prior(batch_size=batch_size, observations=observations)
        assert sample.shape == (batch_size, latent_dimension)

    @pytest.mark.unit
    def test_calls_get_mixture_params(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        observations = feature_dictionary_factory(batch_size=4)
        prior.wire_posterior(posterior=encoder)
        prior.sample_prior(batch_size=4, observations=observations)
        encoder.encode.assert_called_once()

    @pytest.mark.unit
    def test_sample_prior_raises_without_observations(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.wire_posterior(posterior=encoder)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "VampPrior requires observations to compute "
                "q_phi(z | pseudo_actions, observations)."
            ),
        ):
            prior.sample_prior(batch_size=4)

    @pytest.mark.unit
    def test_conditional_output_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        latent_dimension = 16
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(latent_dimension=latent_dimension)
        observations = feature_dictionary_factory(batch_size=batch_size)
        prior.wire_posterior(posterior=encoder)
        sample = prior.sample_prior(batch_size=batch_size, observations=observations)
        assert sample.shape == (batch_size, latent_dimension)


class TestVampPriorForward:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "use_none_target, expectation",
        [
            (
                False,
                does_not_raise(),
            ),
            (
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "VampPrior.forward() requires target_latents for "
                        "log-prob computation. Use sample_prior() for "
                        "inference."
                    ),
                ),
            ),
        ],
    )
    def test_target_latents_validation(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        use_none_target: bool,
        expectation: AbstractContextManager,
    ):
        latent_dimension = 16
        prior = vamp_prior_factory(latent_dimension=latent_dimension)
        encoder = mock_encoder_factory(latent_dimension=latent_dimension)
        prior.wire_posterior(posterior=encoder)
        observations = feature_dictionary_factory(batch_size=4)
        target_latents = (
            None
            if use_none_target
            else latent_tensor_factory(batch_size=4, latent_dimension=latent_dimension)
        )
        with expectation:
            prior.forward(target_latents=target_latents, observations=observations)

    @pytest.mark.unit
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
        prior.wire_posterior(posterior=encoder)
        target_latents = latent_tensor_factory(
            batch_size=4,
            latent_dimension=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=4)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert set(result.keys()) == {
            LatentKey.PRIOR_LOG_PROB.value,
        }

    @pytest.mark.unit
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
        prior.wire_posterior(posterior=encoder)
        target_latents = latent_tensor_factory(
            batch_size=batch_size,
            latent_dimension=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=batch_size)
        result = prior.forward(
            target_latents=target_latents,
            observations=observations,
        )
        assert result[LatentKey.PRIOR_LOG_PROB.value].shape == (batch_size,)


class TestVampPriorLogProb:
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dimension", [8, 32])
    def test_output_shape(
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
        prior.wire_posterior(posterior=encoder)
        z = latent_tensor_factory(
            batch_size=batch_size,
            latent_dimension=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=batch_size)
        log_prob = prior.log_prob(z=z, observations=observations)
        assert log_prob.shape == (batch_size,)

    @pytest.mark.unit
    def test_conditional_output_shape(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 4
        latent_dimension = 16
        num_components = 5
        prior = vamp_prior_factory(
            latent_dimension=latent_dimension,
            num_components=num_components,
        )
        encoder = mock_encoder_factory(latent_dimension=latent_dimension)
        observations = feature_dictionary_factory(batch_size=batch_size)
        prior.wire_posterior(posterior=encoder)
        z = latent_tensor_factory(
            batch_size=batch_size,
            latent_dimension=latent_dimension,
        )
        log_prob = prior.log_prob(z=z, observations=observations)
        assert log_prob.shape == (batch_size,)

    @pytest.mark.unit
    def test_returns_finite_values(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
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
        prior.wire_posterior(posterior=encoder)
        z = latent_tensor_factory(
            batch_size=4,
            latent_dimension=latent_dimension,
        )
        observations = feature_dictionary_factory(batch_size=4)
        log_prob = prior.log_prob(z=z, observations=observations)
        assert torch.all(torch.isfinite(log_prob))

    @pytest.mark.unit
    def test_log_prob_raises_without_observations(
        self,
        vamp_prior_factory: Callable[..., VampPrior],
        mock_encoder_factory: Callable[..., MagicMock],
        latent_tensor_factory: Callable[..., torch.Tensor],
    ):
        prior = vamp_prior_factory()
        encoder = mock_encoder_factory()
        prior.wire_posterior(posterior=encoder)
        z = latent_tensor_factory(batch_size=4)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "VampPrior requires observations to compute "
                "q_phi(z | pseudo_actions, observations)."
            ),
        ):
            prior.log_prob(z=z)
