"""Tests for versatil.models.decoding.algorithm.variational module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.configs.experiment import ExperimentConfig
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.constants import DenoisingAlgorithm, LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.dit_prior import DiTPrior
from versatil.models.decoding.latent.prior.gaussian_prior import GaussianPrior
from versatil.models.decoding.latent.prior.vamp_prior import VampPrior
from versatil.training.callbacks.latent_visualization import LatentVisualizationCallback
from versatil.training.callbacks.prior_target_standardization import (
    PriorTargetStandardizationCallback,
)

LATENT_DIMENSION = 32


@pytest.fixture
def mock_posterior_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock PosteriorLatentEncoder."""

    def factory(
        latent_dimension: int = LATENT_DIMENSION,
        device: str = "cpu",
    ) -> MagicMock:
        posterior = MagicMock(spec=PosteriorLatentEncoder)
        posterior.latent_dimension = latent_dimension
        posterior.device = torch.device(device)
        posterior.encode.return_value = {
            LatentKey.POSTERIOR_LATENT.value: torch.from_numpy(
                rng.standard_normal((2, latent_dimension)).astype(np.float32)
            ),
            LatentKey.POSTERIOR_MU.value: torch.zeros(2, latent_dimension),
            LatentKey.POSTERIOR_LOGVAR.value: torch.zeros(2, latent_dimension),
        }
        posterior.get_auxiliary_output_keys.return_value = {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.POSTERIOR_MU.value,
            LatentKey.POSTERIOR_LOGVAR.value,
        }
        return posterior

    return factory


@pytest.fixture
def mock_prior_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock PriorLatentEncoder."""

    def factory(
        latent_dimension: int = LATENT_DIMENSION,
        include_prior_latent: bool = True,
    ) -> MagicMock:
        prior = MagicMock(spec=PriorLatentEncoder)
        prior.latent_dimension = latent_dimension
        forward_result = {
            LatentKey.PRIOR_MU.value: torch.zeros(2, latent_dimension),
            LatentKey.PRIOR_LOGVAR.value: torch.zeros(2, latent_dimension),
        }
        if include_prior_latent:
            forward_result[LatentKey.PRIOR_LATENT.value] = torch.from_numpy(
                rng.standard_normal((2, latent_dimension)).astype(np.float32)
            )
        prior.forward.return_value = forward_result
        prior.build_training_target.side_effect = lambda posterior_output: (
            posterior_output[LatentKey.POSTERIOR_LATENT.value].detach()
        )
        prior.sample_prior.return_value = torch.from_numpy(
            rng.standard_normal((2, latent_dimension)).astype(np.float32)
        )
        prior.get_auxiliary_output_keys.return_value = {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_MU.value,
            LatentKey.PRIOR_LOGVAR.value,
        }
        return prior

    return factory


@pytest.fixture
def mock_base_algorithm_factory() -> Callable[..., MagicMock]:
    """Factory for mock base DecodingAlgorithm."""

    def factory() -> MagicMock:
        algorithm = MagicMock(spec=DecodingAlgorithm)
        default_output = {
            "position_action": torch.zeros(2, 8, 3),
        }
        algorithm.forward.return_value = default_output
        algorithm.predict.return_value = default_output
        return algorithm

    return factory


@pytest.fixture
def variational_factory(
    mock_posterior_factory: Callable[..., MagicMock],
    mock_prior_factory: Callable[..., MagicMock],
    mock_base_algorithm_factory: Callable[..., MagicMock],
) -> Callable[..., VariationalAlgorithm]:
    """Factory for VariationalAlgorithm instances with mocked dependencies."""

    def factory(
        base_algorithm: MagicMock | None = None,
        posterior_encoder: MagicMock | None = None,
        prior: MagicMock | None = None,
        sampling_from_prior_probability: float = 0.0,
        posterior_decoder_noise_std: float = 0.0,
        latent_dimension: int = LATENT_DIMENSION,
    ) -> VariationalAlgorithm:
        if base_algorithm is None:
            base_algorithm = mock_base_algorithm_factory()
        if posterior_encoder is None:
            posterior_encoder = mock_posterior_factory(
                latent_dimension=latent_dimension
            )
        if prior is None:
            prior = mock_prior_factory(latent_dimension=latent_dimension)
        return VariationalAlgorithm(
            base_algorithm=base_algorithm,
            posterior_encoder=posterior_encoder,
            prior=prior,
            sampling_from_prior_probability=sampling_from_prior_probability,
            posterior_decoder_noise_std=posterior_decoder_noise_std,
        )

    return factory


class TestVariationalAlgorithmInitialization:
    def test_inherits_from_decoding_algorithm(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
    ):
        algo = variational_factory()
        assert isinstance(algo, DecodingAlgorithm)

    @pytest.mark.parametrize("sampling_from_prior_probability", [0.0, 0.5])
    @pytest.mark.parametrize("posterior_decoder_noise_std", [0.0, 0.135])
    def test_stores_configuration(
        self,
        mock_base_algorithm_factory: Callable[..., MagicMock],
        mock_posterior_factory: Callable[..., MagicMock],
        mock_prior_factory: Callable[..., MagicMock],
        sampling_from_prior_probability: float,
        posterior_decoder_noise_std: float,
    ):
        base = mock_base_algorithm_factory()
        posterior = mock_posterior_factory()
        prior = mock_prior_factory()
        algo = VariationalAlgorithm(
            base_algorithm=base,
            posterior_encoder=posterior,
            prior=prior,
            sampling_from_prior_probability=sampling_from_prior_probability,
            posterior_decoder_noise_std=posterior_decoder_noise_std,
        )
        assert algo.p_prior == sampling_from_prior_probability
        assert algo.posterior_decoder_noise_std == posterior_decoder_noise_std
        assert algo.base_algorithm is base
        assert algo.posterior_encoder is posterior
        assert algo.prior is prior

    def test_negative_posterior_decoder_noise_raises(
        self,
        mock_base_algorithm_factory: Callable[..., MagicMock],
        mock_posterior_factory: Callable[..., MagicMock],
        mock_prior_factory: Callable[..., MagicMock],
    ):
        posterior_decoder_noise_std = -0.1
        with pytest.raises(
            ValueError,
            match=re.escape(
                "posterior_decoder_noise_std must be non-negative, "
                f"got {posterior_decoder_noise_std}."
            ),
        ):
            VariationalAlgorithm(
                base_algorithm=mock_base_algorithm_factory(),
                posterior_encoder=mock_posterior_factory(),
                prior=mock_prior_factory(),
                posterior_decoder_noise_std=posterior_decoder_noise_std,
            )

    def test_auto_creates_gaussian_prior_when_none(
        self,
        mock_posterior_factory: Callable[..., MagicMock],
        mock_base_algorithm_factory: Callable[..., MagicMock],
    ):
        posterior = mock_posterior_factory(latent_dimension=16)
        base = mock_base_algorithm_factory()
        algo = VariationalAlgorithm(
            base_algorithm=base,
            posterior_encoder=posterior,
            prior=None,
        )
        assert isinstance(algo.prior, GaussianPrior)
        assert algo.prior.latent_dimension == 16

    def test_latent_dimension_mismatch_raises(
        self,
        mock_posterior_factory: Callable[..., MagicMock],
        mock_prior_factory: Callable[..., MagicMock],
        mock_base_algorithm_factory: Callable[..., MagicMock],
    ):
        posterior = mock_posterior_factory(latent_dimension=32)
        prior = mock_prior_factory(latent_dimension=16)
        base = mock_base_algorithm_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Latent dimension mismatch: prior.latent_dim=16 "
                "!= posterior_encoder.latent_dim=32"
            ),
        ):
            VariationalAlgorithm(
                base_algorithm=base,
                posterior_encoder=posterior,
                prior=prior,
            )

    def test_vamp_prior_receives_encoder(
        self,
        input_tensor_factory: Callable[..., torch.Tensor],
        mock_posterior_factory: Callable[..., MagicMock],
        mock_base_algorithm_factory: Callable[..., MagicMock],
    ):
        posterior = mock_posterior_factory()
        base = mock_base_algorithm_factory()
        vamp_prior = MagicMock(spec=VampPrior)
        vamp_prior.latent_dimension = LATENT_DIMENSION
        vamp_prior.forward.return_value = {
            LatentKey.PRIOR_LATENT.value: input_tensor_factory(
                batch_size=2, input_dimension=LATENT_DIMENSION
            ),
        }
        VariationalAlgorithm(
            base_algorithm=base,
            posterior_encoder=posterior,
            prior=vamp_prior,
        )
        vamp_prior.wire_posterior.assert_called_once_with(posterior)


class TestVariationalAlgorithmForward:
    def test_raises_without_actions(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Actions must be provided during training for variational algorithm."
            ),
        ):
            algo.forward(network=network, features=features, actions=None)

    def test_calls_posterior_encode(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        algo.posterior_encoder.encode.assert_called_once()

    def test_calls_prior_forward(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        algo.prior.forward.assert_called_once()

    def test_prior_receives_detached_latent(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        algo = variational_factory()
        # Replace posterior output with a tensor that requires grad
        posterior_z = input_tensor_factory(
            batch_size=2, input_dimension=LATENT_DIMENSION
        ).requires_grad_(True)
        algo.posterior_encoder.encode.return_value = {
            LatentKey.POSTERIOR_LATENT.value: posterior_z,
            LatentKey.POSTERIOR_MU.value: torch.zeros(2, LATENT_DIMENSION),
            LatentKey.POSTERIOR_LOGVAR.value: torch.zeros(2, LATENT_DIMENSION),
        }
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        # The target_latents passed to prior.forward should be detached
        prior_call_kwargs = algo.prior.forward.call_args.kwargs
        target_latents_passed = prior_call_kwargs["target_latents"]
        assert not target_latents_passed.requires_grad

    def test_prior_builds_training_target(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )

        algo.forward(network=network, features=features, actions=actions)

        algo.prior.build_training_target.assert_called_once_with(
            algo.posterior_encoder.encode.return_value
        )

    def test_delegates_to_base_algorithm(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        algo.base_algorithm.forward.assert_called_once()

    def test_features_passed_to_base_include_latent(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        features_passed = algo.base_algorithm.forward.call_args.kwargs["features"]
        assert LatentKey.POSTERIOR_LATENT.value in features_passed

    def test_output_includes_posterior_keys(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        result = algo.forward(network=network, features=features, actions=actions)
        assert LatentKey.POSTERIOR_LATENT.value in result
        assert LatentKey.POSTERIOR_MU.value in result
        assert LatentKey.POSTERIOR_LOGVAR.value in result

    def test_output_includes_prior_keys(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        result = algo.forward(network=network, features=features, actions=actions)
        assert LatentKey.PRIOR_MU.value in result
        assert LatentKey.PRIOR_LOGVAR.value in result

    def test_training_mode_uses_posterior_latent(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory(sampling_from_prior_probability=0.0)
        algo.train()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        features_passed = algo.base_algorithm.forward.call_args.kwargs["features"]
        # With p_prior=0.0, latent should be from posterior
        posterior_z = algo.posterior_encoder.encode.return_value[
            LatentKey.POSTERIOR_LATENT.value
        ]
        assert torch.equal(
            features_passed[LatentKey.POSTERIOR_LATENT.value],
            posterior_z,
        )

    def test_training_mode_adds_fixed_noise_to_decoder_posterior_latent(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        posterior_decoder_noise_std = 0.25
        algo = variational_factory(
            sampling_from_prior_probability=0.0,
            posterior_decoder_noise_std=posterior_decoder_noise_std,
        )
        algo.train()
        posterior_latent = algo.posterior_encoder.encode.return_value[
            LatentKey.POSTERIOR_LATENT.value
        ]
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        with patch(
            "versatil.models.decoding.algorithm.variational.torch.randn_like",
            return_value=torch.ones_like(posterior_latent),
        ):
            result = algo.forward(network=network, features=features, actions=actions)

        expected_latent = posterior_latent + posterior_decoder_noise_std
        features_passed = algo.base_algorithm.forward.call_args.kwargs["features"]
        assert torch.equal(
            features_passed[LatentKey.POSTERIOR_LATENT.value],
            expected_latent,
        )
        assert torch.equal(
            result[LatentKey.POSTERIOR_LATENT.value],
            posterior_latent,
        )

    def test_posterior_decoder_noise_keeps_prior_target_clean(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory(
            sampling_from_prior_probability=0.0,
            posterior_decoder_noise_std=0.25,
        )
        algo.train()
        posterior_latent = algo.posterior_encoder.encode.return_value[
            LatentKey.POSTERIOR_LATENT.value
        ]
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        with patch(
            "versatil.models.decoding.algorithm.variational.torch.randn_like",
            return_value=torch.ones_like(posterior_latent),
        ):
            algo.forward(network=network, features=features, actions=actions)

        prior_call_kwargs = algo.prior.forward.call_args.kwargs
        assert torch.equal(
            prior_call_kwargs["target_latents"],
            posterior_latent.detach(),
        )

    def test_eval_mode_uses_prior_latent(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory(sampling_from_prior_probability=0.0)
        algo.eval()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        features_passed = algo.base_algorithm.forward.call_args.kwargs["features"]
        # In eval mode, latent should be from prior
        prior_z = algo.prior.forward.return_value[LatentKey.PRIOR_LATENT.value]
        assert torch.equal(
            features_passed[LatentKey.POSTERIOR_LATENT.value],
            prior_z,
        )

    def test_p_prior_one_uses_prior_latent_during_training(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory(sampling_from_prior_probability=1.0)
        algo.train()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        features_passed = algo.base_algorithm.forward.call_args.kwargs["features"]
        prior_z = algo.prior.forward.return_value[LatentKey.PRIOR_LATENT.value]
        assert torch.equal(
            features_passed[LatentKey.POSTERIOR_LATENT.value],
            prior_z,
        )

    def test_p_prior_one_skips_posterior_decoder_noise(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory(
            sampling_from_prior_probability=1.0,
            posterior_decoder_noise_std=0.25,
        )
        algo.train()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        with patch(
            "versatil.models.decoding.algorithm.variational.torch.randn_like",
        ) as random_like:
            algo.forward(network=network, features=features, actions=actions)

        random_like.assert_not_called()

    def test_samples_prior_when_prior_latent_not_in_output(
        self,
        mock_posterior_factory: Callable[..., MagicMock],
        mock_prior_factory: Callable[..., MagicMock],
        mock_base_algorithm_factory: Callable[..., MagicMock],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = mock_prior_factory(include_prior_latent=False)
        algo = VariationalAlgorithm(
            base_algorithm=mock_base_algorithm_factory(),
            posterior_encoder=mock_posterior_factory(),
            prior=prior,
        )
        algo.eval()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        prior.sample_prior.assert_called_once()

    def test_p_prior_zero_skips_sample_prior_during_training(
        self,
        mock_posterior_factory: Callable[..., MagicMock],
        mock_prior_factory: Callable[..., MagicMock],
        mock_base_algorithm_factory: Callable[..., MagicMock],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        prior = mock_prior_factory(include_prior_latent=False)
        algo = VariationalAlgorithm(
            base_algorithm=mock_base_algorithm_factory(),
            posterior_encoder=mock_posterior_factory(),
            prior=prior,
            sampling_from_prior_probability=0.0,
        )
        algo.train()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        algo.forward(network=network, features=features, actions=actions)
        # With p_prior=0, sample_prior should never be called during training
        prior.sample_prior.assert_not_called()


class TestVariationalAlgorithmGetTargets:
    def test_delegates_to_base_algorithm(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_base_algorithm_factory: Callable[..., MagicMock],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        base = mock_base_algorithm_factory()
        expected_targets = {"position_action": torch.ones(2, 8, 3)}
        base.get_targets.return_value = expected_targets
        variational = variational_factory(base_algorithm=base)
        output = {"position_action": torch.zeros(2, 8, 3)}
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        targets = variational.get_targets(
            algorithm_output=output,
            ground_truth_actions=actions,
        )
        base.get_targets.assert_called_once_with(output, actions)
        assert targets is expected_targets


class TestVariationalAlgorithmPredict:
    def test_samples_from_prior(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        algo.predict(network=network, features=features)
        algo.prior.sample_prior.assert_called_once()

    def test_delegates_to_base_algorithm_predict(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        algo.predict(network=network, features=features)
        algo.base_algorithm.predict.assert_called_once()

    def test_does_not_call_base_algorithm_forward(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        algo.predict(network=network, features=features)
        algo.base_algorithm.forward.assert_not_called()

    def test_features_include_latent(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        algo.predict(network=network, features=features)
        features_passed = algo.base_algorithm.predict.call_args.kwargs["features"]
        assert LatentKey.POSTERIOR_LATENT.value in features_passed

    def test_passes_network_to_base_algorithm(
        self,
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        algo = variational_factory()
        network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        algo.predict(network=network, features=features)
        network_passed = algo.base_algorithm.predict.call_args.kwargs["network"]
        assert network_passed is network

    def test_returns_base_algorithm_output(
        self,
        mock_base_algorithm_factory: Callable[..., MagicMock],
        variational_factory: Callable[..., VariationalAlgorithm],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        base = mock_base_algorithm_factory()
        base.predict.return_value = {"position_action": torch.zeros(2, 8, 3)}
        algo = variational_factory(base_algorithm=base)
        network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        result = algo.predict(network=network, features=features)
        assert set(result.keys()) == {"position_action"}


def test_get_auxiliary_output_keys(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    base.get_auxiliary_output_keys.return_value = set()
    algo = variational_factory(base_algorithm=base)
    keys = algo.get_auxiliary_output_keys()
    assert LatentKey.POSTERIOR_LOGVAR.value in keys
    assert LatentKey.PRIOR_LOGVAR.value in keys
    assert LatentKey.POSTERIOR_MU.value in keys
    assert LatentKey.PRIOR_MU.value in keys


def test_deterministic_posterior_excludes_logvar_from_auxiliary_keys(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
    mock_posterior_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    base.get_auxiliary_output_keys.return_value = set()
    posterior = mock_posterior_factory()
    posterior.deterministic = True
    posterior.get_auxiliary_output_keys.return_value = {
        LatentKey.POSTERIOR_LATENT.value,
        LatentKey.POSTERIOR_MU.value,
    }
    algo = variational_factory(base_algorithm=base, posterior_encoder=posterior)
    keys = algo.get_auxiliary_output_keys()
    assert LatentKey.POSTERIOR_LOGVAR.value not in keys
    # Prior is not deterministic by default, so prior logvar should still be present
    assert LatentKey.PRIOR_LOGVAR.value in keys


def test_vq_posterior_keys_propagate_through_algorithm(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
    mock_posterior_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    base.get_auxiliary_output_keys.return_value = set()
    posterior = mock_posterior_factory()
    posterior.get_auxiliary_output_keys.return_value = {
        LatentKey.POSTERIOR_LATENT.value,
        LatentKey.VQ_INDICES.value,
        LatentKey.VQ_Z_CONTINUOUS.value,
        LatentKey.VQ_QUANTIZED.value,
    }
    algo = variational_factory(base_algorithm=base, posterior_encoder=posterior)
    keys = algo.get_auxiliary_output_keys()
    assert LatentKey.VQ_INDICES.value in keys
    assert LatentKey.VQ_Z_CONTINUOUS.value in keys
    assert LatentKey.VQ_QUANTIZED.value in keys
    assert LatentKey.POSTERIOR_LOGVAR.value not in keys


def test_vq_codebook_prior_keys_do_not_collide_with_posterior(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
    mock_posterior_factory: Callable[..., MagicMock],
    mock_prior_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    base.get_auxiliary_output_keys.return_value = set()
    posterior = mock_posterior_factory()
    posterior.get_auxiliary_output_keys.return_value = {
        LatentKey.POSTERIOR_LATENT.value,
        LatentKey.VQ_INDICES.value,
        LatentKey.VQ_Z_CONTINUOUS.value,
        LatentKey.VQ_QUANTIZED.value,
    }
    prior = mock_prior_factory()
    prior.get_auxiliary_output_keys.return_value = {
        LatentKey.PRIOR_LATENT.value,
        LatentKey.VQ_PRIOR_INDICES.value,
        LatentKey.PRIOR_CODE_LOGITS.value,
    }
    algo = variational_factory(
        base_algorithm=base, posterior_encoder=posterior, prior=prior
    )
    keys = algo.get_auxiliary_output_keys()
    assert LatentKey.VQ_INDICES.value in keys
    assert LatentKey.VQ_PRIOR_INDICES.value in keys


def test_vq_forward_preserves_posterior_indices_after_prior_update(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
    mock_posterior_factory: Callable[..., MagicMock],
    mock_prior_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    base.get_auxiliary_output_keys.return_value = set()
    base.forward.return_value = {"position_action": torch.zeros(2, 8, 3)}
    posterior_indices = [torch.tensor([7, 3])]
    prior_indices = [torch.tensor([0, 0])]
    posterior = mock_posterior_factory()
    posterior.encode.return_value = {
        LatentKey.POSTERIOR_LATENT.value: torch.zeros(2, LATENT_DIMENSION),
        LatentKey.VQ_INDICES.value: posterior_indices,
    }
    posterior.get_auxiliary_output_keys.return_value = {
        LatentKey.POSTERIOR_LATENT.value,
        LatentKey.VQ_INDICES.value,
    }
    prior = mock_prior_factory()
    prior.forward.return_value = {
        LatentKey.PRIOR_LATENT.value: torch.zeros(2, LATENT_DIMENSION),
        LatentKey.VQ_PRIOR_INDICES.value: prior_indices,
    }
    prior.get_auxiliary_output_keys.return_value = {
        LatentKey.PRIOR_LATENT.value,
        LatentKey.VQ_PRIOR_INDICES.value,
    }
    algo = variational_factory(
        base_algorithm=base, posterior_encoder=posterior, prior=prior
    )
    algo.train()

    features = {"obs": torch.zeros(2, 4)}
    actions = {"position_action": torch.zeros(2, 8, 3)}
    predictions = algo.forward(network=MagicMock(), features=features, actions=actions)

    for returned, expected in zip(
        predictions[LatentKey.VQ_INDICES.value], posterior_indices, strict=True
    ):
        assert torch.equal(returned, expected)
    for returned, expected in zip(
        predictions[LatentKey.VQ_PRIOR_INDICES.value], prior_indices, strict=True
    ):
        assert torch.equal(returned, expected)


def test_dit_prior_keys_propagate_through_algorithm(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
    mock_prior_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    base.get_auxiliary_output_keys.return_value = set()
    prior = mock_prior_factory()
    prior.get_auxiliary_output_keys.return_value = {
        LatentKey.PRIOR_LATENT.value,
        LatentKey.PRIOR_PREDICTION.value,
        LatentKey.PRIOR_TARGET.value,
    }
    algo = variational_factory(base_algorithm=base, prior=prior)
    keys = algo.get_auxiliary_output_keys()
    assert LatentKey.PRIOR_PREDICTION.value in keys
    assert LatentKey.PRIOR_TARGET.value in keys
    assert LatentKey.PRIOR_MU.value not in keys
    assert LatentKey.PRIOR_LOGVAR.value not in keys


def test_get_callbacks_returns_latent_visualization(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    algo = variational_factory(base_algorithm=base)
    experiment_config = MagicMock(spec=ExperimentConfig)
    experiment_config.val_every = 4
    callbacks = algo.get_callbacks(experiment_config=experiment_config)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LatentVisualizationCallback)
    assert callbacks[0].log_every_n_epochs == 4


def test_get_callbacks_adds_prior_target_standardization_for_enabled_standardizer(
    variational_factory: Callable[..., VariationalAlgorithm],
    mock_base_algorithm_factory: Callable[..., MagicMock],
):
    base = mock_base_algorithm_factory()
    prior = DiTPrior(
        latent_dimension=LATENT_DIMENSION,
        embedding_dimension=16,
        number_of_heads=2,
        number_of_layers=1,
        feedforward_dimension=32,
        device="cpu",
        algorithm_type=DenoisingAlgorithm.DIFFUSION.value,
        latent_standardization_max_batches=3,
    )
    algo = variational_factory(base_algorithm=base, prior=prior)
    experiment_config = MagicMock(spec=ExperimentConfig)
    experiment_config.val_every = 4

    callbacks = algo.get_callbacks(experiment_config=experiment_config)

    prior_callbacks = [
        callback
        for callback in callbacks
        if isinstance(callback, PriorTargetStandardizationCallback)
    ]
    assert len(prior_callbacks) == 1
    assert prior_callbacks[0].max_batches == 3
