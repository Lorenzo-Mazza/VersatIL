"""Tests for versatil.models.decoding.algorithm.flow_matching module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from versatil.data.constants import SampleKey
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.flow_matching import FlowMatching
from versatil.models.decoding.constants import DecoderOutputKey, ODESolver
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler


@pytest.fixture
def flow_matching_factory() -> Callable[..., FlowMatching]:
    """Factory for FlowMatching instances."""
    def factory(
        sigma: float = 0.0,
        num_inference_steps: int = 10,
        ode_solver: str = ODESolver.EULER.value,
        timestep_sampler: str = TimestepSampler.BETA.value,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        max_timestep: float = 0.999,
    ) -> FlowMatching:
        return FlowMatching(
            sigma=sigma,
            num_inference_steps=num_inference_steps,
            ode_solver=ode_solver,
            timestep_sampler=timestep_sampler,
            logit_mean=logit_mean,
            logit_std=logit_std,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            max_timestep=max_timestep,
        )
    return factory


class TestFlowMatchingInitialization:

    def test_inherits_from_decoding_algorithm(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
    ):
        fm = flow_matching_factory()
        assert isinstance(fm, DecodingAlgorithm)

    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    @pytest.mark.parametrize("ode_solver", [
        ODESolver.EULER.value,
        ODESolver.HEUN.value,
    ])
    @pytest.mark.parametrize("sigma", [0.0, 0.1])
    @pytest.mark.parametrize("timestep_sampler", [
        TimestepSampler.BETA.value,
        TimestepSampler.UNIFORM.value,
    ])
    @pytest.mark.parametrize("logit_mean", [0.0, 0.5])
    @pytest.mark.parametrize("logit_std", [0.5, 1.0])
    @pytest.mark.parametrize("beta_alpha", [1.0, 1.5])
    @pytest.mark.parametrize("beta_beta", [0.5, 1.0])
    @pytest.mark.parametrize("max_timestep", [0.99, 0.999])
    def test_stores_configuration(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        num_inference_steps: int,
        ode_solver: str,
        sigma: float,
        timestep_sampler: str,
        logit_mean: float,
        logit_std: float,
        beta_alpha: float,
        beta_beta: float,
        max_timestep: float,
    ):
        fm = flow_matching_factory(
            num_inference_steps=num_inference_steps,
            ode_solver=ode_solver,
            sigma=sigma,
            timestep_sampler=timestep_sampler,
            logit_mean=logit_mean,
            logit_std=logit_std,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            max_timestep=max_timestep,
        )
        assert fm.num_inference_steps == num_inference_steps
        assert fm.ode_solver == ode_solver
        assert fm.timestep_sampler == timestep_sampler
        assert fm.logit_mean == logit_mean
        assert fm.logit_std == logit_std
        assert fm.beta_alpha == beta_alpha
        assert fm.beta_beta == beta_beta
        assert fm.max_timestep == max_timestep
        assert isinstance(fm.flow_matcher, ConditionalFlowMatcher)

    @pytest.mark.parametrize("ode_solver, expectation", [
        (ODESolver.EULER.value, does_not_raise()),
        (ODESolver.RK4.value, does_not_raise()),
        (
            "invalid_solver",
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"Unknown ODE solver: invalid_solver. "
                    f"Expected one of {[e.value for e in ODESolver]}"
                ),
            ),
        ),
    ])
    def test_ode_solver_validation(
        self,
        ode_solver: str,
        expectation,
    ):
        with expectation:
            FlowMatching(ode_solver=ode_solver)

    @pytest.mark.parametrize("timestep_sampler, expectation", [
        (TimestepSampler.BETA.value, does_not_raise()),
        (TimestepSampler.UNIFORM.value, does_not_raise()),
        (
            "invalid_sampler",
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"Unknown timestep sampler: invalid_sampler. "
                    f"Expected one of {[e.value for e in TimestepSampler]}"
                ),
            ),
        ),
    ])
    def test_timestep_sampler_validation(
        self,
        timestep_sampler: str,
        expectation,
    ):
        with expectation:
            FlowMatching(timestep_sampler=timestep_sampler)


class TestFlowMatchingForward:

    def test_raises_without_actions(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        fm = flow_matching_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        with pytest.raises(
            ValueError,
            match="Flow Matching algorithm requires actions during training",
        ):
            fm.forward(network=mock_network, features=features, actions=None)

    @pytest.mark.parametrize("include_padding_mask", [True, False])
    def test_output_contains_exact_keys(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        include_padding_mask: bool,
    ):
        fm = flow_matching_factory()
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=include_padding_mask,
        )
        result = fm.forward(network=mock_network, features=features, actions=actions)
        expected_keys = {
            "position_action",
            DecoderOutputKey.TARGET_VELOCITY.value,
            DecoderOutputKey.NOISE.value,
            DecoderOutputKey.TIMESTEP.value,
            SampleKey.IS_PAD_ACTION.value,
        }
        assert set(result.keys()) == expected_keys
        assert set(result[DecoderOutputKey.TARGET_VELOCITY.value].keys()) == {"position_action"}
        if include_padding_mask:
            assert result[SampleKey.IS_PAD_ACTION.value] is not None
        else:
            assert result[SampleKey.IS_PAD_ACTION.value] is None

    def test_network_receives_timestep_in_features(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        fm = flow_matching_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"], prediction_horizon=8, action_dimension=3,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        fm.forward(network=mock_network, features=features, actions=actions)
        features_passed = mock_network.call_args.kwargs["features"]
        assert DecoderOutputKey.TIMESTEP.value in features_passed


class TestFlowMatchingPredict:

    def test_predict_returns_exact_action_keys(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        fm = flow_matching_factory(num_inference_steps=2)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        with patch(
            "versatil.models.decoding.algorithm.flow_matching.integrate_ode",
        ) as mock_integrate:
            mock_integrate.return_value = torch.zeros(2, 8 * 3)
            result = fm.predict(network=mock_network, features=features)
        assert set(result.keys()) == {"position_action"}

    def test_predict_output_shape(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        fm = flow_matching_factory(num_inference_steps=2)
        mock_network = mock_action_decoder_factory(action_keys=["position_action"])
        features = feature_dictionary_factory()
        with patch(
            "versatil.models.decoding.algorithm.flow_matching.integrate_ode",
        ) as mock_integrate:
            mock_integrate.return_value = torch.zeros(2, 8 * 3)
            result = fm.predict(network=mock_network, features=features)
        assert result["position_action"].shape == (2, 8, 3)