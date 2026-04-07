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
from versatil.models.decoding.algorithm.flow_matching import (
    FlowMatching,
    VelocityWrapper,
)
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
        reverse_flow_convention: bool = False,
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
            reverse_flow_convention=reverse_flow_convention,
        )

    return factory


@pytest.fixture
def velocity_wrapper_factory() -> Callable[..., VelocityWrapper]:
    """Factory for VelocityWrapper instances."""

    def factory(
        action_keys: list[str] | None = None,
        prediction_horizon: int = 8,
        prediction_dimension: int = 3,
        batch_size: int = 2,
        reverse_convention: bool = False,
    ) -> VelocityWrapper:
        if action_keys is None:
            action_keys = ["position_action"]
        network = MagicMock()
        network.return_value = {
            key: torch.ones(batch_size, prediction_horizon, prediction_dimension)
            for key in action_keys
        }
        features = {"feature": torch.zeros(batch_size, 16)}
        shapes = dict.fromkeys(
            action_keys, (batch_size, prediction_horizon, prediction_dimension)
        )
        flat_dims = dict.fromkeys(
            action_keys, prediction_horizon * prediction_dimension
        )
        return VelocityWrapper(
            network=network,
            features=features,
            action_keys=action_keys,
            flat_dimensions=flat_dims,
            tensor_shapes=shapes,
            reverse_convention=reverse_convention,
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
    @pytest.mark.parametrize(
        "ode_solver",
        [
            ODESolver.EULER.value,
            ODESolver.HEUN.value,
        ],
    )
    @pytest.mark.parametrize("sigma", [0.0, 0.1])
    @pytest.mark.parametrize(
        "timestep_sampler",
        [
            TimestepSampler.BETA.value,
            TimestepSampler.UNIFORM.value,
        ],
    )
    @pytest.mark.parametrize("logit_mean", [0.0, 0.5])
    @pytest.mark.parametrize("logit_std", [0.5, 1.0])
    @pytest.mark.parametrize("beta_alpha", [1.0, 1.5])
    @pytest.mark.parametrize("beta_beta", [0.5, 1.0])
    @pytest.mark.parametrize("max_timestep", [0.99, 0.999])
    @pytest.mark.parametrize("reverse_flow_convention", [True, False])
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
        reverse_flow_convention: bool,
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
            reverse_flow_convention=reverse_flow_convention,
        )
        assert fm.num_inference_steps == num_inference_steps
        assert fm.ode_solver == ode_solver
        assert fm.timestep_sampler == timestep_sampler
        assert fm.logit_mean == logit_mean
        assert fm.logit_std == logit_std
        assert fm.beta_alpha == beta_alpha
        assert fm.beta_beta == beta_beta
        assert fm.max_timestep == max_timestep
        assert fm.reverse_flow_convention == reverse_flow_convention
        assert isinstance(fm.flow_matcher, ConditionalFlowMatcher)

    @pytest.mark.parametrize(
        "ode_solver, expectation",
        [
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
        ],
    )
    def test_ode_solver_validation(
        self,
        ode_solver: str,
        expectation,
    ):
        with expectation:
            FlowMatching(ode_solver=ode_solver)

    @pytest.mark.parametrize(
        "timestep_sampler, expectation",
        [
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
        ],
    )
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
        assert set(result[DecoderOutputKey.TARGET_VELOCITY.value].keys()) == {
            "position_action"
        }
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
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        mock_network.return_value = {"position_action": torch.zeros(2, 8, 3)}
        fm.forward(network=mock_network, features=features, actions=actions)
        features_passed = mock_network.call_args.kwargs["features"]
        assert DecoderOutputKey.TIMESTEP.value in features_passed

    def test_forward_with_multiple_action_keys(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        fm = flow_matching_factory()
        action_keys = ["gripper_action", "position_action"]
        mock_network = mock_action_decoder_factory(
            action_keys=action_keys,
            prediction_dimension=3,
        )
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=action_keys,
            prediction_horizon=8,
            action_dimension=3,
            include_padding_mask=True,
        )
        mock_network.return_value = {key: torch.zeros(2, 8, 3) for key in action_keys}
        result = fm.forward(network=mock_network, features=features, actions=actions)
        target_velocity = result[DecoderOutputKey.TARGET_VELOCITY.value]
        noise = result[DecoderOutputKey.NOISE.value]
        for key in action_keys:
            assert key in target_velocity
            assert key in noise


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

    def test_predict_enables_and_disables_encoder_cache(
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
            fm.predict(network=mock_network, features=features)
        mock_network.enable_encoder_cache.assert_called_once()
        mock_network.disable_encoder_cache.assert_called_once()

    def test_predict_with_multiple_action_keys(
        self,
        flow_matching_factory: Callable[..., FlowMatching],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        fm = flow_matching_factory(num_inference_steps=2)
        action_keys = ["gripper_action", "position_action"]
        mock_network = mock_action_decoder_factory(
            action_keys=action_keys,
            prediction_dimension=3,
        )
        features = feature_dictionary_factory()
        # Total flat dim: 2 keys * 8 horizon * 3 dim = 48
        with patch(
            "versatil.models.decoding.algorithm.flow_matching.integrate_ode",
        ) as mock_integrate:
            mock_integrate.return_value = torch.zeros(2, 8 * 3 * 2)
            result = fm.predict(network=mock_network, features=features)
        assert set(result.keys()) == set(action_keys)
        for key in action_keys:
            assert result[key].shape == (2, 8, 3)


class TestVelocityWrapper:
    def test_forward_convention_passes_time_unchanged(
        self,
        velocity_wrapper_factory: Callable[..., VelocityWrapper],
    ):
        wrapper = velocity_wrapper_factory(reverse_convention=False)
        state = torch.zeros(2, 8 * 3)
        time = torch.tensor([0.3, 0.3])
        wrapper(state, time)
        call_kwargs = wrapper.network.call_args.kwargs
        passed_time = call_kwargs["features"][DecoderOutputKey.TIMESTEP.value]
        assert torch.allclose(passed_time, time)

    def test_reverse_convention_flips_timestep(
        self,
        velocity_wrapper_factory: Callable[..., VelocityWrapper],
    ):
        wrapper = velocity_wrapper_factory(reverse_convention=True)
        state = torch.zeros(2, 8 * 3)
        time = torch.tensor([0.3, 0.3])
        wrapper(state, time)
        call_kwargs = wrapper.network.call_args.kwargs
        passed_time = call_kwargs["features"][DecoderOutputKey.TIMESTEP.value]
        assert torch.allclose(passed_time, 1.0 - time)

    def test_reverse_convention_negates_velocity(
        self,
        velocity_wrapper_factory: Callable[..., VelocityWrapper],
    ):
        wrapper_fwd = velocity_wrapper_factory(reverse_convention=False)
        wrapper_rev = velocity_wrapper_factory(reverse_convention=True)
        state = torch.zeros(2, 8 * 3)
        time = torch.tensor([0.5, 0.5])
        velocity_fwd = wrapper_fwd(state, time)
        velocity_rev = wrapper_rev(state, time)
        assert torch.allclose(velocity_fwd, -velocity_rev)

    def test_reshapes_flat_state_to_per_key_dicts(
        self,
        velocity_wrapper_factory: Callable[..., VelocityWrapper],
    ):
        action_keys = ["gripper_action", "position_action"]
        wrapper = velocity_wrapper_factory(
            action_keys=action_keys,
            prediction_dimension=3,
        )
        state = torch.zeros(2, 2 * 8 * 3)
        time = torch.tensor([0.5, 0.5])
        wrapper(state, time)
        call_kwargs = wrapper.network.call_args.kwargs
        actions_passed = call_kwargs["actions"]
        assert set(actions_passed.keys()) == set(action_keys)
        for key in action_keys:
            assert actions_passed[key].shape == (2, 8, 3)
