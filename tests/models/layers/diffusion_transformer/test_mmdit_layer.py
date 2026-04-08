"""Tests for versatil.models.layers.diffusion_transformer.mmdit_layer module."""

from collections.abc import Callable

import pytest
import torch

from tests.models.layers.diffusion_transformer.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.diffusion_transformer.mmdit_layer import MMDiTLayer
from versatil.models.layers.gated_linear_unit import SwiGLU
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def mmdit_layer_factory() -> Callable[..., MMDiTLayer]:
    def factory(
        embedding_dimension: int = 32,
        conditioning_dimension: int = 32,
        number_of_heads: int = 4,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SILU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        normalization_epsilon: float = 1e-6,
        use_query_key_norm: bool = True,
        use_gating: bool = True,
        bias: bool = True,
    ) -> MMDiTLayer:
        return MMDiTLayer(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=conditioning_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            normalization_epsilon=normalization_epsilon,
            use_query_key_norm=use_query_key_norm,
            use_gating=use_gating,
            bias=bias,
        )

    return factory


class TestMMDiTLayerInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("use_gating", [True, False])
    @pytest.mark.parametrize("use_query_key_norm", [True, False])
    def test_stores_configuration(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        embedding_dimension: int,
        use_gating: bool,
        use_query_key_norm: bool,
    ):
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            use_gating=use_gating,
            use_query_key_norm=use_query_key_norm,
        )
        assert layer.embedding_dimension == embedding_dimension
        assert layer.use_gating == use_gating

    def test_feedforward_last_layer_flagged_for_sqrt_init(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
    ):
        layer = mmdit_layer_factory()
        assert layer.feedforward_observation[-1].SQUARE_ROOT_WEIGHT is True
        assert layer.feedforward_action[-1].SQUARE_ROOT_WEIGHT is True

    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.SILU.value, ActivationFunction.SWIGLU.value],
    )
    def test_feedforward_activation_selection(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        activation: str,
    ):
        layer = mmdit_layer_factory(activation=activation)
        if activation == ActivationFunction.SWIGLU.value:
            assert type(layer.feedforward_observation[0]) is SwiGLU
            assert type(layer.feedforward_action[0]) is SwiGLU
        else:
            assert type(layer.feedforward_observation[0]) is torch.nn.Linear
            assert type(layer.feedforward_action[0]) is torch.nn.Linear

    def test_default_feedforward_dimension_is_four_times_embedding(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            feedforward_dimension=None,
            activation=ActivationFunction.SILU.value,
        )
        # With non-SwiGLU activation, the first linear maps to feedforward_dimension
        expected_feedforward_dimension = 4 * embedding_dimension
        assert (
            layer.feedforward_observation[0].out_features
            == expected_feedforward_dimension
        )

    def test_explicit_feedforward_dimension_overrides_default(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
    ):
        layer = mmdit_layer_factory(
            embedding_dimension=32,
            feedforward_dimension=64,
            activation=ActivationFunction.SILU.value,
        )
        assert layer.feedforward_observation[0].out_features == 64


class TestMMDiTLayerForward:
    @pytest.mark.parametrize(
        "batch_size, observation_length, action_length, embedding_dimension",
        [
            (2, 6, 4, 32),
            (1, 8, 4, 64),
        ],
    )
    def test_output_shapes_match_inputs(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        observation_length: int,
        action_length: int,
        embedding_dimension: int,
    ):
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            number_of_heads=4,
        )
        observation = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=observation_length,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=action_length,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=embedding_dimension,
        )
        output_obs, output_act = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert output_obs.shape == (batch_size, observation_length, embedding_dimension)
        assert output_act.shape == (batch_size, action_length, embedding_dimension)

    def test_different_conditioning_produces_different_outputs(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(layer)
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning_a = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        conditioning_b = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_obs_a, output_act_a = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning_a,
        )
        output_obs_b, output_act_b = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(output_obs_a, output_obs_b)
        assert not torch.allclose(output_act_a, output_act_b)

    def test_dual_stream_interaction_observation_affects_action(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=False,
        )
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        _, output_act_original = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        modified_observation = observation + 10.0
        _, output_act_modified = layer(
            hidden_states_observation=modified_observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_act_original, output_act_modified)

    def test_dual_stream_interaction_action_affects_observation(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=False,
        )
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_obs_original, _ = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        modified_action = action + 10.0
        output_obs_modified, _ = layer(
            hidden_states_observation=observation,
            hidden_states_action=modified_action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_obs_original, output_obs_modified)

    @pytest.mark.parametrize("use_gating", [True, False])
    def test_gating_path_produces_valid_output(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_gating: bool,
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=use_gating,
        )
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_obs, output_act = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert torch.all(torch.isfinite(output_obs))
        assert torch.all(torch.isfinite(output_act))

    def test_adaln_zero_output_equals_input_at_initialization(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        # AdaLN-Zero initializes gates to zero, so the gated residual path
        # contributes nothing: output = input + 0 * f(input) = input
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=True,
        )
        layer.eval()
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        output_obs, output_act = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert torch.allclose(output_obs, observation, atol=1e-6)
        assert torch.allclose(output_act, action, atol=1e-6)

    def test_gradient_flows_through_both_streams(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
        )
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        observation.requires_grad_(True)
        action.requires_grad_(True)
        output_obs, output_act = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        loss = output_obs.sum() + output_act.sum()
        loss.backward()
        assert observation.grad is not None
        assert action.grad is not None
        assert torch.all(torch.isfinite(observation.grad))
        assert torch.all(torch.isfinite(action.grad))

    def test_gradient_flows_through_conditioning(
        self,
        mmdit_layer_factory: Callable[..., MMDiTLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        layer = mmdit_layer_factory(
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=False,
        )
        observation = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=embedding_dimension,
        )
        action = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        conditioning = condition_factory(
            batch_size=2,
            condition_dim=embedding_dimension,
        )
        conditioning.requires_grad_(True)
        output_obs, output_act = layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        loss = output_obs.sum() + output_act.sum()
        loss.backward()
        assert conditioning.grad is not None
        assert torch.all(torch.isfinite(conditioning.grad))
