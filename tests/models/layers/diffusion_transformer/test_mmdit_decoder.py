"""Tests for versatil.models.layers.diffusion_transformer.mmdit_decoder module."""

import math
from collections.abc import Callable

import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.diffusion_transformer.mmdit_decoder import MMDiTDecoder
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def mmdit_decoder_factory() -> Callable[..., MMDiTDecoder]:
    def factory(
        number_of_layers: int = 2,
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
        positional_encoding_type: str | None = None,
        maximum_sequence_length_observation: int = 64,
        maximum_sequence_length_action: int = 32,
        bias: bool = True,
        initializer_range: float = 0.02,
    ) -> MMDiTDecoder:
        return MMDiTDecoder(
            number_of_layers=number_of_layers,
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
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length_observation=maximum_sequence_length_observation,
            maximum_sequence_length_action=maximum_sequence_length_action,
            bias=bias,
            initializer_range=initializer_range,
        )

    return factory


class TestMMDiTDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    def test_stores_configuration(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
    ):
        decoder = mmdit_decoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        assert decoder.number_of_layers == number_of_layers
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_heads == number_of_heads

    def test_correct_number_of_layers_created(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
    ):
        number_of_layers = 3
        decoder = mmdit_decoder_factory(number_of_layers=number_of_layers)
        assert len(decoder.layers) == number_of_layers

    def test_no_positional_encoding_excludes_position_parameters(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
    ):
        decoder_without = mmdit_decoder_factory(positional_encoding_type=None)
        decoder_with = mmdit_decoder_factory(
            positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
        )
        params_without = sum(p.numel() for p in decoder_without.parameters())
        params_with = sum(p.numel() for p in decoder_with.parameters())
        # With positional encoding, there should be more parameters
        assert params_without < params_with

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [
            PositionalEncodingType.SINUSOIDAL.value,
            PositionalEncodingType.LEARNED.value,
        ],
    )
    def test_additive_positional_encoding_changes_output(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str,
    ):
        embedding_dimension = 32
        decoder_no_pos = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            positional_encoding_type=None,
            use_gating=False,
        )
        decoder_with_pos = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            positional_encoding_type=positional_encoding_type,
            use_gating=False,
        )
        # Copy shared weights so only positional encoding differs
        decoder_with_pos.layers.load_state_dict(decoder_no_pos.layers.state_dict())
        decoder_with_pos.final_normalization_observation.load_state_dict(
            decoder_no_pos.final_normalization_observation.state_dict()
        )
        decoder_with_pos.final_normalization_action.load_state_dict(
            decoder_no_pos.final_normalization_action.state_dict()
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
        _, output_no_pos = decoder_no_pos(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        _, output_with_pos = decoder_with_pos(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_no_pos, output_with_pos)

    def test_rope_positional_encoding_changes_output(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder_no_pos = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            positional_encoding_type=None,
            use_gating=False,
        )
        decoder_rope = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
            use_gating=False,
        )
        decoder_rope.layers.load_state_dict(decoder_no_pos.layers.state_dict())
        decoder_rope.final_normalization_observation.load_state_dict(
            decoder_no_pos.final_normalization_observation.state_dict()
        )
        decoder_rope.final_normalization_action.load_state_dict(
            decoder_no_pos.final_normalization_action.state_dict()
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
        _, output_no_pos = decoder_no_pos(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        _, output_rope = decoder_rope(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_no_pos, output_rope)

    def test_final_normalization_streams_have_independent_weights(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
    ):
        decoder = mmdit_decoder_factory()
        # Mutate observation normalization and verify action normalization is unaffected
        original_action_weight = decoder.final_normalization_action.weight.data.clone()
        decoder.final_normalization_observation.weight.data.fill_(999.0)
        assert torch.allclose(
            decoder.final_normalization_action.weight.data, original_action_weight
        )

    def test_weight_initialization_linear_layers(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
    ):
        initializer_range = 0.02
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
            initializer_range=initializer_range,
        )
        first_layer = decoder.layers[0]
        weight = first_layer.joint_attention.query_projection_primary.weight
        measured_std = weight.std().item()
        assert abs(measured_std - initializer_range) < 0.01

    def test_weight_initialization_sqrt_layers_have_smaller_std(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
    ):
        number_of_layers = 2
        initializer_range = 0.02
        decoder = mmdit_decoder_factory(
            number_of_layers=number_of_layers,
            initializer_range=initializer_range,
        )
        expected_sqrt_std = initializer_range / math.sqrt(3 * number_of_layers)
        first_layer = decoder.layers[0]
        sqrt_weight = first_layer.joint_attention.output_projection_primary.weight
        measured_std = sqrt_weight.std().item()
        assert measured_std < initializer_range
        assert abs(measured_std - expected_sqrt_std) < 0.01

    def test_bias_initialized_to_zero(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
    ):
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
            bias=True,
        )
        first_layer = decoder.layers[0]
        bias = first_layer.joint_attention.query_projection_primary.bias
        assert torch.allclose(bias, torch.zeros_like(bias))

    @pytest.mark.parametrize(
        "normalization_type",
        [
            NormalizationType.RMS_NORM.value,
            NormalizationType.LAYER_NORM.value,
        ],
    )
    def test_normalization_type_accepted(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        normalization_type: str,
    ):
        mmdit_decoder_factory(
            number_of_layers=1,
            normalization_type=normalization_type,
        )


class TestMMDiTDecoderForward:
    @pytest.mark.parametrize(
        "batch_size, observation_length, action_length, embedding_dimension",
        [
            (2, 6, 4, 32),
            (1, 10, 8, 64),
        ],
    )
    def test_output_shapes_match_inputs(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
        observation_length: int,
        action_length: int,
        embedding_dimension: int,
    ):
        decoder = mmdit_decoder_factory(
            number_of_layers=2,
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
        output_obs, output_act = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert output_obs.shape == (batch_size, observation_length, embedding_dimension)
        assert output_act.shape == (batch_size, action_length, embedding_dimension)

    def test_different_conditioning_produces_different_outputs(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=False,
        )
        reinit_modulation_layers(decoder)
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
        output_obs_a, output_act_a = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning_a,
        )
        output_obs_b, output_act_b = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(output_obs_a, output_obs_b)
        assert not torch.allclose(output_act_a, output_act_b)

    def test_stacked_layers_produce_different_output_than_single_layer(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder_one_layer = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            use_gating=False,
        )
        decoder_two_layers = mmdit_decoder_factory(
            number_of_layers=2,
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
        _, output_act_one = decoder_one_layer(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        _, output_act_two = decoder_two_layers(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_act_one, output_act_two)

    def test_dual_stream_bidirectional_information_flow(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
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
        output_obs_original, output_act_original = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        # Modifying observation should affect action output
        modified_observation = observation + 10.0
        _, output_act_modified = decoder(
            hidden_states_observation=modified_observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_act_original, output_act_modified)
        # Modifying action should affect observation output
        modified_action = action + 10.0
        output_obs_modified, _ = decoder(
            hidden_states_observation=observation,
            hidden_states_action=modified_action,
            conditioning=conditioning,
        )
        assert not torch.allclose(output_obs_original, output_obs_modified)

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [
            None,
            PositionalEncodingType.SINUSOIDAL.value,
            PositionalEncodingType.ROPE.value,
        ],
    )
    def test_positional_encoding_path_produces_valid_output(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str | None,
    ):
        embedding_dimension = 32
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            positional_encoding_type=positional_encoding_type,
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
        output_obs, output_act = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        assert torch.all(torch.isfinite(output_obs))
        assert torch.all(torch.isfinite(output_act))

    def test_final_normalization_is_applied(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
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
        output_obs, output_act = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        # RMSNorm produces outputs with controlled magnitude; the RMS of
        # each feature vector should be approximately 1.0
        rms_obs = torch.sqrt(torch.mean(output_obs**2, dim=-1))
        assert torch.allclose(rms_obs, torch.ones_like(rms_obs), atol=0.2)

    def test_padding_masks_affect_output(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = mmdit_decoder_factory(
            number_of_layers=1,
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
        _, output_no_mask = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        observation_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=2,
        )
        _, output_with_mask = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
            attention_mask_observation=observation_mask,
        )
        assert not torch.allclose(output_no_mask, output_with_mask)

    def test_gradient_flows_through_all_components(
        self,
        mmdit_decoder_factory: Callable[..., MMDiTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        decoder = mmdit_decoder_factory(
            number_of_layers=2,
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
        conditioning.requires_grad_(True)
        output_obs, output_act = decoder(
            hidden_states_observation=observation,
            hidden_states_action=action,
            conditioning=conditioning,
        )
        loss = output_obs.sum() + output_act.sum()
        loss.backward()
        assert observation.grad is not None
        assert action.grad is not None
        assert conditioning.grad is not None
        assert torch.all(torch.isfinite(observation.grad))
        assert torch.all(torch.isfinite(action.grad))
        assert torch.all(torch.isfinite(conditioning.grad))
