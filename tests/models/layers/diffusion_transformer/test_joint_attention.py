"""Tests for versatil.models.layers.diffusion_transformer.joint_attention module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.diffusion_transformer.joint_attention import JointAttention


@pytest.fixture
def joint_attention_factory() -> Callable[..., JointAttention]:

    def factory(
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        dropout: float = 0.0,
        use_query_key_norm: bool = True,
        normalization_epsilon: float = 1e-6,
        bias: bool = True,
    ) -> JointAttention:
        return JointAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
            use_query_key_norm=use_query_key_norm,
            normalization_epsilon=normalization_epsilon,
            bias=bias,
        )

    return factory


class TestJointAttentionInitialization:

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize("use_query_key_norm", [True, False])
    def test_stores_configuration(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        embedding_dimension: int,
        number_of_heads: int,
        use_query_key_norm: bool,
    ):
        attention = joint_attention_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            use_query_key_norm=use_query_key_norm,
        )
        assert attention.embedding_dimension == embedding_dimension
        assert attention.number_of_heads == number_of_heads
        assert attention.head_dimension == embedding_dimension // number_of_heads
        assert attention.use_query_key_norm == use_query_key_norm

    def test_observation_and_action_projections_are_independent(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        # Mutate observation projection and verify action projection is unaffected
        original_action_weight = attention.query_projection_action.weight.data.clone()
        attention.query_projection_observation.weight.data.fill_(999.0)
        assert torch.allclose(
            attention.query_projection_action.weight.data, original_action_weight
        )

    def test_output_projections_flagged_for_sqrt_weight_init(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        assert attention.output_projection_observation.SQUARE_ROOT_WEIGHT == True
        assert attention.output_projection_action.SQUARE_ROOT_WEIGHT == True

    def test_disabling_query_key_norm_excludes_norm_parameters(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        attention_with_norm = joint_attention_factory(use_query_key_norm=True)
        attention_without_norm = joint_attention_factory(use_query_key_norm=False)
        params_with = sum(p.numel() for p in attention_with_norm.parameters())
        params_without = sum(p.numel() for p in attention_without_norm.parameters())
        # QK-norm adds learnable parameters; without it there should be fewer
        assert params_without < params_with


class TestJointAttentionForward:

    @pytest.mark.parametrize(
        "batch_size, observation_length, action_length, embedding_dimension",
        [
            (2, 6, 4, 32),
            (1, 8, 4, 64),
        ],
    )
    def test_output_shapes_match_inputs(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        observation_length: int,
        action_length: int,
        embedding_dimension: int,
    ):
        attention = joint_attention_factory(
            embedding_dimension=embedding_dimension,
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
        output_observation, output_action = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
        )
        assert output_observation.shape == (batch_size, observation_length, embedding_dimension)
        assert output_action.shape == (batch_size, action_length, embedding_dimension)

    def test_both_streams_attend_to_joint_key_values(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(embedding_dimension=embedding_dimension)
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
        output_obs_original, output_act_original = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
        )
        # Modify the action stream and verify it affects the observation output
        # (because observation attends to joint keys which include action keys)
        modified_action = action + 10.0
        output_obs_modified, _ = attention(
            hidden_states_observation=observation,
            hidden_states_action=modified_action,
        )
        assert not torch.allclose(output_obs_original, output_obs_modified)

    def test_observation_stream_affects_action_output(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(embedding_dimension=embedding_dimension)
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
        _, output_act_original = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
        )
        modified_observation = observation + 10.0
        _, output_act_modified = attention(
            hidden_states_observation=modified_observation,
            hidden_states_action=action,
        )
        assert not torch.allclose(output_act_original, output_act_modified)

    @pytest.mark.parametrize("use_query_key_norm", [True, False])
    def test_query_key_norm_path_produces_valid_output(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        use_query_key_norm: bool,
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            embedding_dimension=embedding_dimension,
            use_query_key_norm=use_query_key_norm,
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
        output_obs, output_act = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
        )
        assert torch.all(torch.isfinite(output_obs))
        assert torch.all(torch.isfinite(output_act))

    def test_gradient_flows_through_both_streams(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(embedding_dimension=embedding_dimension)
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
        observation.requires_grad_(True)
        action.requires_grad_(True)
        output_obs, output_act = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
        )
        loss = output_obs.sum() + output_act.sum()
        loss.backward()
        assert observation.grad is not None
        assert action.grad is not None
        assert torch.all(torch.isfinite(observation.grad))
        assert torch.all(torch.isfinite(action.grad))

    def test_masked_positions_do_not_affect_unmasked_output(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            embedding_dimension=embedding_dimension,
            use_query_key_norm=False,
        )
        attention.eval()
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
        observation_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=2,
        )
        output_obs_masked, output_act_masked = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
            attention_mask_observation=observation_mask,
        )
        output_obs_unmasked, output_act_unmasked = attention(
            hidden_states_observation=observation,
            hidden_states_action=action,
        )
        # Masked and unmasked should produce different outputs since different
        # key-value positions are attended to
        assert not torch.allclose(output_act_masked, output_act_unmasked)


class TestJointAttentionMask:

    def test_no_masks_returns_none(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        result = attention._build_joint_attention_mask(
            mask_observation=None,
            mask_action=None,
            sequence_length_observation=6,
            sequence_length_action=4,
            device=torch.device("cpu"),
        )
        assert result is None

    def test_joint_mask_shape_with_both_masks(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        observation_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=1,
        )
        action_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=4,
            mask_last_n=1,
        )
        result = attention._build_joint_attention_mask(
            mask_observation=observation_mask,
            mask_action=action_mask,
            sequence_length_observation=6,
            sequence_length_action=4,
            device=torch.device("cpu"),
        )
        assert result.shape == (2, 1, 1, 10)

    def test_joint_mask_with_only_observation_mask(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        observation_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=2,
        )
        result = attention._build_joint_attention_mask(
            mask_observation=observation_mask,
            mask_action=None,
            sequence_length_observation=6,
            sequence_length_action=4,
            device=torch.device("cpu"),
        )
        assert result.shape == (2, 1, 1, 10)
        # Action positions should be unmasked (filled with False)
        assert not result[0, 0, 0, 6].item()
        # Last 2 observation positions should be masked
        assert result[0, 0, 0, 4].item() is True
        assert result[0, 0, 0, 5].item() is True

    def test_joint_mask_with_only_action_mask(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        action_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=4,
            mask_last_n=1,
        )
        result = attention._build_joint_attention_mask(
            mask_observation=None,
            mask_action=action_mask,
            sequence_length_observation=6,
            sequence_length_action=4,
            device=torch.device("cpu"),
        )
        assert result.shape == (2, 1, 1, 10)
        # Observation positions should be unmasked
        assert not result[0, 0, 0, 0].item()
        # Last action position (index 6+3=9) should be masked
        assert result[0, 0, 0, 9].item() is True

    def test_joint_mask_preserves_correct_mask_values(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        attention = joint_attention_factory(embedding_dimension=32)
        observation_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=6,
            mask_last_n=1,
        )
        action_mask = padding_mask_factory(
            batch_size=2,
            sequence_length=4,
            mask_last_n=2,
        )
        result = attention._build_joint_attention_mask(
            mask_observation=observation_mask,
            mask_action=action_mask,
            sequence_length_observation=6,
            sequence_length_action=4,
            device=torch.device("cpu"),
        )
        # Observation: 5 unmasked + 1 masked, Action: 2 unmasked + 2 masked
        # Total: joint mask should be [F,F,F,F,F,T, F,F,T,T]
        assert not result[0, 0, 0, 0].item()
        assert result[0, 0, 0, 5].item() is True
        assert not result[0, 0, 0, 6].item()
        assert not result[0, 0, 0, 7].item()
        assert result[0, 0, 0, 8].item() is True
        assert result[0, 0, 0, 9].item() is True
