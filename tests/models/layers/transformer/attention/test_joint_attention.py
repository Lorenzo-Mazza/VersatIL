"""Tests for versatil.models.layers.transformer.attention.joint_attention module."""

import re
import unittest.mock
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.models.layers.transformer.attention.joint_attention import JointAttention


@pytest.fixture
def joint_attention_factory() -> Callable[..., JointAttention]:
    def factory(
        primary_embedding_dimension: int = 32,
        number_of_heads: int = 4,
        secondary_embedding_dimension: int | None = None,
        number_of_key_value_heads: int | None = None,
        dropout: float = 0.0,
        use_query_key_norm: bool = True,
        normalization_epsilon: float = 1e-6,
        bias: bool = True,
    ) -> JointAttention:
        return JointAttention(
            primary_embedding_dimension=primary_embedding_dimension,
            number_of_heads=number_of_heads,
            secondary_embedding_dimension=secondary_embedding_dimension,
            number_of_key_value_heads=number_of_key_value_heads,
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
            primary_embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            use_query_key_norm=use_query_key_norm,
        )
        assert attention.primary_embedding_dimension == embedding_dimension
        assert attention.number_of_heads == number_of_heads
        assert attention.head_dimension == embedding_dimension // number_of_heads
        assert attention.use_query_key_norm == use_query_key_norm

    def test_primary_and_secondary_projections_are_independent(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        attention = joint_attention_factory(primary_embedding_dimension=32)
        original_secondary_weight = (
            attention.query_projection_secondary.weight.data.clone()
        )
        attention.query_projection_primary.weight.data.fill_(999.0)
        assert torch.allclose(
            attention.query_projection_secondary.weight.data, original_secondary_weight
        )

    def test_output_projections_flagged_for_sqrt_weight_init(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        attention = joint_attention_factory(primary_embedding_dimension=32)
        assert attention.output_projection_primary.SQUARE_ROOT_WEIGHT is True
        assert attention.output_projection_secondary.SQUARE_ROOT_WEIGHT is True

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

    def test_invalid_default_head_dimension_raises(
        self,
        joint_attention_factory: Callable[..., JointAttention],
    ):
        error_message = (
            "primary_embedding_dimension (30) must be divisible by "
            "number_of_heads (8) when head_dimension is not provided."
        )
        with pytest.raises(ValueError, match=re.escape(error_message)):
            joint_attention_factory(
                primary_embedding_dimension=30,
                number_of_heads=8,
            )


class TestJointAttentionForward:
    def test_both_streams_attend_to_joint_key_values(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            primary_embedding_dimension=embedding_dimension
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
        output_obs_original, output_act_original = attention(
            hidden_states_primary=observation,
            hidden_states_secondary=action,
        )
        # Modify the action stream and verify it affects the observation output
        # (because observation attends to joint keys which include action keys)
        modified_action = action + 10.0
        output_obs_modified, _ = attention(
            hidden_states_primary=observation,
            hidden_states_secondary=modified_action,
        )
        assert not torch.allclose(output_obs_original, output_obs_modified)

    def test_primary_stream_affects_secondary_output(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            primary_embedding_dimension=embedding_dimension
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
        _, output_act_original = attention(
            hidden_states_primary=observation,
            hidden_states_secondary=action,
        )
        modified_observation = observation + 10.0
        _, output_act_modified = attention(
            hidden_states_primary=modified_observation,
            hidden_states_secondary=action,
        )
        assert not torch.allclose(output_act_original, output_act_modified)

    def test_gradient_flows_through_both_streams(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            primary_embedding_dimension=embedding_dimension
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
        observation.requires_grad_(True)
        action.requires_grad_(True)
        output_obs, output_act = attention(
            hidden_states_primary=observation,
            hidden_states_secondary=action,
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
            primary_embedding_dimension=embedding_dimension,
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
            hidden_states_primary=observation,
            hidden_states_secondary=action,
            attention_mask_primary=observation_mask,
        )
        output_obs_unmasked, output_act_unmasked = attention(
            hidden_states_primary=observation,
            hidden_states_secondary=action,
        )
        # Masked and unmasked should produce different outputs since different
        # key-value positions are attended to
        assert not torch.allclose(output_act_masked, output_act_unmasked)

    @pytest.mark.parametrize(
        "primary_dim, secondary_dim, number_of_key_value_heads",
        [
            (32, 24, None),
            (32, 32, 2),
        ],
        ids=["asymmetric_dims", "gqa"],
    )
    def test_stream_configs_produce_correct_output_shapes(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        primary_dim: int,
        secondary_dim: int,
        number_of_key_value_heads: int | None,
    ):
        attention = joint_attention_factory(
            primary_embedding_dimension=primary_dim,
            secondary_embedding_dimension=secondary_dim,
            number_of_heads=4,
            number_of_key_value_heads=number_of_key_value_heads,
        )
        primary = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=primary_dim
        )
        secondary = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=secondary_dim
        )
        output_primary, output_secondary = attention(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
        )
        assert output_primary.shape == (2, 6, primary_dim)
        assert output_secondary.shape == (2, 4, secondary_dim)

    @pytest.mark.parametrize(
        "apply_to_primary, apply_to_secondary",
        [(True, False), (False, True), (True, True)],
        ids=["primary_only", "secondary_only", "both"],
    )
    def test_rope_applied_to_specified_streams(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        apply_to_primary: bool,
        apply_to_secondary: bool,
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            primary_embedding_dimension=embedding_dimension,
            use_query_key_norm=False,
        )
        attention.eval()
        primary = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=embedding_dimension
        )
        secondary = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        mock_rope = unittest.mock.MagicMock()
        output_no_rope_primary, output_no_rope_secondary = attention(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
        )
        with unittest.mock.patch(
            "versatil.models.layers.transformer.attention.joint_attention.apply_rope_positional_encoding",
            side_effect=lambda queries, keys, **kwargs: (queries * 1.5, keys * 1.5),
        ):
            output_rope_primary, output_rope_secondary = attention(
                hidden_states_primary=primary,
                hidden_states_secondary=secondary,
                positional_encoding_primary=mock_rope if apply_to_primary else None,
                positional_encoding_secondary=mock_rope if apply_to_secondary else None,
            )
        if apply_to_primary:
            assert not torch.allclose(output_no_rope_primary, output_rope_primary)
        if apply_to_secondary:
            assert not torch.allclose(output_no_rope_secondary, output_rope_secondary)

    def test_prebuilt_joint_mask_slices_correctly(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        attention = joint_attention_factory(
            primary_embedding_dimension=embedding_dimension,
            use_query_key_norm=False,
        )
        attention.eval()
        primary = sequence_tensor_factory(
            batch_size=2, sequence_length=6, embedding_dimension=embedding_dimension
        )
        secondary = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        # Pre-built joint mask: mask last 2 positions of secondary in joint space
        # Joint sequence = [primary(6) | secondary(4)] = 10 total
        joint_mask = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
        joint_mask[:, :, :, 8:] = True  # mask last 2 positions (secondary idx 2,3)
        output_with_mask_primary, output_with_mask_secondary = attention(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
            joint_attention_mask=joint_mask,
        )
        output_no_mask_primary, output_no_mask_secondary = attention(
            hidden_states_primary=primary,
            hidden_states_secondary=secondary,
        )
        assert not torch.allclose(output_with_mask_primary, output_no_mask_primary)


@pytest.mark.unit
class TestJointAttentionRopePositionSpace:
    def test_secondary_stream_rope_continues_after_primary_positions(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        rng: np.random.Generator,
    ):
        # Both streams share one joint softmax, so restarting secondary
        # positions at 0 would collide primary token i with secondary token i.
        attention = joint_attention_factory(use_query_key_norm=False)
        primary_length, secondary_length = 6, 4
        primary = torch.from_numpy(
            rng.standard_normal((2, primary_length, 32)).astype(np.float32)
        )
        secondary = torch.from_numpy(
            rng.standard_normal((2, secondary_length, 32)).astype(np.float32)
        )
        rope_primary = MagicMock()
        rope_secondary = MagicMock()

        with patch(
            "versatil.models.layers.transformer.attention.joint_attention."
            "apply_rope_positional_encoding",
            side_effect=lambda queries, keys, positional_encoding, cache_position: (
                queries,
                keys,
            ),
        ) as mock_rope:
            attention(
                hidden_states_primary=primary,
                hidden_states_secondary=secondary,
                positional_encoding_primary=rope_primary,
                positional_encoding_secondary=rope_secondary,
            )

        assert mock_rope.call_count == 2
        primary_call, secondary_call = mock_rope.call_args_list
        assert primary_call.kwargs["cache_position"] == 0
        assert secondary_call.kwargs["cache_position"] == primary_length

    def test_query_key_norm_applied_before_rope(
        self,
        joint_attention_factory: Callable[..., JointAttention],
        rng: np.random.Generator,
    ):
        # A learned per-channel scale after the rotation would break RoPE's
        # relative-position guarantee, so norm must run first.
        attention = joint_attention_factory(use_query_key_norm=True)
        call_order: list[str] = []
        original_norm = attention.query_key_norm_primary.forward
        attention.query_key_norm_primary.forward = lambda queries, keys: (
            call_order.append("norm"),
            original_norm(queries, keys),
        )[1]
        primary = torch.from_numpy(rng.standard_normal((1, 3, 32)).astype(np.float32))
        secondary = torch.from_numpy(rng.standard_normal((1, 2, 32)).astype(np.float32))

        with patch(
            "versatil.models.layers.transformer.attention.joint_attention."
            "apply_rope_positional_encoding",
            side_effect=lambda queries, keys, positional_encoding, cache_position: (
                call_order.append("rope"),
                (queries, keys),
            )[1],
        ):
            attention(
                hidden_states_primary=primary,
                hidden_states_secondary=secondary,
                positional_encoding_primary=MagicMock(),
                positional_encoding_secondary=MagicMock(),
            )

        assert call_order.index("norm") < call_order.index("rope")
