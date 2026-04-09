"""Tests for versatil.models.layers.transformer.attention.joint_attention_base module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.transformer.attention.joint_attention_base import (
    JointAttentionBase,
)

NUMBER_OF_HEADS = 4
HEAD_DIMENSION = 8


@pytest.fixture
def base_attention() -> JointAttentionBase:
    return JointAttentionBase(
        number_of_heads=NUMBER_OF_HEADS,
        number_of_key_value_heads=NUMBER_OF_HEADS,
        head_dimension=HEAD_DIMENSION,
    )


class TestReshapeMethods:
    def test_reshape_for_query(
        self,
        base_attention: JointAttentionBase,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        flat = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=NUMBER_OF_HEADS * HEAD_DIMENSION,
        )
        reshaped = base_attention._reshape_for_query(flat)
        assert reshaped.shape == (2, NUMBER_OF_HEADS, 6, HEAD_DIMENSION)

    @pytest.mark.parametrize(
        "number_of_kv_heads, expected_heads",
        [(4, 4), (2, 2)],
        ids=["mha", "gqa"],
    )
    def test_reshape_for_key_value(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        number_of_kv_heads: int,
        expected_heads: int,
    ):
        attention = JointAttentionBase(
            number_of_heads=NUMBER_OF_HEADS,
            number_of_key_value_heads=number_of_kv_heads,
            head_dimension=HEAD_DIMENSION,
        )
        flat_dim = number_of_kv_heads * HEAD_DIMENSION
        flat = sequence_tensor_factory(
            batch_size=2,
            sequence_length=6,
            embedding_dimension=flat_dim,
        )
        reshaped = attention._reshape_for_key_value(flat)
        assert reshaped.shape == (2, expected_heads, 6, HEAD_DIMENSION)


class TestJointSdpa:
    def test_output_shapes(
        self,
        base_attention: JointAttentionBase,
        head_split_tensor_factory: Callable[..., torch.Tensor],
    ):
        query_primary = head_split_tensor_factory(sequence_length=6)
        key_primary = head_split_tensor_factory(sequence_length=6)
        value_primary = head_split_tensor_factory(sequence_length=6)
        query_secondary = head_split_tensor_factory(sequence_length=4)
        key_secondary = head_split_tensor_factory(sequence_length=4)
        value_secondary = head_split_tensor_factory(sequence_length=4)
        output_primary, output_secondary = base_attention._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary,
            value_secondary=value_secondary,
            sequence_length_primary=6,
            sequence_length_secondary=4,
        )
        query_dimension = NUMBER_OF_HEADS * HEAD_DIMENSION
        assert output_primary.shape == (2, 6, query_dimension)
        assert output_secondary.shape == (2, 4, query_dimension)

    def test_modifying_secondary_kv_changes_primary_output(
        self,
        base_attention: JointAttentionBase,
        head_split_tensor_factory: Callable[..., torch.Tensor],
    ):
        base_attention.eval()
        query_primary = head_split_tensor_factory(sequence_length=6)
        key_primary = head_split_tensor_factory(sequence_length=6)
        value_primary = head_split_tensor_factory(sequence_length=6)
        query_secondary = head_split_tensor_factory(sequence_length=4)
        key_secondary_a = head_split_tensor_factory(sequence_length=4)
        value_secondary_a = head_split_tensor_factory(sequence_length=4)
        key_secondary_b = head_split_tensor_factory(sequence_length=4)
        value_secondary_b = head_split_tensor_factory(sequence_length=4)
        output_a, _ = base_attention._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary_a,
            value_secondary=value_secondary_a,
            sequence_length_primary=6,
            sequence_length_secondary=4,
        )
        output_b, _ = base_attention._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary_b,
            value_secondary=value_secondary_b,
            sequence_length_primary=6,
            sequence_length_secondary=4,
        )
        # Primary attends to joint K/V (includes secondary), so changing secondary K/V
        # should change primary output
        assert not torch.allclose(output_a, output_b)

    def test_gqa_expansion(
        self,
        head_split_tensor_factory: Callable[..., torch.Tensor],
    ):
        attention = JointAttentionBase(
            number_of_heads=4,
            number_of_key_value_heads=2,
            head_dimension=HEAD_DIMENSION,
        )
        query_primary = head_split_tensor_factory(number_of_heads=4, sequence_length=6)
        key_primary = head_split_tensor_factory(number_of_heads=2, sequence_length=6)
        value_primary = head_split_tensor_factory(number_of_heads=2, sequence_length=6)
        query_secondary = head_split_tensor_factory(
            number_of_heads=4, sequence_length=4
        )
        key_secondary = head_split_tensor_factory(number_of_heads=2, sequence_length=4)
        value_secondary = head_split_tensor_factory(
            number_of_heads=2, sequence_length=4
        )
        output_primary, output_secondary = attention._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary,
            value_secondary=value_secondary,
            sequence_length_primary=6,
            sequence_length_secondary=4,
        )
        query_dimension = 4 * HEAD_DIMENSION
        assert output_primary.shape == (2, 6, query_dimension)
        assert output_secondary.shape == (2, 4, query_dimension)

    def test_per_stream_mask_changes_output(
        self,
        base_attention: JointAttentionBase,
        head_split_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        base_attention.eval()
        query_primary = head_split_tensor_factory(sequence_length=6)
        key_primary = head_split_tensor_factory(sequence_length=6)
        value_primary = head_split_tensor_factory(sequence_length=6)
        query_secondary = head_split_tensor_factory(sequence_length=4)
        key_secondary = head_split_tensor_factory(sequence_length=4)
        value_secondary = head_split_tensor_factory(sequence_length=4)
        mask = padding_mask_factory(batch_size=2, sequence_length=6, mask_last_n=2)
        output_masked, _ = base_attention._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary,
            value_secondary=value_secondary,
            sequence_length_primary=6,
            sequence_length_secondary=4,
            attention_mask_primary=mask,
        )
        output_unmasked, _ = base_attention._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary,
            value_secondary=value_secondary,
            sequence_length_primary=6,
            sequence_length_secondary=4,
        )
        assert not torch.allclose(output_masked, output_unmasked)


class TestBuildJointAttentionMask:
    def test_no_masks_returns_none(self, base_attention: JointAttentionBase):
        result = base_attention._build_joint_attention_mask(
            mask_primary=None,
            mask_secondary=None,
            sequence_length_primary=6,
            sequence_length_secondary=4,
            device=torch.device("cpu"),
        )
        assert result is None

    @pytest.mark.parametrize(
        "mask_primary_last_n, mask_secondary_last_n",
        [(2, None), (None, 1), (1, 2)],
        ids=["primary_only", "secondary_only", "both"],
    )
    def test_mask_combination_produces_correct_joint_shape(
        self,
        base_attention: JointAttentionBase,
        padding_mask_factory: Callable[..., torch.Tensor],
        mask_primary_last_n: int | None,
        mask_secondary_last_n: int | None,
    ):
        mask_primary = (
            padding_mask_factory(
                batch_size=2, sequence_length=6, mask_last_n=mask_primary_last_n
            )
            if mask_primary_last_n is not None
            else None
        )
        mask_secondary = (
            padding_mask_factory(
                batch_size=2, sequence_length=4, mask_last_n=mask_secondary_last_n
            )
            if mask_secondary_last_n is not None
            else None
        )
        result = base_attention._build_joint_attention_mask(
            mask_primary=mask_primary,
            mask_secondary=mask_secondary,
            sequence_length_primary=6,
            sequence_length_secondary=4,
            device=torch.device("cpu"),
        )
        assert result.shape == (2, 1, 1, 10)  # S+T = 6+4

    def test_mask_values_preserve_per_stream_masking(
        self,
        base_attention: JointAttentionBase,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        mask_primary = padding_mask_factory(
            batch_size=2, sequence_length=6, mask_last_n=1
        )
        mask_secondary = padding_mask_factory(
            batch_size=2, sequence_length=4, mask_last_n=2
        )
        result = base_attention._build_joint_attention_mask(
            mask_primary=mask_primary,
            mask_secondary=mask_secondary,
            sequence_length_primary=6,
            sequence_length_secondary=4,
            device=torch.device("cpu"),
        )
        # Primary: 5 unmasked + 1 masked, Secondary: 2 unmasked + 2 masked
        # Joint: [F,F,F,F,F,T, F,F,T,T]
        assert not result[0, 0, 0, 0].item()
        assert result[0, 0, 0, 5].item() is True
        assert not result[0, 0, 0, 6].item()
        assert result[0, 0, 0, 8].item() is True
        assert result[0, 0, 0, 9].item() is True
