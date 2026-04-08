"""Tests for versatil.models.decoding.decoders.vla_interleaved module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from tests.models.layers.conftest import reinit_modulation_layers
from versatil.models.decoding.decoders.vla_interleaved import (
    VLACrossAttentionLayer,
    VLAJointAttentionLayer,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.constants import NormalizationType

VLM_HIDDEN_DIMENSION = 32
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 2
HEAD_DIMENSION = VLM_HIDDEN_DIMENSION // NUM_ATTENTION_HEADS
EXPERT_HIDDEN_SIZE = 16
EXPERT_FEEDFORWARD_DIMENSION = EXPERT_HIDDEN_SIZE * 4
BATCH_SIZE = 2
EXPERT_SEQUENCE_LENGTH = 4
VLM_SEQUENCE_LENGTH = 8


@pytest.fixture
def cross_attention_layer_factory() -> Callable[..., VLACrossAttentionLayer]:

    def factory(
        expert_embedding_dimension: int = EXPERT_HIDDEN_SIZE,
        vlm_key_value_dimension: int = NUM_KEY_VALUE_HEADS * HEAD_DIMENSION,
        expert_number_of_heads: int = NUM_ATTENTION_HEADS,
        expert_number_of_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        expert_head_dimension: int = HEAD_DIMENSION,
        expert_feedforward_dimension: int = EXPERT_FEEDFORWARD_DIMENSION,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        activation: str = ActivationFunction.SILU.value,
        dropout: float = 0.0,
    ) -> VLACrossAttentionLayer:
        return VLACrossAttentionLayer(
            expert_embedding_dimension=expert_embedding_dimension,
            vlm_key_value_dimension=vlm_key_value_dimension,
            expert_number_of_heads=expert_number_of_heads,
            expert_number_of_key_value_heads=expert_number_of_key_value_heads,
            expert_head_dimension=expert_head_dimension,
            expert_feedforward_dimension=expert_feedforward_dimension,
            normalization_type=normalization_type,
            activation=activation,
            dropout=dropout,
        )

    return factory


@pytest.fixture
def joint_attention_layer_factory() -> Callable[..., VLAJointAttentionLayer]:

    def factory(
        vlm_embedding_dimension: int = VLM_HIDDEN_DIMENSION,
        expert_embedding_dimension: int = EXPERT_HIDDEN_SIZE,
        number_of_heads: int = NUM_ATTENTION_HEADS,
        number_of_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        head_dimension: int = HEAD_DIMENSION,
        expert_feedforward_dimension: int = EXPERT_FEEDFORWARD_DIMENSION,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        condition_dim: int | None = None,
        use_gating: bool = False,
        activation: str = ActivationFunction.SILU.value,
        dropout: float = 0.0,
        bias: bool = False,
        use_query_key_norm: bool = False,
    ) -> VLAJointAttentionLayer:
        return VLAJointAttentionLayer(
            vlm_embedding_dimension=vlm_embedding_dimension,
            expert_embedding_dimension=expert_embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            expert_feedforward_dimension=expert_feedforward_dimension,
            normalization_type=normalization_type,
            condition_dim=condition_dim,
            use_gating=use_gating,
            activation=activation,
            dropout=dropout,
            bias=bias,
            use_query_key_norm=use_query_key_norm,
        )

    return factory


@pytest.fixture
def expert_hidden_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = EXPERT_SEQUENCE_LENGTH,
        embedding_dimension: int = EXPERT_HIDDEN_SIZE,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, embedding_dimension)
            ).astype(np.float32)
        )

    return factory


@pytest.fixture
def vlm_kv_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = VLM_SEQUENCE_LENGTH,
        kv_dimension: int = NUM_KEY_VALUE_HEADS * HEAD_DIMENSION,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, sequence_length, kv_dimension)
        keys = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        values = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        return keys, values

    return factory


@pytest.fixture
def precomputed_primary_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = VLM_SEQUENCE_LENGTH,
        number_of_heads: int = NUM_ATTENTION_HEADS,
        number_of_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        head_dimension: int = HEAD_DIMENSION,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_shape = (batch_size, number_of_heads, sequence_length, head_dimension)
        kv_shape = (
            batch_size,
            number_of_key_value_heads,
            sequence_length,
            head_dimension,
        )
        queries = torch.from_numpy(rng.standard_normal(query_shape).astype(np.float32))
        keys = torch.from_numpy(rng.standard_normal(kv_shape).astype(np.float32))
        values = torch.from_numpy(rng.standard_normal(kv_shape).astype(np.float32))
        return queries, keys, values

    return factory


@pytest.fixture
def condition_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:

    def factory(
        batch_size: int = BATCH_SIZE,
        condition_dim: int = EXPERT_HIDDEN_SIZE,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, condition_dim)).astype(np.float32)
        )

    return factory


class TestVLACrossAttentionLayer:
    @pytest.mark.parametrize("vlm_kv_dim", [NUM_KEY_VALUE_HEADS * HEAD_DIMENSION, 128])
    @pytest.mark.parametrize(
        "activation",
        [ActivationFunction.SILU.value, ActivationFunction.SWIGLU.value],
    )
    def test_forward_produces_valid_output(
        self,
        cross_attention_layer_factory: Callable[..., VLACrossAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        vlm_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        vlm_kv_dim: int,
        activation: str,
    ):
        layer = cross_attention_layer_factory(
            vlm_key_value_dimension=vlm_kv_dim,
            activation=activation,
        )
        expert_hidden = expert_hidden_factory()
        vlm_keys, vlm_values = vlm_kv_factory(kv_dimension=vlm_kv_dim)
        output = layer(
            expert_hidden_states=expert_hidden,
            vlm_key_states=vlm_keys,
            vlm_value_states=vlm_values,
        )
        assert output.shape == expert_hidden.shape
        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, expert_hidden)

    def test_different_vlm_kv_produce_different_outputs(
        self,
        cross_attention_layer_factory: Callable[..., VLACrossAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        vlm_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        layer = cross_attention_layer_factory()
        layer.eval()
        expert_hidden = expert_hidden_factory()
        keys_a, values_a = vlm_kv_factory()
        keys_b, values_b = vlm_kv_factory()
        output_a = layer(
            expert_hidden_states=expert_hidden,
            vlm_key_states=keys_a,
            vlm_value_states=values_a,
        )
        output_b = layer(
            expert_hidden_states=expert_hidden,
            vlm_key_states=keys_b,
            vlm_value_states=values_b,
        )
        assert not torch.allclose(output_a, output_b)


class TestVLAJointAttentionLayer:
    def test_forward_returns_both_streams(
        self,
        joint_attention_layer_factory: Callable[..., VLAJointAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        layer = joint_attention_layer_factory()
        expert_hidden = expert_hidden_factory()
        precomputed_primary = precomputed_primary_factory()
        vlm_output, expert_output = layer(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=expert_hidden,
        )
        # VLM output: (B, VLM_S, H * D_head)
        assert vlm_output.shape == (
            BATCH_SIZE,
            VLM_SEQUENCE_LENGTH,
            NUM_ATTENTION_HEADS * HEAD_DIMENSION,
        )
        assert expert_output.shape == expert_hidden.shape
        assert torch.all(torch.isfinite(vlm_output))
        assert torch.all(torch.isfinite(expert_output))

    def test_different_precomputed_primary_produces_different_outputs(
        self,
        joint_attention_layer_factory: Callable[..., VLAJointAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        layer = joint_attention_layer_factory()
        layer.eval()
        expert_hidden = expert_hidden_factory()
        primary_a = precomputed_primary_factory()
        primary_b = precomputed_primary_factory()
        _, expert_output_a = layer(
            precomputed_primary=primary_a,
            hidden_states_secondary=expert_hidden,
        )
        _, expert_output_b = layer(
            precomputed_primary=primary_b,
            hidden_states_secondary=expert_hidden,
        )
        assert not torch.allclose(expert_output_a, expert_output_b)

    def test_conditioning_sensitivity(
        self,
        joint_attention_layer_factory: Callable[..., VLAJointAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = joint_attention_layer_factory(
            normalization_type=NormalizationType.ADARMS.value,
            condition_dim=EXPERT_HIDDEN_SIZE,
        )
        reinit_modulation_layers(layer)
        expert_hidden = expert_hidden_factory()
        precomputed_primary = precomputed_primary_factory()
        conditioning_a = condition_factory()
        conditioning_b = condition_factory()
        _, expert_output_a = layer(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=expert_hidden,
            conditioning=conditioning_a,
        )
        _, expert_output_b = layer(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=expert_hidden,
            conditioning=conditioning_b,
        )
        assert not torch.allclose(expert_output_a, expert_output_b)

    def test_gated_conditioning_at_init_preserves_expert_input(
        self,
        joint_attention_layer_factory: Callable[..., VLAJointAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = joint_attention_layer_factory(
            normalization_type=NormalizationType.ADARMS.value,
            condition_dim=EXPERT_HIDDEN_SIZE,
            use_gating=True,
        )
        layer.eval()
        expert_hidden = expert_hidden_factory()
        precomputed_primary = precomputed_primary_factory()
        conditioning = condition_factory()
        _, expert_output = layer(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=expert_hidden,
            conditioning=conditioning,
        )
        assert torch.allclose(expert_output, expert_hidden, atol=1e-6)

    def test_unconditioned_mode_ignores_conditioning_argument(
        self,
        joint_attention_layer_factory: Callable[..., VLAJointAttentionLayer],
        expert_hidden_factory: Callable[..., torch.Tensor],
        precomputed_primary_factory: Callable[
            ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        condition_factory: Callable[..., torch.Tensor],
    ):
        layer = joint_attention_layer_factory(
            normalization_type=NormalizationType.RMS_NORM.value,
        )
        layer.eval()
        expert_hidden = expert_hidden_factory()
        precomputed_primary = precomputed_primary_factory()
        conditioning = condition_factory()
        _, expert_with_cond = layer(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=expert_hidden,
            conditioning=conditioning,
        )
        _, expert_without_cond = layer(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=expert_hidden,
        )
        assert torch.allclose(expert_with_cond, expert_without_cond)
