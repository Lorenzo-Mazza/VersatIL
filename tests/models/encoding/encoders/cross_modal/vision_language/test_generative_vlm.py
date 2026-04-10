"""Tests for versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm import (
    GenerativeVLMEncoder,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.paligemma import (
    PaliGemmaEncoder,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm import (
    SmolVLMEncoder,
)

BATCH_SIZE = 2
SEQUENCE_LENGTH = 8


@pytest.fixture(
    scope="session",
    params=["smolvlm", "paligemma"],
)
def vlm_encoder(
    request: pytest.FixtureRequest,
    real_smolvlm_encoder: Callable[..., SmolVLMEncoder],
    real_paligemma_encoder: Callable[..., PaliGemmaEncoder],
) -> GenerativeVLMEncoder:
    if request.param == "smolvlm":
        return real_smolvlm_encoder()
    return real_paligemma_encoder()


@pytest.fixture
def position_ids_factory() -> Callable[..., torch.Tensor]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> torch.Tensor:
        return torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)

    return factory


class TestGenerativeVLMStaticMethods:
    @pytest.mark.integration
    def test_compute_rope_unsqueezes_for_head_broadcast(
        self,
        vlm_encoder: GenerativeVLMEncoder,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ):
        rotary_embedding = vlm_encoder.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_encoder.hidden_dim,
        )
        position_ids = position_ids_factory()
        cos, sin = GenerativeVLMEncoder.compute_rope(
            rotary_embedding=rotary_embedding,
            hidden_states=hidden,
            position_ids=position_ids,
        )
        head_dim = vlm_encoder.get_backbone_layers()[0].self_attn.head_dim
        assert cos.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH, head_dim)
        assert sin.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH, head_dim)
        raw_cos, raw_sin = rotary_embedding(hidden, position_ids)
        assert torch.allclose(cos.squeeze(1), raw_cos, atol=1e-5)
        assert torch.allclose(sin.squeeze(1), raw_sin, atol=1e-5)

    @pytest.mark.integration
    def test_extract_key_value_returns_unprojected_kv(
        self,
        vlm_encoder: GenerativeVLMEncoder,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = vlm_encoder.get_backbone_layers()[0]
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_encoder.hidden_dim,
        )
        key, value = GenerativeVLMEncoder.extract_key_value(
            vlm_layer=layer,
            hidden_states=hidden,
        )
        normalized = layer.input_layernorm(hidden)
        expected_key = layer.self_attn.k_proj(normalized)
        expected_value = layer.self_attn.v_proj(normalized)
        assert torch.allclose(key, expected_key, atol=1e-5)
        assert torch.allclose(value, expected_value, atol=1e-5)

    @pytest.mark.integration
    def test_extract_key_value_with_rope_applies_rotation_to_keys(
        self,
        vlm_encoder: GenerativeVLMEncoder,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ):
        layer = vlm_encoder.get_backbone_layers()[0]
        rotary_embedding = vlm_encoder.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_encoder.hidden_dim,
        )
        position_ids = position_ids_factory()
        key, value = GenerativeVLMEncoder.extract_key_value_with_rope(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary_embedding,
            position_ids=position_ids,
        )
        normalized = layer.input_layernorm(hidden)
        raw_key = layer.self_attn.k_proj(normalized)
        raw_value = layer.self_attn.v_proj(normalized)
        assert not torch.allclose(key, raw_key, atol=1e-5)
        assert torch.allclose(value, raw_value, atol=1e-5)
        key_value_dimension = (
            layer.self_attn.config.num_key_value_heads * layer.self_attn.head_dim
        )
        assert key.shape == (BATCH_SIZE, SEQUENCE_LENGTH, key_value_dimension)
        assert value.shape == (BATCH_SIZE, SEQUENCE_LENGTH, key_value_dimension)

    @pytest.mark.integration
    def test_extract_query_key_value_applies_rope_to_query_and_key(
        self,
        vlm_encoder: GenerativeVLMEncoder,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ):
        layer = vlm_encoder.get_backbone_layers()[0]
        rotary_embedding = vlm_encoder.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_encoder.hidden_dim,
        )
        position_ids = position_ids_factory()
        query, key, value = GenerativeVLMEncoder.extract_query_key_value(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary_embedding,
            position_ids=position_ids,
        )
        normalized = layer.input_layernorm(hidden)
        head_dim = layer.self_attn.head_dim
        number_of_heads = layer.self_attn.config.num_attention_heads
        number_of_key_value_heads = layer.self_attn.config.num_key_value_heads
        raw_query = (
            layer.self_attn.q_proj(normalized)
            .view(BATCH_SIZE, SEQUENCE_LENGTH, number_of_heads, head_dim)
            .transpose(1, 2)
        )
        raw_key = (
            layer.self_attn.k_proj(normalized)
            .view(BATCH_SIZE, SEQUENCE_LENGTH, number_of_key_value_heads, head_dim)
            .transpose(1, 2)
        )
        raw_value = (
            layer.self_attn.v_proj(normalized)
            .view(BATCH_SIZE, SEQUENCE_LENGTH, number_of_key_value_heads, head_dim)
            .transpose(1, 2)
        )
        # RoPE applied to Q and K
        assert not torch.allclose(query, raw_query, atol=1e-5)
        assert not torch.allclose(key, raw_key, atol=1e-5)
        # V is untouched
        assert torch.allclose(value, raw_value, atol=1e-5)

    @pytest.mark.integration
    def test_apply_residual_feedforward_matches_manual_forward(
        self,
        vlm_encoder: GenerativeVLMEncoder,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        layer = vlm_encoder.get_backbone_layers()[0]
        attention = layer.self_attn
        attention_output_dimension = (
            attention.config.num_attention_heads * attention.head_dim
        )
        residual = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_encoder.hidden_dim,
        )
        attention_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=attention_output_dimension,
        )
        result = GenerativeVLMEncoder.apply_residual_feedforward(
            vlm_layer=layer,
            vlm_residual=residual,
            vlm_attention_output=attention_output,
        )
        o_proj_output = layer.self_attn.o_proj(attention_output)
        if hasattr(layer, "pre_feedforward_layernorm"):
            post_attention_normed = layer.post_attention_layernorm(o_proj_output)
            after_first_residual = residual + post_attention_normed
            pre_feedforward = layer.pre_feedforward_layernorm(after_first_residual)
            mlp_output = layer.mlp(pre_feedforward)
            post_feedforward = layer.post_feedforward_layernorm(mlp_output)
            expected = after_first_residual + post_feedforward
        else:
            after_first_residual = residual + o_proj_output
            normed = layer.post_attention_layernorm(after_first_residual)
            mlp_output = layer.mlp(normed)
            expected = after_first_residual + mlp_output
        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.integration
    def test_extract_key_value_with_rope_consistent_with_extract_query_key_value(
        self,
        vlm_encoder: GenerativeVLMEncoder,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ):
        layer = vlm_encoder.get_backbone_layers()[0]
        rotary_embedding = vlm_encoder.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_encoder.hidden_dim,
        )
        position_ids = position_ids_factory()
        key_with_rope, value_from_kv = GenerativeVLMEncoder.extract_key_value_with_rope(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary_embedding,
            position_ids=position_ids,
        )
        _, key_from_qkv, value_from_qkv = GenerativeVLMEncoder.extract_query_key_value(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary_embedding,
            position_ids=position_ids,
        )
        head_dim = layer.self_attn.head_dim
        number_of_key_value_heads = layer.self_attn.config.num_key_value_heads
        key_with_rope_headed = key_with_rope.view(
            BATCH_SIZE, SEQUENCE_LENGTH, number_of_key_value_heads, head_dim
        ).transpose(1, 2)
        assert torch.allclose(key_with_rope_headed, key_from_qkv, atol=1e-5)
        value_from_kv_headed = value_from_kv.view(
            BATCH_SIZE, SEQUENCE_LENGTH, number_of_key_value_heads, head_dim
        ).transpose(1, 2)
        assert torch.allclose(value_from_kv_headed, value_from_qkv, atol=1e-5)
