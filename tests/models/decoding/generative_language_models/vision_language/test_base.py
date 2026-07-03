"""Tests for versatil.models.decoding.generative_language_models.vision_language.base module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import Gemma2Config, LlamaConfig, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2RotaryEmbedding,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

from versatil.data.constants import SampleKey
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.paligemma import (
    PaliGemmaVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.smolvlm import (
    SmolVLM,
)
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.training.constants import PrecisionType

BATCH_SIZE = 2
SEQUENCE_LENGTH = 8
TINY_HIDDEN_DIM = 32
TINY_NUM_HEADS = 2
TINY_NUM_KV_HEADS = 2
TINY_INTERMEDIATE_DIM = 64


def _make_tiny_llama_layer() -> tuple[LlamaDecoderLayer, LlamaRotaryEmbedding]:
    config = LlamaConfig(
        hidden_size=TINY_HIDDEN_DIM,
        num_attention_heads=TINY_NUM_HEADS,
        num_key_value_heads=TINY_NUM_KV_HEADS,
        intermediate_size=TINY_INTERMEDIATE_DIM,
        num_hidden_layers=1,
        max_position_embeddings=64,
    )
    layer = LlamaDecoderLayer(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config)
    return layer, rotary


def _make_tiny_gemma2_layer() -> tuple[Gemma2DecoderLayer, Gemma2RotaryEmbedding]:
    config = Gemma2Config(
        hidden_size=TINY_HIDDEN_DIM,
        num_attention_heads=TINY_NUM_HEADS,
        num_key_value_heads=TINY_NUM_KV_HEADS,
        intermediate_size=TINY_INTERMEDIATE_DIM,
        num_hidden_layers=1,
        max_position_embeddings=64,
        head_dim=TINY_HIDDEN_DIM // TINY_NUM_HEADS,
    )
    layer = Gemma2DecoderLayer(config, layer_idx=0).eval()
    rotary = Gemma2RotaryEmbedding(config)
    return layer, rotary


class ConcreteGenerativeVLM(GenerativeVLM):
    def _compute_num_image_tokens(self, config: PretrainedConfig) -> int:
        return GenerativeVLM._compute_num_image_tokens(self, config)

    def _embed_images(
        self,
        inputs: dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return GenerativeVLM._embed_images(
            self,
            inputs=inputs,
            batch_size=batch_size,
        )

    def _get_language_model(self) -> torch.nn.Module:
        return GenerativeVLM._get_language_model(self)


@pytest.fixture(
    params=["llama", "gemma2"],
)
def tiny_vlm_layer_and_rotary(
    request: pytest.FixtureRequest,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if request.param == "llama":
        return _make_tiny_llama_layer()
    return _make_tiny_gemma2_layer()


@pytest.fixture(
    scope="session",
    params=["smolvlm", "paligemma"],
)
def vlm_backbone(
    request: pytest.FixtureRequest,
    real_smolvlm_backbone: Callable[..., SmolVLM],
    real_paligemma_backbone: Callable[..., PaliGemmaVLM],
) -> GenerativeVLM:
    if request.param == "smolvlm":
        return real_smolvlm_backbone(model_dtype=PrecisionType.FP32.value)
    return real_paligemma_backbone(model_dtype=PrecisionType.FP32.value)


@pytest.fixture
def position_ids_factory() -> Callable[..., torch.Tensor]:

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> torch.Tensor:
        return torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)

    return factory


@pytest.fixture
def tiny_hidden_factory(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, TINY_HIDDEN_DIM)).astype(
                np.float32
            )
        )

    return factory


@pytest.fixture
def concrete_vlm() -> ConcreteGenerativeVLM:
    return ConcreteGenerativeVLM(
        input_keys="left",
        pretrained=False,
        frozen=False,
        model_dtype=None,
        max_text_length=SEQUENCE_LENGTH,
    )


class TestGenerativeVLMStaticMethods:
    @pytest.mark.unit
    def test_forward_language_model_forwards_cache_arguments(
        self,
        concrete_vlm: ConcreteGenerativeVLM,
    ) -> None:
        language_model = MagicMock()
        language_output = MagicMock(spec=CausalLanguageModelOutput)
        language_output.logits = torch.ones(BATCH_SIZE, 1, TINY_HIDDEN_DIM)
        language_model.return_value = language_output
        concrete_vlm._get_language_model = MagicMock(return_value=language_model)
        input_ids = torch.ones(BATCH_SIZE, 1, dtype=torch.long)
        attention_mask = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.long)
        past_key_values = MagicMock(spec=Cache)
        cache_position = torch.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH + 1)
        position_ids = torch.full((BATCH_SIZE, 1), SEQUENCE_LENGTH)

        result = concrete_vlm.forward_language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_ids=position_ids,
            output_hidden_states=False,
        )

        language_model.assert_called_once_with(
            input_ids=input_ids,
            inputs_embeds=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        torch.testing.assert_close(result.logits, language_output.logits)

    @pytest.mark.unit
    def test_build_additive_attention_mask_maps_masked_entries_to_min_value(
        self,
    ) -> None:
        attention_mask = torch.tensor(
            [[[[False, True, False], [False, False, True]]]],
            dtype=torch.bool,
        )
        result = GenerativeVLM.build_additive_attention_mask(
            attention_mask=attention_mask,
            dtype=torch.float32,
        )
        expected = torch.zeros_like(attention_mask, dtype=torch.float32)
        expected = expected.masked_fill(attention_mask, torch.finfo(torch.float32).min)
        torch.testing.assert_close(result, expected)

    @pytest.mark.unit
    def test_build_additive_attention_mask_keeps_all_valid_mask_explicit(
        self,
    ) -> None:
        attention_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        result = GenerativeVLM.build_additive_attention_mask(
            attention_mask=attention_mask,
            dtype=torch.float32,
        )
        # None would make HF decoder layers fall back to causal attention,
        # so an all-visible mask must stay an explicit zero tensor.
        torch.testing.assert_close(result, torch.zeros(2, 1, 4, 4, dtype=torch.float32))

    @pytest.mark.unit
    def test_build_additive_attention_mask_rejects_non_float_dtype(self) -> None:
        attention_mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        with pytest.raises(
            ValueError,
            match=re.escape("dtype must be floating point, got torch.int64."),
        ):
            GenerativeVLM.build_additive_attention_mask(
                attention_mask=attention_mask,
                dtype=torch.int64,
            )

    @pytest.mark.integration
    def test_compute_rope_unsqueezes_for_head_broadcast(
        self,
        vlm_backbone: GenerativeVLM,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        rotary_embedding = vlm_backbone.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_backbone.hidden_dimension,
        )
        position_ids = position_ids_factory()
        cos, sin = GenerativeVLM.compute_rope(
            rotary_embedding=rotary_embedding,
            hidden_states=hidden,
            position_ids=position_ids,
        )
        head_dim = vlm_backbone.get_backbone_layers()[0].self_attn.head_dim
        assert cos.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH, head_dim)
        assert sin.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH, head_dim)
        raw_cos, raw_sin = rotary_embedding(hidden, position_ids)
        assert torch.allclose(cos.squeeze(1), raw_cos, atol=1e-5)
        assert torch.allclose(sin.squeeze(1), raw_sin, atol=1e-5)

    @pytest.mark.integration
    def test_extract_key_value_returns_unprojected_kv(
        self,
        vlm_backbone: GenerativeVLM,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer = vlm_backbone.get_backbone_layers()[0]
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_backbone.hidden_dimension,
        )
        key, value = GenerativeVLM.extract_key_value(
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
        vlm_backbone: GenerativeVLM,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer = vlm_backbone.get_backbone_layers()[0]
        rotary_embedding = vlm_backbone.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_backbone.hidden_dimension,
        )
        position_ids = position_ids_factory()
        key, value = GenerativeVLM.extract_key_value_with_rope(
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
        vlm_backbone: GenerativeVLM,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer = vlm_backbone.get_backbone_layers()[0]
        rotary_embedding = vlm_backbone.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_backbone.hidden_dimension,
        )
        position_ids = position_ids_factory()
        query, key, value = GenerativeVLM.extract_query_key_value(
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
        vlm_backbone: GenerativeVLM,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer = vlm_backbone.get_backbone_layers()[0]
        attention = layer.self_attn
        attention_output_dimension = (
            attention.config.num_attention_heads * attention.head_dim
        )
        residual = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_backbone.hidden_dimension,
        )
        attention_output = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=attention_output_dimension,
        )
        result = GenerativeVLM.apply_residual_feedforward(
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
        vlm_backbone: GenerativeVLM,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer = vlm_backbone.get_backbone_layers()[0]
        rotary_embedding = vlm_backbone.get_rotary_embedding()
        hidden = sequence_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dimension=vlm_backbone.hidden_dimension,
        )
        position_ids = position_ids_factory()
        key_with_rope, value_from_kv = GenerativeVLM.extract_key_value_with_rope(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary_embedding,
            position_ids=position_ids,
        )
        _, key_from_qkv, value_from_qkv = GenerativeVLM.extract_query_key_value(
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


@pytest.mark.unit
class TestGenerativeVLMTemporalHandling:
    def test_forward_flattens_and_unflattens_temporal_inputs(
        self,
        concrete_vlm: ConcreteGenerativeVLM,
    ) -> None:
        batch_size = 2
        temporal_length = 3
        sequence_length = 5
        output_sequence_length = 4
        hidden_dimension = 3
        inputs = {
            SampleKey.TOKENIZED_OBSERVATIONS.value: torch.arange(
                batch_size * temporal_length * sequence_length,
                dtype=torch.long,
            ).reshape(batch_size, temporal_length, sequence_length),
            "left": torch.arange(
                batch_size * temporal_length * 3 * 2 * 2,
                dtype=torch.float32,
            ).reshape(batch_size, temporal_length, 3, 2, 2),
        }
        flattened_fused = torch.arange(
            batch_size * temporal_length * output_sequence_length * hidden_dimension,
            dtype=torch.float32,
        ).reshape(
            batch_size * temporal_length, output_sequence_length, hidden_dimension
        )
        flattened_mask = torch.zeros(
            batch_size * temporal_length,
            output_sequence_length,
            dtype=torch.bool,
        )
        flattened_mask[:, -1] = True

        def encode(
            inputs: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            assert inputs[SampleKey.TOKENIZED_OBSERVATIONS.value].shape == (
                batch_size * temporal_length,
                sequence_length,
            )
            assert inputs["left"].shape == (batch_size * temporal_length, 3, 2, 2)
            return {
                EncoderOutputKeys.FUSED_RGB_LANGUAGE.value: flattened_fused,
                concrete_vlm.padding_mask_name: flattened_mask,
            }

        concrete_vlm.encode = encode

        output = concrete_vlm.forward(inputs=inputs)

        torch.testing.assert_close(
            output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value],
            flattened_fused.reshape(
                batch_size,
                temporal_length,
                output_sequence_length,
                hidden_dimension,
            ),
        )
        torch.testing.assert_close(
            output[concrete_vlm.padding_mask_name],
            flattened_mask.reshape(
                batch_size,
                temporal_length,
                output_sequence_length,
            ),
        )

    def test_flatten_temporal_rejects_non_tensor_inputs(
        self,
        concrete_vlm: ConcreteGenerativeVLM,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("VLM input 'bad' must be a torch.Tensor, got int."),
        ):
            concrete_vlm._flatten_temporal(
                inputs={
                    SampleKey.TOKENIZED_OBSERVATIONS.value: torch.zeros(
                        2, 3, 4, dtype=torch.long
                    ),
                    "bad": 1,
                }
            )

    def test_flatten_temporal_rejects_mismatched_leading_shape(
        self,
        concrete_vlm: ConcreteGenerativeVLM,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "VLM input 'left' has leading shape (2, 4), expected (2, 3)."
            ),
        ):
            concrete_vlm._flatten_temporal(
                inputs={
                    SampleKey.TOKENIZED_OBSERVATIONS.value: torch.zeros(
                        2, 3, 5, dtype=torch.long
                    ),
                    "left": torch.zeros(2, 4, 3, 2, 2),
                }
            )

    def test_build_prefix_flattens_temporal_outputs(
        self,
        concrete_vlm: ConcreteGenerativeVLM,
    ) -> None:
        batch_size = 2
        temporal_length = 3
        sequence_length = 4
        hidden_dimension = 5
        embeddings = torch.arange(
            batch_size * temporal_length * sequence_length * hidden_dimension,
            dtype=torch.float32,
        ).reshape(batch_size * temporal_length, sequence_length, hidden_dimension)
        padding_mask = torch.zeros(
            batch_size * temporal_length,
            sequence_length,
            dtype=torch.bool,
        )
        padding_mask[:, -1] = True

        def assemble(
            inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert inputs[SampleKey.TOKENIZED_OBSERVATIONS.value].shape == (
                batch_size * temporal_length,
                sequence_length,
            )
            return embeddings, padding_mask

        concrete_vlm._assemble_multimodal_embeddings = assemble

        prefix, mask = concrete_vlm.build_prefix(
            inputs={
                SampleKey.TOKENIZED_OBSERVATIONS.value: torch.zeros(
                    batch_size,
                    temporal_length,
                    sequence_length,
                    dtype=torch.long,
                )
            }
        )

        torch.testing.assert_close(
            prefix,
            embeddings.reshape(
                batch_size, temporal_length * sequence_length, hidden_dimension
            ),
        )
        torch.testing.assert_close(
            mask,
            padding_mask.reshape(batch_size, temporal_length * sequence_length),
        )


@pytest.mark.unit
class TestGenerativeVLMStaticMethodsBehavioral:
    def test_compute_rope_unsqueezes_for_head_broadcast(
        self,
        tiny_vlm_layer_and_rotary: tuple[torch.nn.Module, torch.nn.Module],
        tiny_hidden_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        _, rotary = tiny_vlm_layer_and_rotary
        hidden = tiny_hidden_factory()
        position_ids = position_ids_factory()
        cos, sin = GenerativeVLM.compute_rope(
            rotary_embedding=rotary,
            hidden_states=hidden,
            position_ids=position_ids,
        )
        head_dim = TINY_HIDDEN_DIM // TINY_NUM_HEADS
        assert cos.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH, head_dim)
        assert sin.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH, head_dim)
        raw_cos, raw_sin = rotary(hidden, position_ids)
        assert torch.allclose(cos.squeeze(1), raw_cos, atol=1e-5)
        assert torch.allclose(sin.squeeze(1), raw_sin, atol=1e-5)

    def test_extract_key_value_returns_unprojected_kv(
        self,
        tiny_vlm_layer_and_rotary: tuple[torch.nn.Module, torch.nn.Module],
        tiny_hidden_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer, _ = tiny_vlm_layer_and_rotary
        hidden = tiny_hidden_factory()
        key, value = GenerativeVLM.extract_key_value(
            vlm_layer=layer,
            hidden_states=hidden,
        )
        normalized = layer.input_layernorm(hidden)
        expected_key = layer.self_attn.k_proj(normalized)
        expected_value = layer.self_attn.v_proj(normalized)
        assert torch.allclose(key, expected_key, atol=1e-5)
        assert torch.allclose(value, expected_value, atol=1e-5)

    def test_extract_key_value_with_rope_applies_rotation_to_keys(
        self,
        tiny_vlm_layer_and_rotary: tuple[torch.nn.Module, torch.nn.Module],
        tiny_hidden_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer, rotary = tiny_vlm_layer_and_rotary
        hidden = tiny_hidden_factory()
        position_ids = position_ids_factory()
        key, value = GenerativeVLM.extract_key_value_with_rope(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary,
            position_ids=position_ids,
        )
        normalized = layer.input_layernorm(hidden)
        raw_key = layer.self_attn.k_proj(normalized)
        raw_value = layer.self_attn.v_proj(normalized)
        assert not torch.allclose(key, raw_key, atol=1e-5)
        assert torch.allclose(value, raw_value, atol=1e-5)
        key_value_dimension = TINY_NUM_KV_HEADS * (TINY_HIDDEN_DIM // TINY_NUM_HEADS)
        assert key.shape == (BATCH_SIZE, SEQUENCE_LENGTH, key_value_dimension)
        assert value.shape == (BATCH_SIZE, SEQUENCE_LENGTH, key_value_dimension)

    def test_extract_query_key_value_applies_rope_to_query_and_key(
        self,
        tiny_vlm_layer_and_rotary: tuple[torch.nn.Module, torch.nn.Module],
        tiny_hidden_factory: Callable[..., torch.Tensor],
        position_ids_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer, rotary = tiny_vlm_layer_and_rotary
        hidden = tiny_hidden_factory()
        position_ids = position_ids_factory()
        query, key, value = GenerativeVLM.extract_query_key_value(
            vlm_layer=layer,
            hidden_states=hidden,
            rotary_embedding=rotary,
            position_ids=position_ids,
        )
        normalized = layer.input_layernorm(hidden)
        head_dim = TINY_HIDDEN_DIM // TINY_NUM_HEADS
        raw_query = (
            layer.self_attn.q_proj(normalized)
            .view(BATCH_SIZE, SEQUENCE_LENGTH, TINY_NUM_HEADS, head_dim)
            .transpose(1, 2)
        )
        raw_key = (
            layer.self_attn.k_proj(normalized)
            .view(BATCH_SIZE, SEQUENCE_LENGTH, TINY_NUM_KV_HEADS, head_dim)
            .transpose(1, 2)
        )
        raw_value = (
            layer.self_attn.v_proj(normalized)
            .view(BATCH_SIZE, SEQUENCE_LENGTH, TINY_NUM_KV_HEADS, head_dim)
            .transpose(1, 2)
        )
        assert not torch.allclose(query, raw_query, atol=1e-5)
        assert not torch.allclose(key, raw_key, atol=1e-5)
        assert torch.allclose(value, raw_value, atol=1e-5)

    def test_apply_residual_feedforward_matches_manual_forward(
        self,
        tiny_vlm_layer_and_rotary: tuple[torch.nn.Module, torch.nn.Module],
        tiny_hidden_factory: Callable[..., torch.Tensor],
    ) -> None:
        layer, _ = tiny_vlm_layer_and_rotary
        attention_output_dim = TINY_NUM_HEADS * (TINY_HIDDEN_DIM // TINY_NUM_HEADS)
        residual = tiny_hidden_factory()
        attention_output = tiny_hidden_factory(sequence_length=SEQUENCE_LENGTH)[
            ..., :attention_output_dim
        ]
        result = GenerativeVLM.apply_residual_feedforward(
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
            normalized = layer.post_attention_layernorm(after_first_residual)
            mlp_output = layer.mlp(normalized)
            expected = after_first_residual + mlp_output
        assert torch.allclose(result, expected, atol=1e-5)


class TestGenerativeVLMStaticMethodsUnit:
    @pytest.mark.unit
    def test_scale_language_embeddings_default_is_identity(self) -> None:
        hidden = torch.zeros(2, 4, 8)
        result = GenerativeVLM._scale_language_embeddings(GenerativeVLM, hidden)
        assert torch.equal(result, hidden)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method_name",
        ["_compute_num_image_tokens", "_embed_images", "_get_language_model"],
    )
    def test_abstract_method_default_body_raises_not_implemented(
        self,
        method_name: str,
    ) -> None:
        method = getattr(GenerativeVLM, method_name)
        with pytest.raises(NotImplementedError):
            if method_name == "_compute_num_image_tokens":
                method(GenerativeVLM, config=None)
            elif method_name == "_embed_images":
                method(GenerativeVLM, inputs={}, batch_size=1)
            else:
                method(GenerativeVLM)
