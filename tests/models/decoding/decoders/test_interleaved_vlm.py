"""Tests for versatil.models.decoding.decoders.interleaved_vlm module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, TimeConditioning
from versatil.models.decoding.decoders.interleaved_vlm import (
    BaseInterleavedVLMDecoder,
    InterleavedLayerType,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.input_specification import InputSpecification
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    ConditioningLayerCache,
)

PREFIX_TOKEN_COUNT = 3
ACTION_TOKEN_COUNT = 2
HIDDEN_DIMENSION = 4
BATCH_SIZE = 2
RAW_IMAGE_KEY = "raw_image"
ENCODED_FEATURE_KEY = "encoded_proprio"


class _FakeRotaryEmbedding(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_shape = (*position_ids.shape, hidden_states.shape[-1])
        return torch.ones(embedding_shape), torch.zeros(embedding_shape)


class _TestInterleavedDecoder(BaseInterleavedVLMDecoder):
    def build_action_expert(
        self,
        vlm_layers: torch.nn.ModuleList,
        rotary_emb: torch.nn.Module,
        vlm_hidden_dimension: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        self.backbone_hidden_dimension = vlm_hidden_dimension

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {}


class _FakeVLMLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_mock = MagicMock(spec=self.forward)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.forward_mock(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        return hidden_states + 1.0


class _FakeJointExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_with_secondary_mock = MagicMock(spec=self.forward_with_secondary)
        self.forward_mock = MagicMock(spec=self.forward)

    def forward_with_secondary(
        self,
        hidden_states_primary: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        conditioning: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        precomputed_primary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_with_secondary_mock(
            hidden_states_primary=hidden_states_primary,
            conditioning_cache=conditioning_cache,
            conditioning=conditioning,
            joint_attention_mask=joint_attention_mask,
            precomputed_primary_rope=precomputed_primary_rope,
        )
        return hidden_states_primary + 2.0, hidden_states_primary + 3.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        conditioning: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.forward_mock(
            hidden_states=hidden_states,
            conditioning_cache=conditioning_cache,
            conditioning=conditioning,
            attention_mask=attention_mask,
            precomputed_rope=precomputed_rope,
        )
        return hidden_states + 2.0


class _FakeCrossExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_mock = MagicMock(spec=self.forward)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        attention_mask: torch.Tensor | None = None,
        precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.forward_mock(
            hidden_states=hidden_states,
            conditioning_cache=conditioning_cache,
            attention_mask=attention_mask,
            precomputed_rope=precomputed_rope,
        )
        return hidden_states + 4.0


@pytest.fixture
def fake_vlm_backbone_factory() -> Callable[..., MagicMock]:
    def factory(
        raw_input_keys: list[str] | None = None,
        hidden_dimension: int = HIDDEN_DIMENSION,
    ) -> MagicMock:
        if raw_input_keys is None:
            raw_input_keys = [RAW_IMAGE_KEY]
        backbone = MagicMock(spec=GenerativeVLM)
        backbone.input_specification = InputSpecification(keys=raw_input_keys)
        backbone.layers = torch.nn.ModuleList()
        backbone.rotary_embedding = _FakeRotaryEmbedding()
        backbone.hidden_dimension = hidden_dimension
        backbone.text_config = PretrainedConfig()
        return backbone

    return factory


@pytest.fixture
def routed_interleaved_decoder_factory(
    interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
) -> Callable[[], _TestInterleavedDecoder]:
    def factory() -> _TestInterleavedDecoder:
        decoder = interleaved_decoder_factory()
        decoder.vlm_layers = torch.nn.ModuleList(
            [_FakeVLMLayer(), _FakeVLMLayer(), _FakeVLMLayer()]
        )
        decoder.expert_layers = torch.nn.ModuleList(
            [_FakeJointExpertLayer(), _FakeCrossExpertLayer()]
        )
        decoder.vlm_rotary_embedding = _FakeRotaryEmbedding()
        decoder._layer_types = [
            InterleavedLayerType.VLM_ONLY.value,
            InterleavedLayerType.JOINT_SELF_ATTENTION.value,
            InterleavedLayerType.CROSS_ATTENTION.value,
        ]
        return decoder

    return factory


@pytest.fixture
def interleaved_decoder_factory(
    mock_action_space_factory: Callable[..., ActionSpace],
    mock_observation_space_factory: Callable[..., ObservationSpace],
    action_head_factory: Callable[..., ActionHead],
    fake_vlm_backbone_factory: Callable[..., MagicMock],
) -> Callable[..., _TestInterleavedDecoder]:
    def factory(
        input_keys: list[str] | None = None,
        raw_input_keys: list[str] | None = None,
    ) -> _TestInterleavedDecoder:
        if input_keys is None:
            input_keys = [ENCODED_FEATURE_KEY]
        action_space = mock_action_space_factory(position_dim=3)
        return _TestInterleavedDecoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads={
                "joint_action": action_head_factory(input_dimension=HIDDEN_DIMENSION)
            },
            observation_space=mock_observation_space_factory(),
            observation_horizon=1,
            prediction_horizon=2,
            device="cpu",
            vlm_backbone=fake_vlm_backbone_factory(raw_input_keys=raw_input_keys),
            requires_actions=True,
        )

    return factory


@pytest.mark.unit
class TestBaseInterleavedVLMDecoderWiring:
    def test_decoder_input_contains_encoded_and_raw_vlm_keys(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory(
            input_keys=[ENCODED_FEATURE_KEY],
            raw_input_keys=[RAW_IMAGE_KEY, "tokenized_text"],
        )

        assert decoder.decoder_input.keys == [
            ENCODED_FEATURE_KEY,
            RAW_IMAGE_KEY,
            "tokenized_text",
        ]
        assert decoder.decoder_input.needs_raw_observations

    def test_cache_toggles_clear_prefix_cache(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        decoder._prefix_cache = MagicMock(spec=ConditioningCache)

        decoder.enable_encoder_cache()

        assert decoder._encoder_cache_enabled
        assert decoder._prefix_cache is None

        decoder._prefix_cache = MagicMock(spec=ConditioningCache)
        decoder.disable_encoder_cache()

        assert not decoder._encoder_cache_enabled
        assert decoder._prefix_cache is None

    @pytest.mark.parametrize(
        "time_conditioning, expected_conditioning_input_features",
        [
            (TimeConditioning.CONCAT_MLP.value, HIDDEN_DIMENSION * 2),
            (TimeConditioning.ADANORM.value, HIDDEN_DIMENSION),
        ],
    )
    def test_set_action_suffix_modules_builds_shared_layers(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
        time_conditioning: str,
        expected_conditioning_input_features: int,
    ) -> None:
        decoder = interleaved_decoder_factory()

        decoder._set_action_suffix_modules(
            expert_hidden_dimension=HIDDEN_DIMENSION,
            time_conditioning=time_conditioning,
            min_period=0.01,
            max_period=10.0,
            normalization_type=NormalizationType.RMS_NORM.value,
        )

        assert decoder.action_input_projection.in_features == 3
        assert decoder.action_input_projection.out_features == HIDDEN_DIMENSION
        assert decoder._single_action_head().input_dimension == HIDDEN_DIMENSION
        assert decoder._single_action_head().output_dim == 3
        assert (
            decoder.time_conditioning_input.in_features
            == expected_conditioning_input_features
        )
        assert decoder.time_conditioning_input.out_features == HIDDEN_DIMENSION
        assert decoder.time_conditioning_output.in_features == HIDDEN_DIMENSION
        assert decoder.time_conditioning_output.out_features == HIDDEN_DIMENSION
        timestep = torch.tensor([0.25, 0.75])
        assert decoder.timestep_embedding(timestep).shape == (
            BATCH_SIZE,
            HIDDEN_DIMENSION,
        )
        normalized = decoder.expert_final_normalization(
            torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION)
        )
        assert normalized.shape == (
            BATCH_SIZE,
            ACTION_TOKEN_COUNT,
            HIDDEN_DIMENSION,
        )

    def test_get_vlm_attention_dimensions_read_config_values(self) -> None:
        config = PretrainedConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
        )

        head_dimension = BaseInterleavedVLMDecoder._get_vlm_head_dimension(
            vlm_text_config=config,
            vlm_hidden_dimension=64,
        )
        key_value_heads = BaseInterleavedVLMDecoder._get_vlm_num_key_value_heads(
            vlm_text_config=config
        )

        assert head_dimension == 16
        assert key_value_heads == 2

    def test_get_vlm_attention_dimensions_fall_back_to_multi_head_attention(
        self,
    ) -> None:
        config = PretrainedConfig(num_attention_heads=4)

        head_dimension = BaseInterleavedVLMDecoder._get_vlm_head_dimension(
            vlm_text_config=config,
            vlm_hidden_dimension=32,
        )
        key_value_heads = BaseInterleavedVLMDecoder._get_vlm_num_key_value_heads(
            vlm_text_config=config
        )

        assert head_dimension == 8
        assert key_value_heads == 4


@pytest.mark.unit
class TestBaseInterleavedVLMPrefixHelpers:
    def test_append_valid_prefix_tokens_extends_embeddings_and_mask(self) -> None:
        prefix_embeddings = torch.arange(
            BATCH_SIZE * PREFIX_TOKEN_COUNT * HIDDEN_DIMENSION,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION)
        prefix_padding_mask = torch.tensor(
            [[False, True, False], [False, False, True]],
            dtype=torch.bool,
        )
        appended_tokens = torch.ones(BATCH_SIZE, HIDDEN_DIMENSION)

        updated_embeddings, updated_mask, appended_token_count = (
            BaseInterleavedVLMDecoder._append_valid_prefix_tokens(
                prefix_embeddings=prefix_embeddings,
                prefix_padding_mask=prefix_padding_mask,
                prefix_tokens=appended_tokens,
            )
        )

        assert appended_token_count == 1
        assert torch.equal(
            updated_embeddings[:, :PREFIX_TOKEN_COUNT], prefix_embeddings
        )
        assert torch.equal(updated_embeddings[:, PREFIX_TOKEN_COUNT], appended_tokens)
        expected_mask = torch.tensor(
            [[False, True, False, False], [False, False, True, False]],
            dtype=torch.bool,
        )
        assert torch.equal(updated_mask, expected_mask)

    def test_append_projected_prefix_feature_calls_projection_with_feature_key(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        prefix_embeddings = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_COUNT,
            HIDDEN_DIMENSION,
        )
        prefix_padding_mask = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_COUNT,
            dtype=torch.bool,
        )
        feature_tensor = torch.ones(BATCH_SIZE, HIDDEN_DIMENSION)
        projected_tensor = feature_tensor * 2.0
        projection = MagicMock(spec=FeatureProjection)
        projection.return_value = {ENCODED_FEATURE_KEY: projected_tensor}

        updated_embeddings, updated_mask, appended_token_count = (
            decoder._append_projected_prefix_feature(
                prefix_embeddings=prefix_embeddings,
                prefix_padding_mask=prefix_padding_mask,
                features={ENCODED_FEATURE_KEY: feature_tensor},
                feature_key=ENCODED_FEATURE_KEY,
                projection=projection,
            )
        )

        projection.assert_called_once()
        assert torch.equal(
            projection.call_args.kwargs["features"][ENCODED_FEATURE_KEY],
            feature_tensor,
        )
        assert appended_token_count == 1
        assert torch.equal(updated_embeddings[:, -1], projected_tensor)
        assert not updated_mask[:, -1].any()

    def test_append_projected_prefix_feature_raises_for_missing_feature(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        expected_message = (
            f"Missing '{ENCODED_FEATURE_KEY}' in features for projected VLM "
            "prefix token."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._append_projected_prefix_feature(
                prefix_embeddings=torch.zeros(
                    BATCH_SIZE,
                    PREFIX_TOKEN_COUNT,
                    HIDDEN_DIMENSION,
                ),
                prefix_padding_mask=None,
                features={},
                feature_key=ENCODED_FEATURE_KEY,
                projection=MagicMock(spec=FeatureProjection),
            )

    def test_append_optional_projected_prefix_feature_returns_inputs_without_config(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        prefix_embeddings = torch.zeros(
            BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION
        )
        prefix_padding_mask = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_COUNT,
            dtype=torch.bool,
        )

        updated_embeddings, updated_mask, appended_token_count = (
            decoder._append_optional_projected_prefix_feature(
                prefix_embeddings=prefix_embeddings,
                prefix_padding_mask=prefix_padding_mask,
                features={},
                feature_key=None,
                projection=None,
            )
        )

        assert torch.equal(updated_embeddings, prefix_embeddings)
        assert torch.equal(updated_mask, prefix_padding_mask)
        assert appended_token_count == 0


@pytest.mark.unit
class TestBaseInterleavedVLMActionSuffix:
    def test_concat_mlp_suffix_embeds_actions_and_timestep(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        actions = {
            "position_action": torch.ones(
                BATCH_SIZE,
                ACTION_TOKEN_COUNT,
                3,
            )
        }
        timestep = torch.tensor([0.25, 0.75])
        action_embedding = torch.full(
            (BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
            2.0,
        )
        time_embedding = torch.full((BATCH_SIZE, HIDDEN_DIMENSION), 3.0)
        hidden_embedding = torch.full(
            (BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
            4.0,
        )
        suffix_embedding = torch.full(
            (BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
            5.0,
        )
        action_input_projection = MagicMock(
            spec=torch.nn.Module,
            return_value=action_embedding,
        )
        timestep_embedding = MagicMock(
            spec=torch.nn.Module,
            return_value=time_embedding,
        )
        conditioning_input_layer = MagicMock(
            spec=torch.nn.Module,
            return_value=hidden_embedding,
        )
        conditioning_output_layer = MagicMock(
            spec=torch.nn.Module,
            return_value=suffix_embedding,
        )

        embedded_actions, adaptive_norm_conditioning = (
            decoder._embed_timestep_conditioned_action_suffix(
                actions=actions,
                timestep=timestep,
                action_input_projection=action_input_projection,
                timestep_embedding=timestep_embedding,
                time_conditioning=TimeConditioning.CONCAT_MLP.value,
                conditioning_input_layer=conditioning_input_layer,
                conditioning_output_layer=conditioning_output_layer,
            )
        )

        assert torch.equal(embedded_actions, suffix_embedding)
        assert adaptive_norm_conditioning is None
        assert torch.equal(
            action_input_projection.call_args.args[0], actions["position_action"]
        )
        assert torch.equal(timestep_embedding.call_args.args[0], timestep)
        fused_embedding = conditioning_input_layer.call_args.args[0]
        assert fused_embedding.shape == (
            BATCH_SIZE,
            ACTION_TOKEN_COUNT,
            HIDDEN_DIMENSION * 2,
        )
        assert torch.equal(fused_embedding[:, :, :HIDDEN_DIMENSION], action_embedding)
        assert torch.equal(
            fused_embedding[:, :, HIDDEN_DIMENSION:],
            time_embedding.unsqueeze(1).expand_as(action_embedding),
        )

    def test_adanorm_suffix_returns_action_embedding_and_conditioning(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        actions = {
            "position_action": torch.ones(
                BATCH_SIZE,
                ACTION_TOKEN_COUNT,
                3,
            )
        }
        timestep = torch.tensor([0.1, 0.2])
        action_embedding = torch.full(
            (BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
            2.0,
        )
        time_embedding = torch.full((BATCH_SIZE, HIDDEN_DIMENSION), 3.0)
        time_hidden = torch.full((BATCH_SIZE, HIDDEN_DIMENSION), 4.0)
        conditioning_source = torch.full((BATCH_SIZE, HIDDEN_DIMENSION), 5.0)
        action_input_projection = MagicMock(
            spec=torch.nn.Module,
            return_value=action_embedding,
        )
        timestep_embedding = MagicMock(
            spec=torch.nn.Module,
            return_value=time_embedding,
        )
        conditioning_input_layer = MagicMock(
            spec=torch.nn.Module,
            return_value=time_hidden,
        )
        conditioning_output_layer = MagicMock(
            spec=torch.nn.Module,
            return_value=conditioning_source,
        )

        embedded_actions, adaptive_norm_conditioning = (
            decoder._embed_timestep_conditioned_action_suffix(
                actions=actions,
                timestep=timestep,
                action_input_projection=action_input_projection,
                timestep_embedding=timestep_embedding,
                time_conditioning=TimeConditioning.ADANORM.value,
                conditioning_input_layer=conditioning_input_layer,
                conditioning_output_layer=conditioning_output_layer,
            )
        )

        assert torch.equal(embedded_actions, action_embedding)
        expected_conditioning = torch.nn.functional.silu(conditioning_source)
        assert torch.equal(adaptive_norm_conditioning, expected_conditioning)
        assert torch.equal(conditioning_input_layer.call_args.args[0], time_embedding)


@pytest.mark.unit
class TestBaseInterleavedVLMAttentionState:
    def test_build_interleaved_attention_state_calls_shared_dependencies(
        self,
    ) -> None:
        prefix_embeddings = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_COUNT,
            HIDDEN_DIMENSION,
        )
        expert_tokens = torch.zeros(
            BATCH_SIZE,
            ACTION_TOKEN_COUNT,
            HIDDEN_DIMENSION,
        )
        prefix_padding_mask = torch.tensor(
            [[False, True, False], [False, False, False]],
            dtype=torch.bool,
        )
        total_token_count = PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT
        source_attention_mask = torch.zeros(
            BATCH_SIZE,
            1,
            total_token_count,
            total_token_count,
            dtype=torch.bool,
        )
        source_attention_mask[:, :, 0, -1] = True
        key_padding_mask = torch.cat(
            [
                prefix_padding_mask,
                torch.zeros(BATCH_SIZE, ACTION_TOKEN_COUNT, dtype=torch.bool),
            ],
            dim=1,
        )
        additive_mask = torch.full(
            (BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, PREFIX_TOKEN_COUNT),
            -1.0,
        )
        cosine = torch.ones(BATCH_SIZE, 1, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION)
        sine = torch.zeros(BATCH_SIZE, 1, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION)

        with (
            patch(
                "versatil.models.decoding.decoders.interleaved_vlm.make_attention_mask",
                return_value=(source_attention_mask, key_padding_mask),
            ) as attention_builder,
            patch.object(
                GenerativeVLM,
                "build_additive_attention_mask",
                return_value=additive_mask,
            ) as additive_builder,
            patch.object(
                GenerativeVLM,
                "compute_rope",
                return_value=(cosine, sine),
            ) as rope_builder,
        ):
            state = BaseInterleavedVLMDecoder._build_interleaved_attention_state(
                prefix_embeddings=prefix_embeddings,
                prefix_padding_mask=prefix_padding_mask,
                expert_tokens=expert_tokens,
                rotary_embedding=_FakeRotaryEmbedding(),
                causal_actions=True,
                causal_prefix_suffix_length=1,
            )

        attention_builder.assert_called_once()
        assert torch.equal(
            attention_builder.call_args.kwargs["feature_token_mask"],
            prefix_padding_mask,
        )
        assert attention_builder.call_args.kwargs["causal_actions"]
        assert attention_builder.call_args.kwargs["causal_prefix_suffix_length"] == 1
        additive_builder.assert_called_once()
        assert torch.equal(
            additive_builder.call_args.kwargs["attention_mask"],
            source_attention_mask[:, :, :PREFIX_TOKEN_COUNT, :PREFIX_TOKEN_COUNT],
        )
        rope_builder.assert_called_once()
        assert torch.equal(
            rope_builder.call_args.kwargs["position_ids"],
            state.expert_position_ids,
        )
        permutation = torch.tensor([3, 4, 0, 1, 2])
        expected_attention_mask = source_attention_mask[:, :, permutation, :][
            :, :, :, permutation
        ]
        assert torch.equal(state.attention_mask, expected_attention_mask)
        assert torch.equal(state.key_padding_mask, key_padding_mask)
        assert torch.equal(state.vlm_prefix_attention_mask, additive_mask)
        assert torch.equal(state.expert_action_rope[0], cosine)
        assert torch.equal(state.expert_action_rope[1], sine)
        expected_position_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
        assert torch.equal(state.position_ids, expected_position_ids)

    def test_slice_expert_to_vlm_attention_mask_returns_cross_block(self) -> None:
        attention_mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
        attention_mask[:, :, :ACTION_TOKEN_COUNT, ACTION_TOKEN_COUNT:] = True

        cross_attention_mask = (
            BaseInterleavedVLMDecoder._slice_expert_to_vlm_attention_mask(
                attention_mask=attention_mask,
                action_token_count=ACTION_TOKEN_COUNT,
            )
        )

        assert cross_attention_mask.shape == (1, 1, ACTION_TOKEN_COUNT, 3)
        assert cross_attention_mask.all()

    def test_zero_based_position_ids_shifts_each_row(self) -> None:
        position_ids = torch.tensor([[3, 4, 5], [7, 9, 10]])

        shifted_position_ids = BaseInterleavedVLMDecoder._zero_based_position_ids(
            position_ids=position_ids,
        )

        expected_position_ids = torch.tensor([[0, 1, 2], [0, 2, 3]])
        assert torch.equal(shifted_position_ids, expected_position_ids)


@pytest.mark.unit
class TestBaseInterleavedVLMForwardHelpers:
    def test_require_forward_actions_returns_actions(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        actions = {"position_action": torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, 3)}

        result = decoder._require_forward_actions(actions=actions)

        assert torch.equal(result["position_action"], actions["position_action"])

    def test_require_forward_actions_raises_for_missing_actions(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        expected_message = (
            "_TestInterleavedDecoder requires actions during forward "
            "(noisy actions for denoising)."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._require_forward_actions(actions=None)

    def test_get_forward_timestep_returns_algorithm_timestep(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        timestep = torch.tensor([0.25, 0.75])

        result = decoder._get_forward_timestep(
            features={DecoderOutputKey.TIMESTEP.value: timestep}
        )

        assert torch.equal(result, timestep)

    def test_get_forward_timestep_raises_for_missing_timestep(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        expected_message = (
            f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
            "The algorithm should inject timesteps into features."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._get_forward_timestep(features={})

    def test_project_expert_actions_normalizes_projects_and_splits_by_action_head(
        self,
        interleaved_decoder_factory: Callable[..., _TestInterleavedDecoder],
    ) -> None:
        decoder = interleaved_decoder_factory()
        joint_head = ActionHead(input_dimension=HIDDEN_DIMENSION)
        joint_head.set_output_dim(3)
        decoder.action_heads.clear()
        decoder.action_heads["joint_action"] = joint_head
        expert_hidden = torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION)
        normalized_hidden = expert_hidden + 1.0
        projected_actions = torch.arange(
            BATCH_SIZE * decoder.prediction_horizon * 3,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, decoder.prediction_horizon, 3)
        expert_final_normalization = MagicMock(
            spec=torch.nn.Module,
            return_value=normalized_hidden,
        )
        joint_head.forward = MagicMock(
            spec=joint_head.forward,
            return_value=projected_actions,
        )

        predictions = decoder._project_expert_actions(
            expert_hidden=expert_hidden,
            expert_final_normalization=expert_final_normalization,
        )

        expert_final_normalization.assert_called_once()
        assert torch.equal(expert_final_normalization.call_args.args[0], expert_hidden)
        joint_head.forward.assert_called_once()
        assert torch.equal(
            joint_head.forward.call_args.args[0],
            normalized_hidden[:, -decoder.prediction_horizon :, :],
        )
        assert torch.equal(predictions["position_action"], projected_actions)


@pytest.mark.unit
class TestBaseInterleavedVLMLayerRouting:
    def test_training_forward_routes_vlm_only_joint_and_cross_layers(
        self,
        routed_interleaved_decoder_factory: Callable[[], _TestInterleavedDecoder],
    ) -> None:
        decoder = routed_interleaved_decoder_factory()
        prefix_embeddings = torch.zeros(
            BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION
        )
        expert_hidden = torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION)
        attention_mask = torch.zeros(
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT,
            PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT,
            dtype=torch.bool,
        )
        cross_attention_mask = torch.zeros(
            BATCH_SIZE,
            1,
            ACTION_TOKEN_COUNT,
            PREFIX_TOKEN_COUNT,
            dtype=torch.bool,
        )
        vlm_prefix_attention_mask = torch.zeros(
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_COUNT,
            PREFIX_TOKEN_COUNT,
        )
        position_ids = torch.arange(PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT).repeat(
            BATCH_SIZE,
            1,
        )
        rope = (
            torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
            torch.zeros(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
        )
        cross_rope = (
            torch.full((BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION), 2.0),
            torch.zeros(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
        )
        adaptive_norm_conditioning = torch.ones(BATCH_SIZE, HIDDEN_DIMENSION)
        query = torch.full((BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 3.0)
        key = torch.full((BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 4.0)
        value = torch.full((BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 5.0)
        cross_key = torch.full((BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 6.0)
        cross_value = torch.full(
            (BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
            7.0,
        )

        with (
            patch.object(
                GenerativeVLM,
                "extract_query_key_value",
                return_value=(query, key, value),
            ) as query_key_value_extractor,
            patch.object(
                GenerativeVLM,
                "extract_key_value_with_rope",
                return_value=(cross_key, cross_value),
            ) as key_value_extractor,
            patch.object(
                GenerativeVLM,
                "apply_residual_feedforward",
                return_value=prefix_embeddings + 8.0,
            ) as residual_feedforward,
        ):
            decoder._run_training_forward(
                prefix_embeddings=prefix_embeddings,
                expert_hidden=expert_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                expert_action_rope=rope,
                adaptive_norm_conditioning=adaptive_norm_conditioning,
                cross_attention_mask=cross_attention_mask,
                expert_cross_attention_rope=cross_rope,
                vlm_prefix_attention_mask=vlm_prefix_attention_mask,
            )

        decoder.vlm_layers[0].forward_mock.assert_called_once()
        assert torch.equal(
            decoder.vlm_layers[0].forward_mock.call_args.kwargs["attention_mask"],
            vlm_prefix_attention_mask,
        )
        query_key_value_extractor.assert_called_once()
        residual_feedforward.assert_called_once()
        key_value_extractor.assert_called_once()
        joint_layer = decoder.expert_layers[0]
        cross_layer = decoder.expert_layers[1]
        joint_layer.forward_with_secondary_mock.assert_called_once()
        joint_call = joint_layer.forward_with_secondary_mock.call_args.kwargs
        assert torch.equal(joint_call["joint_attention_mask"], attention_mask)
        assert torch.equal(joint_call["conditioning"], adaptive_norm_conditioning)
        assert torch.equal(joint_call["conditioning_cache"].queries, query)
        assert torch.equal(joint_call["precomputed_primary_rope"][0], rope[0])
        cross_layer.forward_mock.assert_called_once()
        cross_call = cross_layer.forward_mock.call_args.kwargs
        assert torch.equal(cross_call["attention_mask"], cross_attention_mask)
        assert torch.equal(cross_call["conditioning_cache"].keys, cross_key)
        assert torch.equal(cross_call["precomputed_rope"][0], cross_rope[0])

    def test_fill_prefix_cache_skips_vlm_only_layers(
        self,
        routed_interleaved_decoder_factory: Callable[[], _TestInterleavedDecoder],
    ) -> None:
        decoder = routed_interleaved_decoder_factory()
        prefix_embeddings = torch.zeros(
            BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION
        )
        position_ids = torch.arange(PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT).repeat(
            BATCH_SIZE,
            1,
        )
        query = torch.full((BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 3.0)
        key = torch.full((BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 4.0)
        value = torch.full((BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 5.0)
        cross_key = torch.full((BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION), 6.0)
        cross_value = torch.full(
            (BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
            7.0,
        )

        with (
            patch.object(
                GenerativeVLM,
                "extract_query_key_value",
                return_value=(query, key, value),
            ),
            patch.object(
                GenerativeVLM,
                "extract_key_value_with_rope",
                return_value=(cross_key, cross_value),
            ),
        ):
            cache = decoder._fill_prefix_cache(
                prefix_embeddings=prefix_embeddings,
                position_ids=position_ids,
                prefix_attention_mask=None,
            )

        assert len(cache.layers) == 2
        assert torch.equal(cache.layers[0].queries, query)
        assert torch.equal(cache.layers[0].keys, key)
        assert torch.equal(cache.layers[1].keys, cross_key)
        assert torch.equal(cache.layers[1].values, cross_value)
        for vlm_layer in decoder.vlm_layers:
            vlm_layer.forward_mock.assert_called_once()

    def test_expert_with_cache_routes_joint_and_cross_cache_entries(
        self,
        routed_interleaved_decoder_factory: Callable[[], _TestInterleavedDecoder],
    ) -> None:
        decoder = routed_interleaved_decoder_factory()
        expert_hidden = torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION)
        attention_mask = torch.zeros(
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT,
            PREFIX_TOKEN_COUNT + ACTION_TOKEN_COUNT,
            dtype=torch.bool,
        )
        cross_attention_mask = torch.zeros(
            BATCH_SIZE,
            1,
            ACTION_TOKEN_COUNT,
            PREFIX_TOKEN_COUNT,
            dtype=torch.bool,
        )
        rope = (
            torch.ones(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
            torch.zeros(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
        )
        cross_rope = (
            torch.full((BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION), 2.0),
            torch.zeros(BATCH_SIZE, ACTION_TOKEN_COUNT, HIDDEN_DIMENSION),
        )
        joint_cache = ConditioningLayerCache(
            queries=torch.ones(BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
            keys=torch.ones(BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
            values=torch.ones(BATCH_SIZE, 1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
        )
        cross_cache = ConditioningLayerCache(
            keys=torch.ones(BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
            values=torch.ones(BATCH_SIZE, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION),
        )
        adaptive_norm_conditioning = torch.ones(BATCH_SIZE, HIDDEN_DIMENSION)

        decoder._run_expert_with_cache(
            expert_hidden=expert_hidden,
            vlm_cache=ConditioningCache(layers=[joint_cache, cross_cache]),
            attention_mask=attention_mask,
            expert_action_rope=rope,
            adaptive_norm_conditioning=adaptive_norm_conditioning,
            cross_attention_mask=cross_attention_mask,
            expert_cross_attention_rope=cross_rope,
        )

        joint_call = decoder.expert_layers[0].forward_mock.call_args.kwargs
        assert torch.equal(
            joint_call["conditioning_cache"].queries, joint_cache.queries
        )
        assert torch.equal(joint_call["conditioning"], adaptive_norm_conditioning)
        assert torch.equal(joint_call["attention_mask"], attention_mask)
        assert torch.equal(joint_call["precomputed_rope"][0], rope[0])
        cross_call = decoder.expert_layers[1].forward_mock.call_args.kwargs
        assert torch.equal(cross_call["conditioning_cache"].keys, cross_cache.keys)
        assert torch.equal(cross_call["attention_mask"], cross_attention_mask)
        assert torch.equal(cross_call["precomputed_rope"][0], cross_rope[0])


@pytest.mark.integration
class TestBaseInterleavedVLMAttentionBehaviour:
    @pytest.mark.parametrize(
        "causal_actions, expected_action_mask",
        [
            (
                True,
                torch.tensor(
                    [
                        [False, True, True],
                        [False, False, True],
                        [False, False, False],
                    ],
                    dtype=torch.bool,
                ),
            ),
            (
                False,
                torch.zeros(3, 3, dtype=torch.bool),
            ),
        ],
    )
    def test_attention_state_controls_action_visibility(
        self,
        causal_actions: bool,
        expected_action_mask: torch.Tensor,
    ) -> None:
        prefix_embeddings = torch.zeros(1, PREFIX_TOKEN_COUNT, HIDDEN_DIMENSION)
        expert_tokens = torch.zeros(1, 3, HIDDEN_DIMENSION)

        state = BaseInterleavedVLMDecoder._build_interleaved_attention_state(
            prefix_embeddings=prefix_embeddings,
            prefix_padding_mask=None,
            expert_tokens=expert_tokens,
            rotary_embedding=_FakeRotaryEmbedding(),
            causal_actions=causal_actions,
            causal_prefix_suffix_length=0,
        )

        action_mask = state.attention_mask[0, 0, :3, :3]
        action_to_prefix_mask = state.attention_mask[0, 0, :3, 3:]
        prefix_to_action_mask = state.attention_mask[0, 0, 3:, :3]
        assert torch.equal(action_mask, expected_action_mask)
        assert not action_to_prefix_mask.any()
        assert prefix_to_action_mask.all()
