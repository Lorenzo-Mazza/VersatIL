"""Tests for versatil.models.decoding.decoders.factory.pi0 module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig

from versatil.data.constants import Cameras, SampleKey
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, TimeConditioning
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.pi0 import Pi0Decoder
from versatil.models.encoding.encoders.constants import PaliGemmaModelType
from versatil.models.encoding.encoders.cross_modal.vision_language.paligemma import (
    PaliGemmaEncoder,
)
from versatil.models.layers.normalization.constants import NormalizationType

VLM_HIDDEN_DIMENSION = 32
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 1
HEAD_DIMENSION = VLM_HIDDEN_DIMENSION // NUM_ATTENTION_HEADS
EXPERT_HIDDEN_SIZE = 16
EXPERT_INTERMEDIATE_SIZE = 64
EXPERT_HEAD_DIMENSION = HEAD_DIMENSION
NUM_EXPERT_LAYERS = 2
PREDICTION_HORIZON = 4
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
PREFIX_SEQUENCE_LENGTH = 8
POSITION_DIM = 3
FEATURE_KEY = "vlm_fused_rgb_language"
PROPRIO_KEY = "robot_state_proprio"
PROPRIO_DIM = 8


def _make_tiny_paligemma_config() -> AutoConfig:
    config = AutoConfig.from_pretrained(PaliGemmaModelType.PALIGEMMA2_3B_224.value)
    config.text_config.num_hidden_layers = NUM_EXPERT_LAYERS
    config.text_config.hidden_size = VLM_HIDDEN_DIMENSION
    config.text_config.intermediate_size = VLM_HIDDEN_DIMENSION * 4
    config.text_config.num_attention_heads = NUM_ATTENTION_HEADS
    config.text_config.num_key_value_heads = NUM_KEY_VALUE_HEADS
    config.text_config.head_dim = HEAD_DIMENSION
    config.text_config.vocab_size = 1000
    config.vision_config.hidden_size = VLM_HIDDEN_DIMENSION
    config.vision_config.intermediate_size = VLM_HIDDEN_DIMENSION * 4
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = NUM_ATTENTION_HEADS
    config.vision_config.image_size = 56
    config.vision_config.patch_size = 14
    config.projection_dim = VLM_HIDDEN_DIMENSION
    config.vision_config.projection_dim = VLM_HIDDEN_DIMENSION
    return config


@pytest.fixture(scope="session")
def real_paligemma_encoder() -> PaliGemmaEncoder:
    tiny_config = _make_tiny_paligemma_config()
    with patch(
        "versatil.models.encoding.encoders.cross_modal.vision_language"
        ".generative_vlm.AutoConfig.from_pretrained",
        return_value=tiny_config,
    ):
        encoder = PaliGemmaEncoder(
            input_keys=[
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
            use_embeddings_only=True,
        )
    encoder.vlm = encoder.vlm.float()
    return encoder


@pytest.fixture
def pi0_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., Pi0Decoder]:
    def factory(
        expert_hidden_size: int = EXPERT_HIDDEN_SIZE,
        expert_intermediate_size: int = EXPERT_INTERMEDIATE_SIZE,
        expert_number_of_heads: int = NUM_ATTENTION_HEADS,
        expert_number_of_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        expert_number_of_layers: int = NUM_EXPERT_LAYERS,
        expert_head_dimension: int = EXPERT_HEAD_DIMENSION,
        prediction_horizon: int = PREDICTION_HORIZON,
        observation_horizon: int = OBSERVATION_HORIZON,
        position_dim: int = POSITION_DIM,
        time_conditioning: str = TimeConditioning.CONCAT_MLP.value,
        proprioceptive_feature_key: str | None = None,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        dropout: float = 0.0,
        input_keys: list[str] | None = None,
    ) -> Pi0Decoder:
        if input_keys is None:
            input_keys = [FEATURE_KEY]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            key: action_head_factory(input_dim=expert_hidden_size)
            for key in action_space.actions_metadata
        }
        return Pi0Decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            expert_hidden_size=expert_hidden_size,
            expert_intermediate_size=expert_intermediate_size,
            expert_number_of_heads=expert_number_of_heads,
            expert_number_of_key_value_heads=expert_number_of_key_value_heads,
            expert_number_of_layers=expert_number_of_layers,
            expert_head_dimension=expert_head_dimension,
            time_conditioning=time_conditioning,
            proprioceptive_feature_key=proprioceptive_feature_key,
            normalization_type=normalization_type,
            dropout=dropout,
        )

    return factory


@pytest.fixture
def initialized_decoder_factory(
    pi0_decoder_factory: Callable[..., Pi0Decoder],
    real_paligemma_encoder: PaliGemmaEncoder,
) -> Callable[..., Pi0Decoder]:
    def factory(
        expert_hidden_size: int = EXPERT_HIDDEN_SIZE,
        expert_intermediate_size: int = EXPERT_INTERMEDIATE_SIZE,
        expert_number_of_layers: int = NUM_EXPERT_LAYERS,
        expert_head_dimension: int = EXPERT_HEAD_DIMENSION,
        prediction_horizon: int = PREDICTION_HORIZON,
        time_conditioning: str = TimeConditioning.CONCAT_MLP.value,
        proprioceptive_feature_key: str | None = None,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        dropout: float = 0.0,
    ) -> Pi0Decoder:
        decoder = pi0_decoder_factory(
            expert_hidden_size=expert_hidden_size,
            expert_intermediate_size=expert_intermediate_size,
            expert_number_of_layers=expert_number_of_layers,
            expert_head_dimension=expert_head_dimension,
            prediction_horizon=prediction_horizon,
            time_conditioning=time_conditioning,
            proprioceptive_feature_key=proprioceptive_feature_key,
            normalization_type=normalization_type,
            dropout=dropout,
        )
        decoder.set_backbone(
            vlm_layers=real_paligemma_encoder.get_backbone_layers(),
            rotary_emb=real_paligemma_encoder.get_rotary_embedding(),
            vlm_hidden_dimension=real_paligemma_encoder.get_backbone_hidden_dim(),
            vlm_text_config=real_paligemma_encoder.get_text_config(),
        )
        return decoder

    return factory


@pytest.fixture
def prefix_features_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = PREFIX_SEQUENCE_LENGTH,
        hidden_dimension: int = VLM_HIDDEN_DIMENSION,
        include_timestep: bool = True,
        include_proprioceptive: bool = False,
        proprioceptive_dimension: int = PROPRIO_DIM,
    ) -> dict[str, torch.Tensor]:
        features = {
            FEATURE_KEY: torch.from_numpy(
                rng.standard_normal(
                    (batch_size, sequence_length, hidden_dimension)
                ).astype(np.float32)
            ),
        }
        if include_timestep:
            features[DecoderOutputKey.TIMESTEP.value] = torch.from_numpy(
                rng.uniform(low=0.0, high=1.0, size=(batch_size,)).astype(np.float32)
            )
        if include_proprioceptive:
            features[PROPRIO_KEY] = torch.from_numpy(
                rng.standard_normal((batch_size, proprioceptive_dimension)).astype(
                    np.float32
                )
            )
        return features

    return factory


class TestPi0DecoderInitialization:
    def test_inherits_from_action_decoder(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = pi0_decoder_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("expert_hidden_size", [16, 32])
    @pytest.mark.parametrize("expert_number_of_layers", [2, 4])
    @pytest.mark.parametrize(
        "time_conditioning",
        [TimeConditioning.CONCAT_MLP.value, TimeConditioning.ADANORM.value],
    )
    def test_stores_configuration(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        expert_hidden_size: int,
        expert_number_of_layers: int,
        time_conditioning: str,
    ):
        decoder = pi0_decoder_factory(
            expert_hidden_size=expert_hidden_size,
            expert_number_of_layers=expert_number_of_layers,
            time_conditioning=time_conditioning,
        )
        assert decoder.expert_hidden_size == expert_hidden_size
        assert decoder.expert_number_of_layers == expert_number_of_layers
        assert decoder.time_conditioning == time_conditioning

    def test_decoder_input_requires_actions(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = pi0_decoder_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_decoder_input_requires_vlm_backbone(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = pi0_decoder_factory()
        assert decoder.decoder_input.requires_vlm_backbone is True

    def test_forward_raises_before_set_backbone(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = pi0_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("set_backbone() must be called before forward()."),
        ):
            decoder(features=features, actions=actions)

    @pytest.mark.parametrize(
        "time_conditioning, proprioceptive_feature_key",
        [
            (TimeConditioning.CONCAT_MLP.value, PROPRIO_KEY),
            (TimeConditioning.CONCAT_MLP.value, None),
            (TimeConditioning.ADANORM.value, PROPRIO_KEY),
        ],
    )
    def test_proprioceptive_projection_deferred_until_set_backbone(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        time_conditioning: str,
        proprioceptive_feature_key: str | None,
    ):
        # Before set_backbone, projection is always None regardless of config
        decoder = pi0_decoder_factory(
            time_conditioning=time_conditioning,
            proprioceptive_feature_key=proprioceptive_feature_key,
        )
        assert decoder.proprioceptive_projection is None

    def test_raises_on_unknown_time_conditioning(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unknown time_conditioning: invalid_mode. "
                f"Use {[m.value for m in TimeConditioning]}"
            ),
        ):
            pi0_decoder_factory(time_conditioning="invalid_mode")

    def test_set_backbone_accepts_adanorm_with_plain_norm(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = pi0_decoder_factory(
            time_conditioning=TimeConditioning.ADANORM.value,
            normalization_type=NormalizationType.RMS_NORM.value,
        )
        mock_vlm_layers = nn.ModuleList(
            [MagicMock(spec=nn.Module) for _ in range(NUM_EXPERT_LAYERS)]
        )
        mock_rotary_emb = MagicMock(spec=nn.Module)
        mock_text_config = MagicMock(spec=PretrainedConfig)
        decoder.set_backbone(
            vlm_layers=mock_vlm_layers,
            rotary_emb=mock_rotary_emb,
            vlm_hidden_dimension=VLM_HIDDEN_DIMENSION,
            vlm_text_config=mock_text_config,
        )
        assert decoder.expert_layers is not None


@pytest.mark.integration
class TestPi0DecoderSetBackbone:
    def test_expert_layer_count_matches_vlm(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = initialized_decoder_factory()
        assert len(decoder.expert_layers) == NUM_EXPERT_LAYERS

    @pytest.mark.parametrize(
        "time_conditioning, normalization_type, proprioceptive_feature_key, expect_projection",
        [
            (
                TimeConditioning.CONCAT_MLP.value,
                NormalizationType.RMS_NORM.value,
                PROPRIO_KEY,
                True,
            ),
            (
                TimeConditioning.CONCAT_MLP.value,
                NormalizationType.RMS_NORM.value,
                None,
                False,
            ),
            (
                TimeConditioning.ADANORM.value,
                NormalizationType.RMS_NORM.value,
                PROPRIO_KEY,
                False,
            ),
        ],
    )
    def test_proprioceptive_projection_created_only_for_concat_mlp_with_key(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        time_conditioning: str,
        normalization_type: str,
        proprioceptive_feature_key: str | None,
        expect_projection: bool,
    ):
        decoder = initialized_decoder_factory(
            time_conditioning=time_conditioning,
            normalization_type=normalization_type,
            proprioceptive_feature_key=proprioceptive_feature_key,
        )
        has_projection = decoder.proprioceptive_projection is not None
        assert has_projection == expect_projection

    def test_raises_when_vlm_layer_count_mismatches_expert_count(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        real_paligemma_encoder: PaliGemmaEncoder,
    ):
        decoder = pi0_decoder_factory(expert_number_of_layers=99)
        vlm_layer_count = len(real_paligemma_encoder.get_backbone_layers())
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Pi0 requires equal VLM ({vlm_layer_count}) and expert "
                f"(99) layer counts."
            ),
        ):
            decoder.set_backbone(
                vlm_layers=real_paligemma_encoder.get_backbone_layers(),
                rotary_emb=real_paligemma_encoder.get_rotary_embedding(),
                vlm_hidden_dimension=real_paligemma_encoder.get_backbone_hidden_dim(),
                vlm_text_config=real_paligemma_encoder.get_text_config(),
            )


@pytest.mark.integration
class TestPi0DecoderForward:
    def test_raises_without_actions(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Pi0Decoder requires actions during forward "
                "(noisy actions for denoising)."
            ),
        ):
            decoder(features=features, actions=None)

    def test_raises_without_timestep(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory(include_timestep=False)
        actions = noisy_actions_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features "
                "dict. The algorithm should inject timesteps into features."
            ),
        ):
            decoder(features=features, actions=actions)

    def test_output_shape_and_keys(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        assert set(outputs.keys()) == set(decoder.action_heads.keys())
        for action_key, output_tensor in outputs.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                decoder.action_heads[action_key].output_dim,
            )

    def test_output_cast_to_float32(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # Pi0 explicitly casts output to float32 for mixed-precision compatibility
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        for output_tensor in outputs.values():
            assert output_tensor.dtype == torch.float32

    def test_proprioceptive_feature_changes_output(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory(
            proprioceptive_feature_key=PROPRIO_KEY,
        )
        decoder.eval()
        features_without = prefix_features_factory()
        features_with = {
            key: tensor.clone() for key, tensor in features_without.items()
        }
        features_with[PROPRIO_KEY] = torch.from_numpy(
            np.ones((BATCH_SIZE, PROPRIO_DIM), dtype=np.float32)
        )
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_without = decoder(features=features_without, actions=actions)
            output_with = decoder(features=features_with, actions=actions)
        for action_key in actions:
            assert not torch.allclose(
                output_without[action_key], output_with[action_key]
            )


@pytest.mark.integration
class TestPi0DecoderBehavior:
    def test_expert_layers_receive_gradients_while_vlm_frozen(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        # Freeze VLM, then restore after test to avoid polluting session-scoped encoder
        original_requires_grad = {
            parameter: parameter.requires_grad
            for parameter in decoder.vlm_layers.parameters()
        }
        for parameter in decoder.vlm_layers.parameters():
            parameter.requires_grad = False
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        loss = sum(tensor.sum() for tensor in outputs.values())
        loss.backward()
        expert_has_grad = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in decoder.expert_layers.parameters()
            if parameter.requires_grad
        )
        assert expert_has_grad
        vlm_has_grad = any(
            parameter.grad is not None for parameter in decoder.vlm_layers.parameters()
        )
        assert not vlm_has_grad
        for parameter, grad_flag in original_requires_grad.items():
            parameter.requires_grad = grad_flag

    def test_concat_mlp_timestep_sensitivity(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory(
            time_conditioning=TimeConditioning.CONCAT_MLP.value,
        )
        decoder.eval()
        features_low = prefix_features_factory()
        features_low[DecoderOutputKey.TIMESTEP.value] = torch.full((BATCH_SIZE,), 0.01)
        features_high = {key: tensor.clone() for key, tensor in features_low.items()}
        features_high[DecoderOutputKey.TIMESTEP.value] = torch.full((BATCH_SIZE,), 0.99)
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_low = decoder(features=features_low, actions=actions)
            output_high = decoder(features=features_high, actions=actions)
        for action_key in actions:
            assert not torch.allclose(output_low[action_key], output_high[action_key])

    def test_adanorm_timestep_has_no_effect_at_init(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # AdaLN-Zero gating is zero-initialized — conditioning has no effect
        # at init. This is by design: the network starts as identity.
        decoder = initialized_decoder_factory(
            time_conditioning=TimeConditioning.ADANORM.value,
            normalization_type=NormalizationType.RMS_NORM.value,
        )
        decoder.eval()
        features_low = prefix_features_factory()
        features_low[DecoderOutputKey.TIMESTEP.value] = torch.full((BATCH_SIZE,), 0.01)
        features_high = {key: tensor.clone() for key, tensor in features_low.items()}
        features_high[DecoderOutputKey.TIMESTEP.value] = torch.full((BATCH_SIZE,), 0.99)
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_low = decoder(features=features_low, actions=actions)
            output_high = decoder(features=features_high, actions=actions)
        for action_key in actions:
            torch.testing.assert_close(output_low[action_key], output_high[action_key])

    def test_different_prefix_features_produce_different_outputs(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        features_a = prefix_features_factory()
        features_b = prefix_features_factory()
        features_b[DecoderOutputKey.TIMESTEP.value] = features_a[
            DecoderOutputKey.TIMESTEP.value
        ].clone()
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_a = decoder(features=features_a, actions=actions)
            output_b = decoder(features=features_b, actions=actions)
        for action_key in actions:
            assert not torch.allclose(output_a[action_key], output_b[action_key])

    def test_expert_normalization_changes_hidden_states(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # Verify expert_final_normalization affects the output by comparing
        # against identity — if normalization were missing, output would differ
        decoder = initialized_decoder_factory()
        decoder.eval()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_with_norm = decoder(features=features, actions=actions)
        original_norm = decoder.expert_final_normalization
        decoder.expert_final_normalization = torch.nn.Identity()
        with torch.no_grad():
            output_without_norm = decoder(features=features, actions=actions)
        decoder.expert_final_normalization = original_norm
        for action_key in actions:
            assert not torch.allclose(
                output_with_norm[action_key], output_without_norm[action_key]
            )


@pytest.mark.integration
class TestPi0DecoderCaching:
    def test_cache_populated_on_first_forward(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        decoder.enable_encoder_cache()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        with torch.no_grad():
            decoder(features=features, actions=actions)
        assert len(decoder._prefix_cache) == NUM_EXPERT_LAYERS

    def test_disable_cache_allows_new_prefix_to_take_effect(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        decoder.enable_encoder_cache()
        features_a = prefix_features_factory()
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_a = decoder(features=features_a, actions=actions)
        # With cache, different prefix is ignored
        features_b = prefix_features_factory()
        features_b[DecoderOutputKey.TIMESTEP.value] = features_a[
            DecoderOutputKey.TIMESTEP.value
        ].clone()
        with torch.no_grad():
            output_stale = decoder(features=features_b, actions=actions)
        for action_key in actions:
            torch.testing.assert_close(output_a[action_key], output_stale[action_key])
        # After disable + re-enable, new prefix takes effect
        decoder.disable_encoder_cache()
        decoder.enable_encoder_cache()
        with torch.no_grad():
            output_fresh = decoder(features=features_b, actions=actions)
        for action_key in actions:
            assert not torch.allclose(output_a[action_key], output_fresh[action_key])

    def test_cached_forwards_are_self_consistent(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        decoder.enable_encoder_cache()
        actions = noisy_actions_factory()
        base_features = prefix_features_factory()
        features_first = {key: tensor.clone() for key, tensor in base_features.items()}
        with torch.no_grad():
            output_first = decoder(features=features_first, actions=actions)
        features_second = {key: tensor.clone() for key, tensor in base_features.items()}
        with torch.no_grad():
            output_second = decoder(features=features_second, actions=actions)
        for action_key in actions:
            torch.testing.assert_close(
                output_first[action_key],
                output_second[action_key],
            )

    def test_cached_forward_with_proprioceptive_feature(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory(
            proprioceptive_feature_key=PROPRIO_KEY,
        )
        decoder.eval()
        decoder.enable_encoder_cache()
        features = prefix_features_factory(include_proprioceptive=True)
        actions = noisy_actions_factory()
        with torch.no_grad():
            output = decoder(features=features, actions=actions)
        for action_key, output_tensor in output.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                decoder.action_heads[action_key].output_dim,
            )
