"""Tests for versatil.models.decoding.decoders.factory.smolvla module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from versatil.data.constants import Cameras, SampleKey
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.smolvla import (
    SmolVLADecoder,
    SmolVLALayerType,
)
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    SmolVLMModelType,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm import (
    SmolVLMEncoder,
)
from versatil.models.layers.normalization.constants import NormalizationType

VLM_HIDDEN_DIMENSION = 32
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 1
HEAD_DIMENSION = VLM_HIDDEN_DIMENSION // NUM_ATTENTION_HEADS
EXPERT_WIDTH_MULTIPLIER = 0.5
EXPERT_HIDDEN_SIZE = int(VLM_HIDDEN_DIMENSION * EXPERT_WIDTH_MULTIPLIER)
PREDICTION_HORIZON = 4
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
PREFIX_SEQUENCE_LENGTH = 8
POSITION_DIM = 3
NUM_VLM_LAYERS = 4
SELF_ATTENTION_EVERY_N_LAYERS = 2
FEATURE_KEY = "vlm_fused_rgb_language"
PADDING_MASK_KEY = f"{FEATURE_KEY}_{EncoderOutputKeys.PADDING_MASK.value}"


def _make_tiny_smolvlm_config() -> AutoConfig:
    """Create a real but tiny SmolVLM config for integration tests."""
    config = AutoConfig.from_pretrained(SmolVLMModelType.SMOLVLM_256M.value)
    config.text_config.num_hidden_layers = NUM_VLM_LAYERS
    config.text_config.hidden_size = VLM_HIDDEN_DIMENSION
    config.text_config.intermediate_size = VLM_HIDDEN_DIMENSION * 4
    config.text_config.num_attention_heads = NUM_ATTENTION_HEADS
    config.text_config.num_key_value_heads = NUM_KEY_VALUE_HEADS
    config.vision_config.hidden_size = VLM_HIDDEN_DIMENSION
    config.vision_config.intermediate_size = VLM_HIDDEN_DIMENSION * 4
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = NUM_ATTENTION_HEADS
    config.vision_config.image_size = 56
    config.vision_config.patch_size = 14
    return config


@pytest.fixture(scope="session")
def real_smolvlm_encoder() -> SmolVLMEncoder:
    """Session-scoped real tiny SmolVLM encoder for integration tests."""
    tiny_config = _make_tiny_smolvlm_config()
    with patch(
        "versatil.models.encoding.encoders.cross_modal.vision_language"
        ".generative_vlm.AutoConfig.from_pretrained",
        return_value=tiny_config,
    ):
        encoder = SmolVLMEncoder(
            input_keys=[
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            model_name=SmolVLMModelType.SMOLVLM_256M.value,
            use_embeddings_only=True,
        )
    encoder.vlm = encoder.vlm.float()
    return encoder


@pytest.fixture
def smolvla_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., SmolVLADecoder]:
    """Factory for SmolVLADecoder without backbone (unit tests)."""

    def factory(
        expert_width_multiplier: float = EXPERT_WIDTH_MULTIPLIER,
        num_expert_layers: int = -1,
        num_vlm_layers: int = NUM_VLM_LAYERS,
        self_attention_every_n_layers: int = SELF_ATTENTION_EVERY_N_LAYERS,
        prediction_horizon: int = PREDICTION_HORIZON,
        observation_horizon: int = OBSERVATION_HORIZON,
        position_dim: int = POSITION_DIM,
        proprioceptive_feature_key: str | None = None,
        freeze_vlm: bool = True,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        dropout: float = 0.0,
        input_keys: list[str] | None = None,
    ) -> SmolVLADecoder:
        if input_keys is None:
            input_keys = [FEATURE_KEY]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            key: action_head_factory(input_dim=EXPERT_HIDDEN_SIZE)
            for key in action_space.actions_metadata
        }
        return SmolVLADecoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            expert_width_multiplier=expert_width_multiplier,
            num_expert_layers=num_expert_layers,
            num_vlm_layers=num_vlm_layers,
            self_attention_every_n_layers=self_attention_every_n_layers,
            proprioceptive_feature_key=proprioceptive_feature_key,
            freeze_vlm=freeze_vlm,
            normalization_type=normalization_type,
            dropout=dropout,
        )

    return factory


@pytest.fixture
def initialized_decoder_factory(
    smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    real_smolvlm_encoder: SmolVLMEncoder,
) -> Callable[..., SmolVLADecoder]:
    """Factory that returns a SmolVLADecoder with real VLM backbone wired."""

    def factory(
        expert_width_multiplier: float = EXPERT_WIDTH_MULTIPLIER,
        num_expert_layers: int = -1,
        num_vlm_layers: int = NUM_VLM_LAYERS,
        self_attention_every_n_layers: int = SELF_ATTENTION_EVERY_N_LAYERS,
        prediction_horizon: int = PREDICTION_HORIZON,
        proprioceptive_feature_key: str | None = None,
        freeze_vlm: bool = True,
        dropout: float = 0.0,
    ) -> SmolVLADecoder:
        decoder = smolvla_decoder_factory(
            expert_width_multiplier=expert_width_multiplier,
            num_expert_layers=num_expert_layers,
            num_vlm_layers=num_vlm_layers,
            self_attention_every_n_layers=self_attention_every_n_layers,
            prediction_horizon=prediction_horizon,
            proprioceptive_feature_key=proprioceptive_feature_key,
            freeze_vlm=freeze_vlm,
            dropout=dropout,
        )
        decoder.set_backbone(
            vlm_layers=real_smolvlm_encoder.get_backbone_layers(),
            rotary_emb=real_smolvlm_encoder.get_rotary_embedding(),
            vlm_hidden_dimension=real_smolvlm_encoder.get_backbone_hidden_dim(),
            vlm_text_config=real_smolvlm_encoder.get_text_config(),
        )
        return decoder

    return factory


@pytest.fixture
def prefix_features_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for prefix feature dicts with optional timestep and padding mask."""

    def factory(
        batch_size: int = BATCH_SIZE,
        sequence_length: int = PREFIX_SEQUENCE_LENGTH,
        hidden_dimension: int = VLM_HIDDEN_DIMENSION,
        include_timestep: bool = True,
        include_padding_mask: bool = False,
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
        if include_padding_mask:
            mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
            mask[:, -2:] = True
            features[PADDING_MASK_KEY] = mask
        return features

    return factory


class TestSmolVLADecoderInitialization:
    def test_inherits_from_action_decoder(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = smolvla_decoder_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("expert_width_multiplier", [0.5, 0.75])
    @pytest.mark.parametrize("num_vlm_layers", [4, 16])
    @pytest.mark.parametrize("self_attention_every_n_layers", [0, 2])
    def test_stores_configuration(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
        expert_width_multiplier: float,
        num_vlm_layers: int,
        self_attention_every_n_layers: int,
    ):
        decoder = smolvla_decoder_factory(
            expert_width_multiplier=expert_width_multiplier,
            num_vlm_layers=num_vlm_layers,
            self_attention_every_n_layers=self_attention_every_n_layers,
        )
        assert decoder.expert_width_multiplier == expert_width_multiplier
        assert decoder.num_vlm_layers == num_vlm_layers
        assert decoder.self_attention_every_n_layers == self_attention_every_n_layers

    def test_decoder_input_requires_actions(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = smolvla_decoder_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_decoder_input_requires_vlm_backbone(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = smolvla_decoder_factory()
        assert decoder.decoder_input.requires_vlm_backbone is True

    def test_layers_none_before_set_backbone(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = smolvla_decoder_factory()
        assert decoder.vlm_layers is None
        assert decoder.expert_layers is None
        assert decoder._layer_types is None

    def test_caching_initially_disabled(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = smolvla_decoder_factory()
        assert decoder._encoder_cache_enabled is False
        assert decoder._prefix_cache is None

    def test_raises_without_set_backbone(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = smolvla_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("set_backbone() must be called before forward()."),
        ):
            decoder(features=features, actions=actions)


class TestGetIntermediateSize:
    @pytest.mark.parametrize(
        "hidden_dimension, expected",
        [
            (256, 768),
            (512, 1536),
        ],
    )
    def test_rounds_to_multiple_of_256(self, hidden_dimension: int, expected: int):
        result = SmolVLADecoder._get_intermediate_size(hidden_dimension)
        assert result == expected
        assert result % 256 == 0


@pytest.mark.integration
class TestSmolVLADecoderSetBackbone:
    def test_action_input_projection_dimensions(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = initialized_decoder_factory()
        assert decoder.action_input_projection.in_features == POSITION_DIM
        assert decoder.action_input_projection.out_features == EXPERT_HIDDEN_SIZE

    def test_action_output_projection_dimensions(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = initialized_decoder_factory()
        assert decoder.action_output_projection.in_features == EXPERT_HIDDEN_SIZE
        assert decoder.action_output_projection.out_features == POSITION_DIM

    @pytest.mark.parametrize(
        "self_attention_every_n_layers, expected_types",
        [
            (
                2,
                [
                    SmolVLALayerType.SELF_ATTENTION.value,
                    SmolVLALayerType.CROSS_ATTENTION.value,
                    SmolVLALayerType.SELF_ATTENTION.value,
                    SmolVLALayerType.CROSS_ATTENTION.value,
                ],
            ),
            (
                0,
                [
                    SmolVLALayerType.CROSS_ATTENTION.value,
                    SmolVLALayerType.CROSS_ATTENTION.value,
                    SmolVLALayerType.CROSS_ATTENTION.value,
                    SmolVLALayerType.CROSS_ATTENTION.value,
                ],
            ),
        ],
    )
    def test_layer_type_interleaving_pattern(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        self_attention_every_n_layers: int,
        expected_types: list[str],
    ):
        decoder = initialized_decoder_factory(
            self_attention_every_n_layers=self_attention_every_n_layers,
        )
        assert decoder._layer_types == expected_types

    def test_vlm_parameters_frozen_when_freeze_vlm_true(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = initialized_decoder_factory(freeze_vlm=True)
        for parameter in decoder.vlm_layers.parameters():
            assert parameter.requires_grad is False

    def test_vlm_parameters_trainable_when_freeze_vlm_false(
        self,
        smolvla_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        # Use fresh VLM layers to avoid mutating the session-scoped encoder
        tiny_config = _make_tiny_smolvlm_config()
        with patch(
            "versatil.models.encoding.encoders.cross_modal.vision_language"
            ".generative_vlm.AutoConfig.from_pretrained",
            return_value=tiny_config,
        ):
            fresh_encoder = SmolVLMEncoder(
                input_keys=[
                    Cameras.LEFT.value,
                    SampleKey.TOKENIZED_OBSERVATIONS.value,
                ],
                pretrained=False,
                frozen=False,
                model_name=SmolVLMModelType.SMOLVLM_256M.value,
                use_embeddings_only=True,
            )
        fresh_encoder.vlm = fresh_encoder.vlm.float()
        decoder = smolvla_decoder_factory(freeze_vlm=False)
        decoder.set_backbone(
            vlm_layers=fresh_encoder.get_backbone_layers(),
            rotary_emb=fresh_encoder.get_rotary_embedding(),
            vlm_hidden_dimension=fresh_encoder.get_backbone_hidden_dim(),
            vlm_text_config=fresh_encoder.get_text_config(),
        )
        for parameter in decoder.vlm_layers.parameters():
            assert parameter.requires_grad is True

    def test_truncates_vlm_layers(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = initialized_decoder_factory(num_vlm_layers=2)
        assert len(decoder.vlm_layers) == 2

    def test_uses_all_layers_when_num_vlm_layers_negative(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = initialized_decoder_factory(num_vlm_layers=-1)
        assert len(decoder.vlm_layers) == NUM_VLM_LAYERS

    def test_creates_proprioceptive_projection_when_key_set(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
    ):
        decoder = initialized_decoder_factory(
            proprioceptive_feature_key="robot_state_proprio",
        )
        assert hasattr(decoder, "proprioceptive_projection")
        assert decoder.proprioceptive_projection is not None


@pytest.mark.integration
class TestSmolVLADecoderForward:
    def test_raises_without_actions(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "SmolVLADecoder requires actions during forward "
                "(noisy actions for denoising)."
            ),
        ):
            decoder(features=features, actions=None)

    def test_raises_without_timestep(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
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

    def test_output_keys_match_action_heads(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        assert set(outputs.keys()) == set(decoder.action_heads.keys())

    def test_output_shape(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        for action_key, output_tensor in outputs.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                decoder.action_heads[action_key].output_dim,
            )

    def test_padding_mask_changes_output(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        actions = noisy_actions_factory()
        features_no_pad = prefix_features_factory(include_padding_mask=False)
        features_with_pad = {
            key: tensor.clone() for key, tensor in features_no_pad.items()
        }
        pad_mask = torch.zeros(BATCH_SIZE, PREFIX_SEQUENCE_LENGTH, dtype=torch.bool)
        pad_mask[:, -2:] = True
        features_with_pad[PADDING_MASK_KEY] = pad_mask
        with torch.no_grad():
            output_no_pad = decoder(features=features_no_pad, actions=actions)
            output_with_pad = decoder(features=features_with_pad, actions=actions)
        for action_key in actions:
            assert not torch.allclose(
                output_no_pad[action_key], output_with_pad[action_key]
            )

    def test_forward_with_proprioceptive_feature(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ):
        proprio_key = "robot_state_proprio"
        decoder = initialized_decoder_factory(
            proprioceptive_feature_key=proprio_key,
        )
        features = prefix_features_factory()
        features[proprio_key] = torch.from_numpy(
            rng.standard_normal((BATCH_SIZE, 8)).astype(np.float32)
        )
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        for action_key, output_tensor in outputs.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                decoder.action_heads[action_key].output_dim,
            )


@pytest.mark.integration
class TestSmolVLADecoderBehavior:
    def test_expert_layers_receive_gradients_while_vlm_frozen(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory(freeze_vlm=True)
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

    def test_different_timesteps_produce_different_outputs(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
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

    def test_different_prefix_features_produce_different_outputs(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
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


@pytest.mark.integration
class TestSmolVLADecoderCaching:
    def test_disable_cache_clears_state(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
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
        assert decoder._prefix_cache is not None
        decoder.disable_encoder_cache()
        assert decoder._encoder_cache_enabled is False
        assert decoder._prefix_cache is None

    def test_cache_populated_on_first_forward(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
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
        assert decoder._prefix_cache is not None
        assert len(decoder._prefix_cache) > 0

    def test_cache_not_recomputed_on_second_forward(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # After the first cached forward populates the cache, a second forward
        # with different features should still use the old cache (prefix is
        # not re-encoded). Verify by passing different prefix features and
        # checking the output is unchanged — proving the new features were ignored.
        decoder = initialized_decoder_factory()
        decoder.eval()
        decoder.enable_encoder_cache()
        actions = noisy_actions_factory()
        features_first = prefix_features_factory()
        with torch.no_grad():
            output_first = decoder(features=features_first, actions=actions)
        features_different_prefix = prefix_features_factory()
        features_different_prefix[DecoderOutputKey.TIMESTEP.value] = features_first[
            DecoderOutputKey.TIMESTEP.value
        ].clone()
        with torch.no_grad():
            output_with_stale_cache = decoder(
                features=features_different_prefix, actions=actions
            )
        for action_key in actions:
            torch.testing.assert_close(
                output_first[action_key],
                output_with_stale_cache[action_key],
            )

    def test_cached_forward_with_padding_mask(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        decoder.enable_encoder_cache()
        features = prefix_features_factory(include_padding_mask=True)
        actions = noisy_actions_factory()
        with torch.no_grad():
            output = decoder(features=features, actions=actions)
        for action_key, output_tensor in output.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                decoder.action_heads[action_key].output_dim,
            )

    def test_cached_forwards_are_self_consistent(
        self,
        initialized_decoder_factory: Callable[..., SmolVLADecoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # Cached path runs VLM as pure self-attention (no expert in joint
        # layers), so it differs from training forward. But repeated cached
        # calls with the same prefix must produce identical results.
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
