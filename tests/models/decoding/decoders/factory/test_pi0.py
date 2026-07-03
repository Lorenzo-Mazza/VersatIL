"""Tests for versatil.models.decoding.decoders.factory.pi0 module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import (
    Gemma2Config,
    PaliGemmaConfig,
    SiglipVisionConfig,
)

from versatil.data.constants import Cameras, SampleKey
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, TimeConditioning
from versatil.models.decoding.decoders import interleaved_vlm as interleaved_vlm_module
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.pi0 import Pi0Decoder
from versatil.models.decoding.generative_language_models.constants import (
    PaliGemmaModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.paligemma import (
    PaliGemmaVLM,
)
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
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
PADDING_MASK_KEY = f"{FEATURE_KEY}_{EncoderOutputKeys.PADDING_MASK.value}"
PROPRIO_KEY = "robot_state_proprio"
PROPRIO_DIM = 8


def _make_tiny_paligemma_config() -> PaliGemmaConfig:
    text_config = Gemma2Config(
        num_hidden_layers=NUM_EXPERT_LAYERS,
        hidden_size=VLM_HIDDEN_DIMENSION,
        intermediate_size=VLM_HIDDEN_DIMENSION * 4,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIMENSION,
        vocab_size=1000,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=VLM_HIDDEN_DIMENSION,
        intermediate_size=VLM_HIDDEN_DIMENSION * 4,
        num_hidden_layers=1,
        num_attention_heads=NUM_ATTENTION_HEADS,
        image_size=56,
        patch_size=14,
    )
    config = PaliGemmaConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=VLM_HIDDEN_DIMENSION,
    )
    config.vision_config.num_image_tokens = 16
    config.vision_config.projection_dim = VLM_HIDDEN_DIMENSION
    return config


@pytest.fixture(scope="session")
def real_paligemma_backbone() -> PaliGemmaVLM:
    tiny_config = _make_tiny_paligemma_config()
    with patch(
        "versatil.models.decoding.generative_language_models.vision_language"
        ".huggingface.AutoConfig.from_pretrained",
        return_value=tiny_config,
    ):
        backbone = PaliGemmaVLM(
            input_keys=[
                Cameras.LEFT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
        )
    backbone.vlm = backbone.vlm.float()
    return backbone


@pytest.fixture
def pi0_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
    real_paligemma_backbone: PaliGemmaVLM,
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
        vlm_backbone: PaliGemmaVLM | None = None,
        use_default_vlm_backbone: bool = True,
    ) -> Pi0Decoder:
        if input_keys is None:
            input_keys = [FEATURE_KEY]
        if vlm_backbone is None and use_default_vlm_backbone:
            vlm_backbone = real_paligemma_backbone
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            key: action_head_factory(input_dimension=expert_hidden_size)
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
            vlm_backbone=vlm_backbone,
        )

    return factory


@pytest.fixture
def initialized_decoder_factory(
    pi0_decoder_factory: Callable[..., Pi0Decoder],
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

        def build_prefix(
            features: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            return features[FEATURE_KEY], features.get(PADDING_MASK_KEY)

        decoder._build_prefix = MagicMock(
            spec=decoder._build_prefix,
            side_effect=build_prefix,
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
        include_padding_mask: bool = False,
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
        if include_padding_mask:
            mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
            mask[:, -2:] = True
            features[PADDING_MASK_KEY] = mask
        return features

    return factory


@pytest.fixture
def raw_vlm_features_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        batch_size: int = BATCH_SIZE,
        image_size: int = 56,
        token_length: int = 4,
        include_timestep: bool = True,
    ) -> dict[str, torch.Tensor]:
        features = {
            Cameras.LEFT.value: torch.from_numpy(
                rng.standard_normal((batch_size, 3, image_size, image_size)).astype(
                    np.float32
                )
            ),
            SampleKey.TOKENIZED_OBSERVATIONS.value: torch.from_numpy(
                rng.integers(low=0, high=100, size=(batch_size, token_length)).astype(
                    np.int64
                )
            ),
            SampleKey.IS_PAD_OBSERVATION.value: torch.zeros(
                batch_size, token_length, dtype=torch.bool
            ),
        }
        if include_timestep:
            features[DecoderOutputKey.TIMESTEP.value] = torch.from_numpy(
                rng.uniform(low=0.0, high=1.0, size=(batch_size,)).astype(np.float32)
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
    @pytest.mark.parametrize("expert_number_of_layers", [NUM_EXPERT_LAYERS])
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

    def test_requires_vlm_backbone(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("Pi0Decoder requires a vlm_backbone."),
        ):
            pi0_decoder_factory(use_default_vlm_backbone=False)

    def test_vlm_backbone_needs_raw_observations(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        real_paligemma_backbone: PaliGemmaVLM,
    ):
        decoder = pi0_decoder_factory(
            input_keys=[],
            vlm_backbone=real_paligemma_backbone,
        )
        assert decoder.decoder_input.needs_raw_observations is True
        assert decoder.vlm_backbone is real_paligemma_backbone

    @pytest.mark.parametrize(
        "time_conditioning, proprioceptive_feature_key",
        [
            (TimeConditioning.CONCAT_MLP.value, PROPRIO_KEY),
            (TimeConditioning.CONCAT_MLP.value, None),
            (TimeConditioning.ADANORM.value, PROPRIO_KEY),
            (TimeConditioning.ADANORM.value, None),
        ],
    )
    def test_proprioceptive_projection_created_from_owned_backbone(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        time_conditioning: str,
        proprioceptive_feature_key: str | None,
    ):
        decoder = pi0_decoder_factory(
            time_conditioning=time_conditioning,
            proprioceptive_feature_key=proprioceptive_feature_key,
        )
        # Proprio tokens are appended to the VLM prefix, independent of how
        # the timestep is fused into the expert, so the projection must exist
        # for every time_conditioning mode when the key is configured.
        expected_projection = proprioceptive_feature_key is not None
        assert (decoder.proprioceptive_projection is not None) == expected_projection

    def test_fill_prefix_cache_raises_when_rotary_embedding_missing(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = pi0_decoder_factory()
        decoder.vlm_rotary_embedding = None
        prefix = torch.zeros(BATCH_SIZE, PREFIX_SEQUENCE_LENGTH, EXPERT_HIDDEN_SIZE)
        position_ids = torch.zeros(BATCH_SIZE, PREFIX_SEQUENCE_LENGTH, dtype=torch.long)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "VLM rotary embedding not set. build_action_expert() must be called."
            ),
        ):
            decoder._fill_prefix_cache(
                prefix_embeddings=prefix, position_ids=position_ids
            )

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

    def test_adanorm_initializes_expert_layers(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
    ):
        decoder = pi0_decoder_factory(
            time_conditioning=TimeConditioning.ADANORM.value,
            normalization_type=NormalizationType.RMS_NORM.value,
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
                True,
            ),
            (
                TimeConditioning.ADANORM.value,
                NormalizationType.RMS_NORM.value,
                None,
                False,
            ),
        ],
    )
    def test_proprioceptive_projection_created_whenever_key_is_set(
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
        real_paligemma_backbone: PaliGemmaVLM,
    ):
        vlm_layer_count = len(real_paligemma_backbone.get_backbone_layers())
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Pi0 requires equal VLM ({vlm_layer_count}) and expert "
                f"(99) layer counts."
            ),
        ):
            pi0_decoder_factory(
                expert_number_of_layers=99,
                vlm_backbone=real_paligemma_backbone,
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

    def test_vlm_backbone_builds_prefix_from_observations(
        self,
        pi0_decoder_factory: Callable[..., Pi0Decoder],
        real_paligemma_backbone: PaliGemmaVLM,
        raw_vlm_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = pi0_decoder_factory(
            input_keys=[],
            vlm_backbone=real_paligemma_backbone,
        )
        features = raw_vlm_features_factory()
        actions = noisy_actions_factory()
        with patch.object(
            real_paligemma_backbone,
            "build_prefix",
            wraps=real_paligemma_backbone.build_prefix,
        ) as build_prefix_spy:
            outputs = decoder(features=features, actions=actions)
        build_prefix_spy.assert_called_once_with(inputs=features)
        assert outputs["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
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
        features_without[PROPRIO_KEY] = torch.zeros(
            BATCH_SIZE,
            PROPRIO_DIM,
            dtype=torch.float32,
        )
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

    def test_adanorm_with_proprioceptive_key_appends_proprio_tokens(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # Proprio tokens go into the VLM prefix regardless of how the timestep
        # is fused, so ADANORM must not silently drop a configured key.
        decoder = initialized_decoder_factory(
            time_conditioning=TimeConditioning.ADANORM.value,
            proprioceptive_feature_key=PROPRIO_KEY,
        )
        features = prefix_features_factory(include_proprioceptive=True)
        actions = noisy_actions_factory()

        with patch.object(
            decoder,
            "_build_interleaved_attention_state",
            wraps=decoder._build_interleaved_attention_state,
        ) as attention_state_spy:
            decoder(features=features, actions=actions)

        attention_state_call = attention_state_spy.call_args
        prefix_embeddings = attention_state_call.kwargs["prefix_embeddings"]
        assert prefix_embeddings.shape[1] == PREFIX_SEQUENCE_LENGTH + 1
        assert attention_state_call.kwargs["causal_prefix_suffix_length"] == 1


@pytest.mark.integration
class TestPi0DecoderBehavior:
    def test_expert_layers_receive_gradients_while_vlm_frozen(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        # Freeze VLM, then restore after test to avoid polluting session-scoped backbone
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

    def test_attention_mask_reordered_to_action_prefix_for_joint_sdpa(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory()
        actions = noisy_actions_factory()
        action_len = PREDICTION_HORIZON
        with patch.object(
            decoder,
            "_run_training_forward",
            wraps=decoder._run_training_forward,
        ) as spy:
            decoder(features=features, actions=actions)
        mask = spy.call_args.kwargs["attention_mask"]
        # After permutation: rows :action_len = action queries, rows action_len: = prefix queries
        # Action queries can attend to prefix columns (action_len:) — unmasked
        assert not mask[0, 0, :action_len, action_len:].any()
        # Prefix queries cannot attend to action columns (:action_len) — all masked
        assert mask[0, 0, action_len:, :action_len].all()

    def test_prefix_padding_mask_is_forwarded_to_attention_builder(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        features = prefix_features_factory(include_padding_mask=True)
        actions = noisy_actions_factory()
        with patch.object(
            interleaved_vlm_module,
            "make_attention_mask",
            wraps=interleaved_vlm_module.make_attention_mask,
        ) as spy:
            decoder(features=features, actions=actions)
        received_mask = spy.call_args.kwargs["feature_token_mask"]
        assert torch.equal(received_mask, features[PADDING_MASK_KEY])

    def test_cached_prefix_uses_additive_padding_mask(
        self,
        initialized_decoder_factory: Callable[..., Pi0Decoder],
        prefix_features_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = initialized_decoder_factory()
        decoder.eval()
        decoder.enable_encoder_cache()
        features = prefix_features_factory(include_padding_mask=True)
        actions = noisy_actions_factory()
        with (
            patch.object(
                decoder,
                "_fill_prefix_cache",
                wraps=decoder._fill_prefix_cache,
            ) as spy,
            torch.no_grad(),
        ):
            decoder(features=features, actions=actions)
        prefix_attention_mask = spy.call_args.kwargs["prefix_attention_mask"]
        assert prefix_attention_mask.shape == (
            BATCH_SIZE,
            1,
            PREFIX_SEQUENCE_LENGTH,
            PREFIX_SEQUENCE_LENGTH,
        )
        expected_masked_value = torch.finfo(features[FEATURE_KEY].dtype).min
        assert torch.all(prefix_attention_mask[:, :, :, -2:] == expected_masked_value)
        assert torch.all(prefix_attention_mask[:, :, :, :-2] == 0)


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
        assert len(decoder._prefix_cache.layers) == NUM_EXPERT_LAYERS

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
