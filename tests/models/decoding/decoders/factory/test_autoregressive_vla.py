"""Tests for versatil.models.decoding.decoders.factory.autoregressive_vla module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.cache_utils import Cache

from versatil.data.constants import ActionTokenIdMappingType, SampleKey
from versatil.data.tokenization.action_discretizer import (
    ActionDiscretizer,
    BinnedActionDiscretizer,
)
from versatil.data.tokenization.action_token_id_mapping import ActionTokenIdMapping
from versatil.data.tokenization.action_tokenizer import ActionTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.factory.autoregressive_vla import (
    AutoregressiveVLADecoder,
)
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.prismatic import (
    PrismaticVLM,
)
from versatil.models.input_specification import InputSpecification

BATCH_SIZE = 2
PREFIX_TOKEN_LENGTH = 5
ACTION_TOKEN_LENGTH = 4
LANGUAGE_HIDDEN_DIMENSION = 16
VOCABULARY_SIZE = 32
PADDED_VOCABULARY_SIZE = 64
CAMERA_KEY = "agentview"


@pytest.fixture
def vlm_backbone_factory() -> Callable[..., MagicMock]:
    def factory() -> MagicMock:
        vlm_backbone = MagicMock(spec=GenerativeVLM)
        vlm_backbone.hidden_dim = LANGUAGE_HIDDEN_DIMENSION
        vlm_backbone.input_specification = InputSpecification(
            keys=[
                CAMERA_KEY,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
                SampleKey.IS_PAD_OBSERVATION.value,
            ],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        vlm_backbone.get_vocab_size.return_value = VOCABULARY_SIZE
        vlm_backbone.build_prefix.return_value = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, dtype=torch.bool),
        )
        language_output = MagicMock(spec=CausalLanguageModelOutput)
        language_output.logits = torch.ones(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            VOCABULARY_SIZE,
        )
        language_output.hidden_states = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
        )
        language_output.past_key_values = MagicMock(spec=Cache)
        vlm_backbone.forward_language_model.return_value = language_output
        vlm_backbone.embed_input_ids.return_value = torch.zeros(
            BATCH_SIZE,
            ACTION_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        return vlm_backbone

    return factory


@pytest.fixture
def autoregressive_vla_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    vlm_backbone_factory: Callable[..., MagicMock],
) -> Callable[..., AutoregressiveVLADecoder]:
    def factory(
        input_keys: list[str] | None = None,
        action_heads: dict[str, ActionHead] | None = None,
        vlm_backbone: MagicMock | None = None,
    ) -> AutoregressiveVLADecoder:
        if input_keys is None:
            input_keys = []
        if action_heads is None:
            action_heads = {}
        if vlm_backbone is None:
            vlm_backbone = vlm_backbone_factory()
        return AutoregressiveVLADecoder(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=mock_action_space_factory(position_dim=3),
            observation_space=mock_observation_space_factory(),
            observation_horizon=1,
            prediction_horizon=4,
            device="cpu",
            vlm_backbone=vlm_backbone,
            max_seq_len=32,
            temperature=1.0,
            learnable_temperature=False,
            deterministic=True,
            causal_prefix=False,
        )

    return factory


@pytest.fixture
def language_vocab_tokenizer_factory() -> Callable[..., MagicMock]:
    def factory(
        vocab_size: int = VOCABULARY_SIZE,
        eos_token_id: int = VOCABULARY_SIZE - 1,
        mapping_type: str = ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value,
        token_count: int = 3,
        encoded_token_ids: list[int] | None = None,
        max_token_len: int = 256,
        fixed_length: bool = False,
        time_horizon: int = 4,
        action_dim: int = 3,
    ) -> MagicMock:
        if encoded_token_ids is None:
            encoded_token_ids = [10 + token_id for token_id in range(token_count)]
        if fixed_length:
            action_discretizer = MagicMock(spec=BinnedActionDiscretizer)
            action_discretizer.time_horizon = time_horizon
            action_discretizer.action_dim = action_dim
        else:
            action_discretizer = MagicMock(spec=ActionDiscretizer)
        action_discretizer.token_count = token_count
        token_id_mapping = MagicMock(spec=ActionTokenIdMapping)
        token_id_mapping.state_dict.return_value = {"type": mapping_type}
        token_id_mapping.encode.return_value = np.asarray(encoded_token_ids)
        action_tokenizer = MagicMock(spec=ActionTokenizer)
        action_tokenizer.action_discretizer = action_discretizer
        action_tokenizer.token_id_mapping = token_id_mapping
        action_tokenizer.vocab_size = vocab_size
        action_tokenizer.eos_token_id = eos_token_id
        action_tokenizer.max_token_len = max_token_len
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = action_tokenizer
        return tokenizer

    return factory


@pytest.mark.unit
class TestAutoregressiveVLADecoderInitialization:
    def test_requests_raw_vlm_observations(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])

        assert decoder.decoder_input.needs_raw_observations is True
        assert decoder.decoder_input.requires_actions is True
        assert decoder.decoder_input.keys == [
            CAMERA_KEY,
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        ]

    def test_rejects_extra_input_keys(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        expected_message = (
            "AutoregressiveVLADecoder builds its prefix from vlm_backbone inputs. "
            "Set input_keys to an empty list, got ['encoded_feature']."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            autoregressive_vla_decoder_factory(input_keys=["encoded_feature"])

    def test_rejects_action_heads(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        action_head_factory: Callable[..., ActionHead],
    ) -> None:
        expected_message = (
            "AutoregressiveVLADecoder predicts action tokens with the VLM language vocabulary "
            "head, so action_heads must be empty."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            autoregressive_vla_decoder_factory(
                action_heads={
                    DecoderOutputKey.ACTION_LOGITS.value: action_head_factory()
                }
            )

    def test_auxiliary_output_keys_include_lm_token_outputs(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])

        assert decoder.get_auxiliary_output_keys() == {
            SampleKey.TOKENIZED_ACTIONS.value,
            DecoderOutputKey.ACTION_LOGITS.value,
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value,
        }


@pytest.mark.unit
class TestAutoregressiveVLADecoderTokenizer:
    def test_set_tokenizer_accepts_language_vocabulary_action_ids(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokenizer = language_vocab_tokenizer_factory(
            token_count=3,
            encoded_token_ids=[10, 11, 12],
            eos_token_id=VOCABULARY_SIZE - 1,
        )

        decoder.set_tokenizer(tokenizer=tokenizer)

        assert decoder.tokenizer == tokenizer.action_tokenizer
        assert decoder.vocab_size == VOCABULARY_SIZE
        assert decoder.eos_token_id == VOCABULARY_SIZE - 1
        torch.testing.assert_close(
            decoder.valid_generation_token_ids,
            torch.tensor([10, 11, 12, VOCABULARY_SIZE - 1]),
        )
        decoder.vlm_backbone.resize_token_embeddings.assert_not_called()
        tokenizer.action_tokenizer.token_id_mapping.encode.assert_called_once_with(
            [0, 1, 2]
        )

    def test_set_tokenizer_excludes_eos_when_action_tokenizer_is_fixed_length(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokenizer = language_vocab_tokenizer_factory(
            token_count=3,
            encoded_token_ids=[10, 11, 12],
            eos_token_id=VOCABULARY_SIZE - 1,
            fixed_length=True,
            time_horizon=1,
            action_dim=3,
        )

        decoder.set_tokenizer(tokenizer=tokenizer)

        # An early EOS sample would truncate the fixed-length payload and
        # crash the binned detokenizer, so EOS must not be sampleable.
        torch.testing.assert_close(
            decoder.valid_generation_token_ids,
            torch.tensor([10, 11, 12]),
        )

    def test_set_tokenizer_accepts_padded_vlm_vocabulary(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.vlm_backbone.get_vocab_size.return_value = PADDED_VOCABULARY_SIZE
        tokenizer = language_vocab_tokenizer_factory(
            token_count=3,
            encoded_token_ids=[10, 11, 12],
            eos_token_id=VOCABULARY_SIZE - 1,
        )

        decoder.set_tokenizer(tokenizer=tokenizer)

        assert decoder.vocab_size == PADDED_VOCABULARY_SIZE
        torch.testing.assert_close(
            decoder.valid_generation_token_ids,
            torch.tensor([10, 11, 12, VOCABULARY_SIZE - 1]),
        )
        decoder.vlm_backbone.resize_token_embeddings.assert_not_called()

    def test_set_tokenizer_resizes_smaller_vlm_vocabulary(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.vlm_backbone.get_vocab_size.side_effect = [
            VOCABULARY_SIZE - 1,
            PADDED_VOCABULARY_SIZE,
        ]
        tokenizer = language_vocab_tokenizer_factory()

        decoder.set_tokenizer(tokenizer=tokenizer)

        decoder.vlm_backbone.resize_token_embeddings.assert_called_once_with(
            VOCABULARY_SIZE
        )
        assert decoder.vocab_size == PADDED_VOCABULARY_SIZE

    def test_set_tokenizer_rejects_missing_action_tokenizer(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = None
        expected_message = (
            "AutoregressiveVLADecoder requires an action tokenizer with "
            "language-vocabulary token IDs."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_set_tokenizer_rejects_non_language_vocabulary_mapping(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokenizer = language_vocab_tokenizer_factory(
            mapping_type=ActionTokenIdMappingType.IDENTITY.value
        )
        expected_message = (
            "AutoregressiveVLADecoder requires action_tokenizer.token_id_mapping.type="
            f"{ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_set_tokenizer_rejects_eos_outside_vocabulary(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokenizer = language_vocab_tokenizer_factory(eos_token_id=VOCABULARY_SIZE)
        expected_message = (
            "AutoregressiveVLADecoder received an action tokenizer with eos_token_id "
            "outside the model vocabulary: "
            f"eos_token_id={VOCABULARY_SIZE}, vocab_size={VOCABULARY_SIZE}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_set_tokenizer_rejects_vlm_without_vocab_size(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.vlm_backbone.get_vocab_size.return_value = None
        tokenizer = language_vocab_tokenizer_factory()
        expected_message = (
            "AutoregressiveVLADecoder vlm_backbone must expose get_vocab_size()."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_set_tokenizer_rejects_vlm_vocab_smaller_after_resize(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.vlm_backbone.get_vocab_size.side_effect = [
            VOCABULARY_SIZE - 1,
            VOCABULARY_SIZE - 1,
        ]
        tokenizer = language_vocab_tokenizer_factory()
        expected_message = (
            "AutoregressiveVLADecoder VLM language vocabulary must cover the "
            "action tokenizer vocabulary after resizing, got "
            "tokenizer_vocab_size=32 and vlm_backbone_vocab_size=31."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_set_tokenizer_rejects_valid_ids_outside_vocabulary(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokenizer = language_vocab_tokenizer_factory(
            token_count=2,
            encoded_token_ids=[10, VOCABULARY_SIZE],
        )
        expected_message = (
            "AutoregressiveVLADecoder valid action-token IDs must lie inside "
            "the action tokenizer vocabulary [0, 32)."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_validate_action_tokenizer_is_set_rejects_uninitialized_decoder(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        expected_message = "AutoregressiveVLADecoder requires set_tokenizer() to be called before forward."

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._validate_action_tokenizer_is_set()


@pytest.mark.unit
class TestAutoregressiveVLADecoderPrefix:
    def test_build_projected_prefix_uses_vlm_backbone(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        features = {CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16)}

        prefix_tokens, prefix_mask = decoder._build_projected_prefix(features=features)

        decoder.vlm_backbone.build_prefix.assert_called_once_with(inputs=features)
        assert torch.equal(
            prefix_tokens,
            decoder.vlm_backbone.build_prefix.return_value[0],
        )
        assert torch.equal(
            prefix_mask, decoder.vlm_backbone.build_prefix.return_value[1]
        )

    def test_run_language_model_logits_uses_vlm_language_model(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        tokens = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION)
        attention_mask = torch.ones(
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_LENGTH,
            PREFIX_TOKEN_LENGTH,
            dtype=torch.bool,
        )

        logits = decoder._run_language_model_logits(
            tokens=tokens,
            attention_mask=attention_mask,
        )

        decoder.vlm_backbone.forward_language_model.assert_called_once_with(
            inputs_embeds=tokens,
            attention_mask=attention_mask,
            use_cache=False,
        )
        assert torch.equal(
            logits, decoder.vlm_backbone.forward_language_model.return_value.logits
        )

    def test_forward_routes_through_vlm_prefix_before_training(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        features = {CAMERA_KEY: torch.zeros(BATCH_SIZE, 3, 16, 16)}
        actions = {
            SampleKey.TOKENIZED_ACTIONS.value: torch.zeros(
                BATCH_SIZE,
                ACTION_TOKEN_LENGTH,
                dtype=torch.long,
            )
        }
        predictions = {
            DecoderOutputKey.ACTION_LOGITS.value: torch.zeros(
                BATCH_SIZE,
                ACTION_TOKEN_LENGTH,
                VOCABULARY_SIZE,
            )
        }
        with (
            patch.object(decoder, "_validate_action_tokenizer_is_set") as validate_spy,
            patch.object(
                decoder,
                "_forward_action_token_training",
                return_value=predictions,
            ) as training_spy,
        ):
            output = decoder(features=features, actions=actions)

        validate_spy.assert_called_once_with()
        decoder.vlm_backbone.build_prefix.assert_called_once_with(inputs=features)
        training_spy.assert_called_once_with(
            actions=actions,
            prefix_tokens=decoder.vlm_backbone.build_prefix.return_value[0],
            prefix_token_mask=decoder.vlm_backbone.build_prefix.return_value[1],
        )
        assert torch.equal(
            output[DecoderOutputKey.ACTION_LOGITS.value],
            predictions[DecoderOutputKey.ACTION_LOGITS.value],
        )


@pytest.mark.unit
class TestAutoregressiveVLADecoderTargets:
    def test_get_target_token_ids_casts_to_long_on_decoder_device(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        target_token_ids = torch.ones(
            BATCH_SIZE,
            ACTION_TOKEN_LENGTH,
            dtype=torch.float32,
        )

        output = decoder._get_target_token_ids(
            actions={SampleKey.TOKENIZED_ACTIONS.value: target_token_ids},
            batch_size=BATCH_SIZE,
        )

        assert output.dtype == torch.long
        assert output.device.type == "cpu"
        torch.testing.assert_close(output, target_token_ids.long())

    def test_get_target_token_ids_rejects_missing_tokenized_actions(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        action_key = SampleKey.TOKENIZED_ACTIONS.value
        expected_message = (
            f"AutoregressiveVLADecoder training requires '{action_key}' in actions."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._get_target_token_ids(actions={}, batch_size=BATCH_SIZE)

    def test_get_target_token_ids_rejects_wrong_rank(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        action_key = SampleKey.TOKENIZED_ACTIONS.value
        target_token_ids = torch.zeros(BATCH_SIZE, ACTION_TOKEN_LENGTH, 1)
        expected_message = (
            f"'{action_key}' must have shape (B, token_length), "
            f"got {target_token_ids.shape}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._get_target_token_ids(
                actions={action_key: target_token_ids},
                batch_size=BATCH_SIZE,
            )

    def test_get_target_token_ids_rejects_batch_mismatch(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        action_key = SampleKey.TOKENIZED_ACTIONS.value
        target_token_ids = torch.zeros(BATCH_SIZE + 1, ACTION_TOKEN_LENGTH)
        expected_message = (
            f"'{action_key}' batch size must match feature batch size "
            f"{BATCH_SIZE}, got {BATCH_SIZE + 1}."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._get_target_token_ids(
                actions={action_key: target_token_ids},
                batch_size=BATCH_SIZE,
            )


@pytest.mark.unit
class TestAutoregressiveVLADecoderGeneration:
    def test_sample_next_action_token_uses_greedy_valid_subset(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.valid_generation_token_ids = torch.tensor([4, 7, 9])
        logits = torch.zeros(BATCH_SIZE, 1, VOCABULARY_SIZE)
        logits[:, :, 7] = 3.0
        logits[:, :, 9] = 2.0

        next_token = decoder._sample_next_action_token(logits=logits)

        torch.testing.assert_close(next_token, torch.full((BATCH_SIZE, 1), 7))

    def test_sample_next_action_token_uses_multinomial_for_stochastic_sampling(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.deterministic = False
        decoder.temperature.data.fill_(2.0)
        decoder.valid_generation_token_ids = torch.tensor([4, 7, 9])
        logits = torch.zeros(BATCH_SIZE, 1, VOCABULARY_SIZE)

        with patch(
            "versatil.models.decoding.decoders.factory.autoregressive_vla.torch.multinomial",
            autospec=True,
            return_value=torch.tensor([[2], [0]]),
        ) as multinomial_mock:
            next_token = decoder._sample_next_action_token(logits=logits)

        probabilities = multinomial_mock.call_args.args[0]
        assert probabilities.shape == (BATCH_SIZE, 3)
        multinomial_mock.assert_called_once()
        torch.testing.assert_close(next_token, torch.tensor([[9], [4]]))

    def test_sample_next_action_token_rejects_missing_valid_token_ids(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.valid_generation_token_ids = None
        logits = torch.zeros(BATCH_SIZE, 1, VOCABULARY_SIZE)
        expected_message = (
            "AutoregressiveVLADecoder valid action-token IDs are not initialized."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._sample_next_action_token(logits=logits)

    def test_training_forward_slices_next_token_logits(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        target_token_ids = torch.arange(
            BATCH_SIZE * ACTION_TOKEN_LENGTH,
        ).reshape(BATCH_SIZE, ACTION_TOKEN_LENGTH)
        action_token_embeddings = torch.ones(
            BATCH_SIZE,
            ACTION_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        full_sequence_length = PREFIX_TOKEN_LENGTH + ACTION_TOKEN_LENGTH
        language_logits = torch.arange(
            BATCH_SIZE * full_sequence_length * VOCABULARY_SIZE,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, full_sequence_length, VOCABULARY_SIZE)
        decoder.vlm_backbone.embed_input_ids.return_value = action_token_embeddings

        with patch.object(
            decoder,
            "_run_language_model_logits",
            return_value=language_logits,
        ) as language_spy:
            output = decoder._forward_action_token_training(
                actions={SampleKey.TOKENIZED_ACTIONS.value: target_token_ids},
                prefix_tokens=prefix_tokens,
                prefix_token_mask=None,
            )

        decoder.vlm_backbone.embed_input_ids.assert_called_once()
        torch.testing.assert_close(
            decoder.vlm_backbone.embed_input_ids.call_args.args[0],
            target_token_ids.long(),
        )
        language_spy.assert_called_once()
        expected_logits = language_logits[
            :,
            PREFIX_TOKEN_LENGTH - 1 : PREFIX_TOKEN_LENGTH + ACTION_TOKEN_LENGTH - 1,
            :,
        ]
        torch.testing.assert_close(
            output[DecoderOutputKey.ACTION_LOGITS.value],
            expected_logits,
        )

    def test_training_forward_removes_prefix_padding_before_action_tokens(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.causal_prefix = True
        prefix_tokens = torch.arange(
            BATCH_SIZE * PREFIX_TOKEN_LENGTH * LANGUAGE_HIDDEN_DIMENSION,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION)
        prefix_token_mask = torch.tensor(
            [
                [False, False, True, True, True],
                [False, False, False, False, True],
            ],
            dtype=torch.bool,
        )
        target_token_ids = torch.arange(
            BATCH_SIZE * ACTION_TOKEN_LENGTH,
        ).reshape(BATCH_SIZE, ACTION_TOKEN_LENGTH)
        action_token_embeddings = torch.full(
            (
                BATCH_SIZE,
                ACTION_TOKEN_LENGTH,
                LANGUAGE_HIDDEN_DIMENSION,
            ),
            1000.0,
        )
        language_logits = torch.arange(
            BATCH_SIZE * (4 + ACTION_TOKEN_LENGTH) * VOCABULARY_SIZE,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, 4 + ACTION_TOKEN_LENGTH, VOCABULARY_SIZE)
        decoder.vlm_backbone.embed_input_ids.return_value = action_token_embeddings

        with patch.object(
            decoder,
            "_run_language_model_logits",
            return_value=language_logits,
        ) as language_spy:
            output = decoder._forward_action_token_training(
                actions={SampleKey.TOKENIZED_ACTIONS.value: target_token_ids},
                prefix_tokens=prefix_tokens,
                prefix_token_mask=prefix_token_mask,
            )

        language_call = language_spy.call_args
        full_token_sequence = language_call.kwargs["tokens"]
        attention_mask = language_call.kwargs["attention_mask"]
        assert full_token_sequence.shape == (
            BATCH_SIZE,
            4 + ACTION_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        torch.testing.assert_close(
            full_token_sequence[0, :2],
            prefix_tokens[0, :2],
        )
        torch.testing.assert_close(
            full_token_sequence[0, 2 : 2 + ACTION_TOKEN_LENGTH],
            action_token_embeddings[0],
        )
        torch.testing.assert_close(
            full_token_sequence[0, 2 + ACTION_TOKEN_LENGTH :],
            torch.zeros(2, LANGUAGE_HIDDEN_DIMENSION),
        )
        torch.testing.assert_close(
            full_token_sequence[1, :4],
            prefix_tokens[1, :4],
        )
        torch.testing.assert_close(
            full_token_sequence[1, 4 : 4 + ACTION_TOKEN_LENGTH],
            action_token_embeddings[1],
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        torch.testing.assert_close(attention_mask, expected_attention_mask)
        expected_logits = torch.stack(
            [
                language_logits[0, 1 : 1 + ACTION_TOKEN_LENGTH],
                language_logits[1, 3 : 3 + ACTION_TOKEN_LENGTH],
            ],
            dim=0,
        )
        torch.testing.assert_close(
            output[DecoderOutputKey.ACTION_LOGITS.value],
            expected_logits,
        )

    def test_training_forward_rejects_empty_prefix(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        expected_message = (
            "AutoregressiveVLADecoder requires a non-empty conditioning prefix."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._forward_action_token_training(
                actions={
                    SampleKey.TOKENIZED_ACTIONS.value: torch.zeros(
                        BATCH_SIZE,
                        ACTION_TOKEN_LENGTH,
                    )
                },
                prefix_tokens=torch.zeros(BATCH_SIZE, 0, LANGUAGE_HIDDEN_DIMENSION),
                prefix_token_mask=None,
            )

    def test_training_forward_rejects_sequence_longer_than_maximum(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.max_seq_len = PREFIX_TOKEN_LENGTH + ACTION_TOKEN_LENGTH - 1
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        target_token_ids = torch.zeros(BATCH_SIZE, ACTION_TOKEN_LENGTH)
        expected_message = (
            f"Input token length {PREFIX_TOKEN_LENGTH + ACTION_TOKEN_LENGTH} "
            f"> max_seq_len {decoder.max_seq_len}. Consider increasing max_seq_len "
            "or reducing the text/action/feature token count."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._forward_action_token_training(
                actions={SampleKey.TOKENIZED_ACTIONS.value: target_token_ids},
                prefix_tokens=prefix_tokens,
                prefix_token_mask=None,
            )

    def test_inference_forward_rejects_prefix_that_fills_sequence_limit(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.max_seq_len = PREFIX_TOKEN_LENGTH
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        expected_message = (
            f"Input prefix token length {PREFIX_TOKEN_LENGTH} >= max_seq_len "
            f"{PREFIX_TOKEN_LENGTH}. No room for generated action tokens. "
            "Consider increasing max_seq_len or reducing feature count."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder._forward_action_token_inference(
                prefix_tokens=prefix_tokens,
                prefix_token_mask=None,
            )

    def test_inference_forward_stops_after_all_batches_emit_eos(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.set_tokenizer(
            tokenizer=language_vocab_tokenizer_factory(
                token_count=1,
                encoded_token_ids=[4],
                max_token_len=4,
            )
        )
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        language_logits = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, VOCABULARY_SIZE)
        language_logits[:, -1, VOCABULARY_SIZE - 1] = 10.0
        prefill_output = MagicMock(spec=CausalLanguageModelOutput)
        prefill_output.logits = language_logits
        prefill_output.hidden_states = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
        )
        prefill_output.past_key_values = MagicMock(spec=Cache)

        decoder.vlm_backbone.forward_language_model.return_value = prefill_output

        output = decoder._forward_action_token_inference(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=None,
        )

        decoder.vlm_backbone.forward_language_model.assert_called_once()
        prefill_call = decoder.vlm_backbone.forward_language_model.call_args
        torch.testing.assert_close(prefill_call.kwargs["inputs_embeds"], prefix_tokens)
        prefill_attention_mask = prefill_call.kwargs["attention_mask"]
        assert prefill_attention_mask.shape == (
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_LENGTH,
            PREFIX_TOKEN_LENGTH,
        )
        # Additive mask form: zeros everywhere means every position may attend.
        assert prefill_attention_mask.dtype == prefix_tokens.dtype
        assert (prefill_attention_mask == 0.0).all()
        torch.testing.assert_close(
            prefill_call.kwargs["position_ids"],
            torch.arange(PREFIX_TOKEN_LENGTH).unsqueeze(0).expand(BATCH_SIZE, -1),
        )
        assert prefill_call.kwargs["use_cache"]
        decoder.vlm_backbone.embed_input_ids.assert_not_called()
        torch.testing.assert_close(
            output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            torch.full((BATCH_SIZE, 1), VOCABULARY_SIZE - 1),
        )

    def test_inference_fixed_length_mode_runs_to_payload_cap_when_eos_is_missing(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.max_seq_len = PREFIX_TOKEN_LENGTH + 2
        decoder.set_tokenizer(
            tokenizer=language_vocab_tokenizer_factory(
                token_count=1,
                encoded_token_ids=[4],
                max_token_len=2,
                fixed_length=True,
                time_horizon=2,
                action_dim=1,
            )
        )
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        prefill_logits = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, VOCABULARY_SIZE)
        prefill_logits[:, -1, 4] = 10.0
        decode_logits = torch.zeros(BATCH_SIZE, 1, VOCABULARY_SIZE)
        decode_logits[:, -1, 4] = 10.0
        prefill_output = MagicMock(spec=CausalLanguageModelOutput)
        prefill_output.logits = prefill_logits
        prefill_output.hidden_states = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
        )
        prefill_output.past_key_values = MagicMock(spec=Cache)
        decode_output = MagicMock(spec=CausalLanguageModelOutput)
        decode_output.logits = decode_logits
        decode_output.hidden_states = (
            torch.zeros(BATCH_SIZE, 1, LANGUAGE_HIDDEN_DIMENSION),
        )
        decode_output.past_key_values = MagicMock(spec=Cache)
        decoder.vlm_backbone.forward_language_model.side_effect = [
            prefill_output,
            decode_output,
        ]

        output = decoder._forward_action_token_inference(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=None,
        )

        assert decoder.vlm_backbone.forward_language_model.call_count == 2
        torch.testing.assert_close(
            output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            torch.full((BATCH_SIZE, 2), 4),
        )

    def test_inference_forward_uses_sampled_token_ids_for_cached_next_step(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.max_seq_len = PREFIX_TOKEN_LENGTH + 2
        decoder.set_tokenizer(
            tokenizer=language_vocab_tokenizer_factory(
                token_count=1,
                encoded_token_ids=[4],
                max_token_len=4,
            )
        )
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        first_logits = torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, VOCABULARY_SIZE)
        first_logits[:, -1, 4] = 10.0
        second_logits = torch.zeros(BATCH_SIZE, 1, VOCABULARY_SIZE)
        second_logits[:, -1, VOCABULARY_SIZE - 1] = 10.0
        prefill_cache = MagicMock(spec=Cache)
        decode_cache = MagicMock(spec=Cache)
        prefill_output = MagicMock(spec=CausalLanguageModelOutput)
        prefill_output.logits = first_logits
        prefill_output.hidden_states = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
        )
        prefill_output.past_key_values = prefill_cache
        decode_output = MagicMock(spec=CausalLanguageModelOutput)
        decode_output.logits = second_logits
        decode_output.hidden_states = (
            torch.zeros(BATCH_SIZE, 1, LANGUAGE_HIDDEN_DIMENSION),
        )
        decode_output.past_key_values = decode_cache

        decoder.vlm_backbone.forward_language_model.side_effect = [
            prefill_output,
            decode_output,
        ]

        output = decoder._forward_action_token_inference(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=None,
        )

        assert decoder.vlm_backbone.forward_language_model.call_count == 2
        prefill_call = decoder.vlm_backbone.forward_language_model.call_args_list[0]
        torch.testing.assert_close(prefill_call.kwargs["inputs_embeds"], prefix_tokens)
        prefill_attention_mask = prefill_call.kwargs["attention_mask"]
        assert prefill_attention_mask.shape == (
            BATCH_SIZE,
            1,
            PREFIX_TOKEN_LENGTH,
            PREFIX_TOKEN_LENGTH,
        )
        # Additive mask form: zeros everywhere means every position may attend.
        assert prefill_attention_mask.dtype == prefix_tokens.dtype
        assert (prefill_attention_mask == 0.0).all()
        torch.testing.assert_close(
            prefill_call.kwargs["position_ids"],
            torch.arange(PREFIX_TOKEN_LENGTH).unsqueeze(0).expand(BATCH_SIZE, -1),
        )
        assert prefill_call.kwargs["use_cache"]
        decode_call = decoder.vlm_backbone.forward_language_model.call_args_list[1]
        torch.testing.assert_close(
            decode_call.kwargs["input_ids"],
            torch.full((BATCH_SIZE, 1), 4),
        )
        assert decode_call.kwargs["past_key_values"] is prefill_cache
        torch.testing.assert_close(
            decode_call.kwargs["position_ids"],
            torch.full((BATCH_SIZE, 1), PREFIX_TOKEN_LENGTH),
        )
        assert decode_call.kwargs["use_cache"]
        decoder.vlm_backbone.embed_input_ids.assert_not_called()
        torch.testing.assert_close(
            output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            torch.tensor(
                [
                    [4, VOCABULARY_SIZE - 1],
                    [4, VOCABULARY_SIZE - 1],
                ]
            ),
        )

    @pytest.mark.parametrize(
        "max_token_len, max_seq_len, expected_steps",
        [
            (3, PREFIX_TOKEN_LENGTH + 10, 3),
            (256, PREFIX_TOKEN_LENGTH + 2, 2),
        ],
    )
    def test_inference_caps_generation_by_tokenizer_and_context(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
        max_token_len: int,
        max_seq_len: int,
        expected_steps: int,
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.max_seq_len = max_seq_len
        decoder.set_tokenizer(
            tokenizer=language_vocab_tokenizer_factory(max_token_len=max_token_len)
        )
        prefix_tokens = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            LANGUAGE_HIDDEN_DIMENSION,
        )
        prefill_output = MagicMock(spec=CausalLanguageModelOutput)
        prefill_output.logits = torch.zeros(
            BATCH_SIZE,
            PREFIX_TOKEN_LENGTH,
            VOCABULARY_SIZE,
        )
        prefill_output.hidden_states = (
            torch.zeros(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION),
        )
        prefill_output.past_key_values = MagicMock(spec=Cache)
        decoder.vlm_backbone.forward_language_model.return_value = prefill_output
        predictions = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(
                BATCH_SIZE,
                expected_steps,
                dtype=torch.long,
            )
        }
        generation_mock = MagicMock(
            spec=decoder._run_cached_autoregressive_generation,
            return_value=predictions,
        )

        with patch.object(
            decoder,
            "_run_cached_autoregressive_generation",
            generation_mock,
        ):
            output = decoder._forward_action_token_inference(
                prefix_tokens=prefix_tokens,
                prefix_token_mask=None,
            )

        generation_mock.assert_called_once()
        assert (
            generation_mock.call_args.kwargs["max_generation_steps"] == expected_steps
        )
        torch.testing.assert_close(
            output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
        )

    def test_inference_forward_right_aligns_valid_prefix_tokens(
        self,
        autoregressive_vla_decoder_factory: Callable[..., AutoregressiveVLADecoder],
        language_vocab_tokenizer_factory: Callable[..., MagicMock],
    ) -> None:
        decoder = autoregressive_vla_decoder_factory(input_keys=[])
        decoder.causal_prefix = True
        decoder.set_tokenizer(
            tokenizer=language_vocab_tokenizer_factory(
                token_count=1,
                encoded_token_ids=[4],
                max_token_len=4,
            )
        )
        prefix_tokens = torch.arange(
            BATCH_SIZE * PREFIX_TOKEN_LENGTH * LANGUAGE_HIDDEN_DIMENSION,
            dtype=torch.float32,
        ).reshape(BATCH_SIZE, PREFIX_TOKEN_LENGTH, LANGUAGE_HIDDEN_DIMENSION)
        prefix_token_mask = torch.tensor(
            [
                [False, False, True, True, True],
                [False, False, False, False, True],
            ],
            dtype=torch.bool,
        )
        prefill_output = MagicMock(spec=CausalLanguageModelOutput)
        prefill_output.logits = torch.zeros(
            BATCH_SIZE,
            4,
            VOCABULARY_SIZE,
        )
        prefill_output.hidden_states = (
            torch.zeros(BATCH_SIZE, 4, LANGUAGE_HIDDEN_DIMENSION),
        )
        prefill_output.past_key_values = MagicMock(spec=Cache)
        decoder.vlm_backbone.forward_language_model.return_value = prefill_output
        predictions = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(
                BATCH_SIZE,
                1,
                dtype=torch.long,
            )
        }
        generation_mock = MagicMock(
            spec=decoder._run_cached_autoregressive_generation,
            return_value=predictions,
        )

        with patch.object(
            decoder,
            "_run_cached_autoregressive_generation",
            generation_mock,
        ):
            output = decoder._forward_action_token_inference(
                prefix_tokens=prefix_tokens,
                prefix_token_mask=prefix_token_mask,
            )

        prefill_call = decoder.vlm_backbone.forward_language_model.call_args
        right_aligned_prefix = prefill_call.kwargs["inputs_embeds"]
        right_aligned_padding_mask = torch.tensor(
            [
                [True, True, False, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(
            right_aligned_prefix[0, :2],
            torch.zeros(2, LANGUAGE_HIDDEN_DIMENSION),
        )
        torch.testing.assert_close(right_aligned_prefix[0, 2:], prefix_tokens[0, :2])
        torch.testing.assert_close(right_aligned_prefix[1], prefix_tokens[1, :4])
        torch.testing.assert_close(
            prefill_call.kwargs["attention_mask"],
            (~right_aligned_padding_mask).long(),
        )
        torch.testing.assert_close(
            prefill_call.kwargs["position_ids"],
            torch.tensor(
                [
                    [0, 0, 0, 1],
                    [0, 1, 2, 3],
                ]
            ),
        )
        initial_state = generation_mock.call_args.kwargs["initial_state"]
        assert initial_state.sequence_length == 4
        torch.testing.assert_close(
            initial_state.attention_mask,
            (~right_aligned_padding_mask).long(),
        )
        torch.testing.assert_close(
            initial_state.position_ids,
            torch.tensor([[1], [3]]),
        )
        torch.testing.assert_close(
            output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
        )


@pytest.mark.integration
def test_inference_forward_runs_real_vlm_cached_decode(
    tiny_prismatic_vlm_factory: Callable[..., PrismaticVLM],
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
) -> None:
    vlm_backbone = tiny_prismatic_vlm_factory()
    prefix_length = 3
    decoder = AutoregressiveVLADecoder(
        action_heads={},
        input_keys=[],
        action_space=mock_action_space_factory(position_dim=3),
        observation_space=mock_observation_space_factory(),
        observation_horizon=1,
        prediction_horizon=4,
        device="cpu",
        vlm_backbone=vlm_backbone,
        max_seq_len=prefix_length + 2,
        temperature=1.0,
        learnable_temperature=False,
        deterministic=True,
        causal_prefix=False,
    )
    decoder.valid_generation_token_ids = torch.arange(vlm_backbone.get_vocab_size())
    decoder.eos_token_id = -1
    decoder.tokenizer = MagicMock(spec=ActionTokenizer)
    decoder.tokenizer.action_discretizer = MagicMock(spec=ActionDiscretizer)
    decoder.tokenizer.max_token_len = 2
    prefix_token_ids = torch.arange(
        BATCH_SIZE * prefix_length,
        dtype=torch.long,
    ).reshape(BATCH_SIZE, prefix_length)
    prefix_tokens = vlm_backbone.embed_input_ids(token_ids=prefix_token_ids)

    with (
        torch.no_grad(),
        patch.object(
            vlm_backbone,
            "forward_language_model",
            wraps=vlm_backbone.forward_language_model,
        ) as forward_language_model_spy,
    ):
        output = decoder._forward_action_token_inference(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=None,
        )

    generated_tokens = output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
    assert generated_tokens.shape == (BATCH_SIZE, 2)
    assert generated_tokens.min() >= 0
    assert generated_tokens.max() < vlm_backbone.get_vocab_size()
    assert forward_language_model_spy.call_count == 2
    prefill_call = forward_language_model_spy.call_args_list[0]
    decode_call = forward_language_model_spy.call_args_list[1]
    torch.testing.assert_close(prefill_call.kwargs["inputs_embeds"], prefix_tokens)
    assert prefill_call.kwargs["use_cache"]
    assert decode_call.kwargs["past_key_values"] is not None
    assert decode_call.kwargs["use_cache"]
