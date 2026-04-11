"""Tests for versatil.models.encoding.encoders.language.language module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    LanguageEncoderType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.language.language import LanguageEncoder

HIDDEN_SIZE = 768
VOCAB_SIZE = 30522
MAX_TOKEN_LEN = 16


def _mock_build_encoder(self):
    """Side-effect to set self.encoder with expected attributes."""
    self.encoder = MagicMock()
    self.encoder.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    self.encoder.device = torch.device("cpu")
    self.encoder.embedding_dim = HIDDEN_SIZE
    self.config = MagicMock()
    self.config.hidden_size = HIDDEN_SIZE
    self.config.vocab_size = VOCAB_SIZE


@pytest.fixture
def language_encoder_factory() -> Callable[..., LanguageEncoder]:
    """Factory for LanguageEncoder with mocked backbone."""

    def factory(
        pretrained: bool = False,
        frozen: bool = False,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        model_name: str = LanguageEncoderType.BERT_BASE.value,
        max_token_len: int = MAX_TOKEN_LEN,
        use_embeddings_only: bool = False,
    ) -> LanguageEncoder:
        with patch.object(LanguageEncoder, "_build_encoder", _mock_build_encoder):
            return LanguageEncoder(
                pretrained=pretrained,
                frozen=frozen,
                pooling_method=pooling_method,
                model_name=model_name,
                max_token_len=max_token_len,
                use_embeddings_only=use_embeddings_only,
            )

    return factory


@pytest.fixture
def token_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for tokenized text input tensors."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 10,
        time_steps: int = 1,
        include_padding_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        shape = (batch_size, time_steps, sequence_length)
        token_ids = torch.from_numpy(
            rng.integers(low=0, high=VOCAB_SIZE, size=shape).astype(np.int64)
        )
        result = {SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids}
        if include_padding_mask:
            mask = torch.zeros(shape, dtype=torch.bool)
            result[SampleKey.IS_PAD_OBSERVATION.value] = mask
        return result

    return factory


class TestLanguageEncoderInitialization:
    def test_has_encoder_interface(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        spec = encoder.get_output_specification()
        feature_keys = [m.key for m in spec]
        assert len(feature_keys) == 2
        assert EncoderOutputKeys.LANGUAGE.value in feature_keys
        assert encoder.padding_mask_name in feature_keys

    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.DEFAULT.value,
            PoolingMethod.AVERAGE.value,
        ],
    )
    @pytest.mark.parametrize("max_token_len", [16, 64])
    @pytest.mark.parametrize(
        "model_name",
        [
            LanguageEncoderType.BERT_BASE.value,
            LanguageEncoderType.DISTILBERT_BASE.value,
        ],
    )
    def test_stores_configuration(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        pooling_method: str,
        max_token_len: int,
        model_name: str,
    ):
        encoder = language_encoder_factory(
            pooling_method=pooling_method,
            max_token_len=max_token_len,
            model_name=model_name,
        )
        assert encoder.pooling_method == pooling_method
        assert encoder.max_token_len == max_token_len
        assert encoder.model_name == model_name
        assert encoder.language_key == SampleKey.TOKENIZED_OBSERVATIONS.value
        assert encoder.feature_dim == HIDDEN_SIZE

    @pytest.mark.parametrize(
        "use_embeddings_only, pooling_method, expectation",
        [
            (False, PoolingMethod.DEFAULT.value, does_not_raise()),
            (False, PoolingMethod.NONE.value, does_not_raise()),
            (True, PoolingMethod.NONE.value, does_not_raise()),
            (
                True,
                PoolingMethod.DEFAULT.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "use_embeddings_only=True is only compatible with "
                        "pooling_method=PoolingMethod.NONE"
                    ),
                ),
            ),
            (
                True,
                PoolingMethod.AVERAGE.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "use_embeddings_only=True is only compatible with "
                        "pooling_method=PoolingMethod.NONE"
                    ),
                ),
            ),
        ],
    )
    def test_embeddings_only_pooling_validation(
        self,
        use_embeddings_only: bool,
        pooling_method: str,
        expectation,
    ):
        with (
            expectation,
            patch.object(LanguageEncoder, "_build_encoder", _mock_build_encoder),
        ):
            LanguageEncoder(
                pretrained=False,
                frozen=False,
                pooling_method=pooling_method,
                use_embeddings_only=use_embeddings_only,
            )

    def test_requires_tokenized_specification(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        assert encoder.input_specification.requires_tokenized is True

    def test_padding_mask_name_format(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        expected = (
            f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"
        )
        assert encoder.padding_mask_name == expected


class TestLanguageEncoderPadTextInputs:
    def test_truncation_when_longer_than_max_token_len(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        max_token_len = 8
        encoder = language_encoder_factory(max_token_len=max_token_len)
        longer_sequence_length = 20
        text_ids = torch.from_numpy(
            rng.integers(
                low=0, high=VOCAB_SIZE, size=(2, longer_sequence_length)
            ).astype(np.int64)
        )
        mask = torch.zeros(2, longer_sequence_length, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=mask,
            max_length=max_token_len,
        )
        assert result_ids.shape[1] == max_token_len
        assert result_mask.shape[1] == max_token_len
        assert torch.equal(result_ids, text_ids[:, :max_token_len])

    def test_padding_when_shorter_than_max_token_len(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        max_token_len = 16
        encoder = language_encoder_factory(max_token_len=max_token_len)
        shorter_sequence_length = 5
        text_ids = torch.from_numpy(
            rng.integers(
                low=0, high=VOCAB_SIZE, size=(2, shorter_sequence_length)
            ).astype(np.int64)
        )
        mask = torch.zeros(2, shorter_sequence_length, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=mask,
            max_length=max_token_len,
        )
        assert result_ids.shape[1] == max_token_len
        assert result_mask.shape[1] == max_token_len
        # Original values preserved
        assert torch.equal(result_ids[:, :shorter_sequence_length], text_ids)
        # Padded region is zeros for ids
        assert torch.all(result_ids[:, shorter_sequence_length:] == 0)
        # Padded region is ones (True) for mask
        assert torch.all(result_mask[:, shorter_sequence_length:])

    def test_exact_length_unchanged(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        max_token_len = 10
        encoder = language_encoder_factory(max_token_len=max_token_len)
        text_ids = torch.from_numpy(
            rng.integers(low=0, high=VOCAB_SIZE, size=(2, max_token_len)).astype(
                np.int64
            )
        )
        mask = torch.zeros(2, max_token_len, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=mask,
            max_length=max_token_len,
        )
        assert torch.equal(result_ids, text_ids)
        assert torch.equal(result_mask, mask)

    def test_none_mask_with_padding(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        max_token_len = 16
        encoder = language_encoder_factory(max_token_len=max_token_len)
        text_ids = torch.from_numpy(
            rng.integers(low=0, high=VOCAB_SIZE, size=(2, 5)).astype(np.int64)
        )
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids,
            language_mask=None,
            max_length=max_token_len,
        )
        assert result_ids.shape[1] == max_token_len
        assert result_mask is None


class TestLanguageEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = language_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        encoder.use_embeddings_only = False
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.zeros(
            batch_size * time_steps, MAX_TOKEN_LEN, HIDDEN_SIZE
        )
        encoder.encoder.return_value = mock_output
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=8,
            time_steps=time_steps,
            include_padding_mask=True,
        )
        output = encoder(inputs=inputs)
        features = output[EncoderOutputKeys.LANGUAGE.value]
        assert features.shape == (batch_size, time_steps, HIDDEN_SIZE)

    def test_missing_language_key_raises(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"LanguageEncoder expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            encoder(inputs={"wrong_key": torch.zeros(2, 1, 10)})

    def test_embeddings_only_mode_uses_embedding_layer(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.NONE.value,
            use_embeddings_only=True,
        )
        batch_size = 2
        encoder.encoder.return_value = torch.zeros(
            batch_size, MAX_TOKEN_LEN, HIDDEN_SIZE
        )
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=MAX_TOKEN_LEN,
        )
        output = encoder(inputs=inputs)
        encoder.encoder.assert_called_once()
        assert EncoderOutputKeys.LANGUAGE.value in output

    def test_padding_mask_in_output(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = language_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        encoder.use_embeddings_only = False
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.zeros(
            batch_size, MAX_TOKEN_LEN, HIDDEN_SIZE
        )
        encoder.encoder.return_value = mock_output
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=MAX_TOKEN_LEN,
            include_padding_mask=True,
        )
        output = encoder(inputs=inputs)
        assert encoder.padding_mask_name in output

    def test_none_pooling_output_has_sequence_dimension(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = language_encoder_factory(pooling_method=PoolingMethod.NONE.value)
        encoder.use_embeddings_only = False
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.zeros(
            batch_size, MAX_TOKEN_LEN, HIDDEN_SIZE
        )
        encoder.encoder.return_value = mock_output
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=MAX_TOKEN_LEN,
        )
        output = encoder(inputs=inputs)
        features = output[EncoderOutputKeys.LANGUAGE.value]
        # num_prefix_tokens=1 (CLS) drops the first token
        expected_sequence_length = MAX_TOKEN_LEN - 1
        assert features.shape == (batch_size, 1, expected_sequence_length, HIDDEN_SIZE)

    def test_embeddings_only_strips_prefix_tokens_and_mask_aligns(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.NONE.value,
            use_embeddings_only=True,
        )
        encoder.encoder.return_value = torch.zeros(
            batch_size, MAX_TOKEN_LEN, HIDDEN_SIZE
        )
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=MAX_TOKEN_LEN,
        )
        output = encoder(inputs=inputs)
        features = output[EncoderOutputKeys.LANGUAGE.value]
        mask = output[encoder.padding_mask_name]
        expected_seq_len = MAX_TOKEN_LEN - encoder._num_prefix_tokens
        assert features.shape == (batch_size, 1, expected_seq_len, HIDDEN_SIZE)
        assert mask.shape[-1] == expected_seq_len


class TestLanguageEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                "LanguageEncoder cannot process image data for 'tokenized_observations'. "
                "Got CameraMetadata, expected tokenized text input.",
            ),
            (
                MagicMock(spec=BaseMetadata),
                None,
            ),
        ],
    )
    def test_validates_non_camera_metadata(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = language_encoder_factory()
        result = encoder.validate_input_metadata(
            key="tokenized_observations", metadata=metadata
        )
        assert result == expected_error


class TestLanguageEncoderGetVocabSize:
    def test_returns_config_vocab_size(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        assert encoder.get_vocab_size() == VOCAB_SIZE


class TestLanguageEncoderGetOutputSpecification:
    @pytest.mark.parametrize(
        "pooling_method, expected_dim_fn",
        [
            (PoolingMethod.DEFAULT.value, lambda enc: (HIDDEN_SIZE,)),
            (PoolingMethod.AVERAGE.value, lambda enc: (HIDDEN_SIZE,)),
            (
                PoolingMethod.NONE.value,
                lambda enc: (MAX_TOKEN_LEN - enc._num_prefix_tokens, HIDDEN_SIZE),
            ),
        ],
    )
    def test_output_dimension_matches_pooling_method(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        pooling_method: str,
        expected_dim_fn,
    ):
        encoder = language_encoder_factory(pooling_method=pooling_method)
        specification = encoder.get_output_specification()
        expected = expected_dim_fn(encoder)
        assert (
            next(
                m for m in specification if m.key == EncoderOutputKeys.LANGUAGE.value
            ).dimension
            == expected
        )

    def test_features_include_language_and_padding_mask(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert EncoderOutputKeys.LANGUAGE.value in feature_keys
        assert encoder.padding_mask_name in feature_keys
        assert len(feature_keys) == 2


class TestLanguageEncoderBuildEncoder:
    def test_missing_embedding_and_hidden_size_raises(self):
        model_name = "bert-base-uncased"
        mock_config = MagicMock(spec=[])
        mock_config.vocab_size = VOCAB_SIZE
        with (
            patch(
                "versatil.models.encoding.encoders.language.language.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"Config for {model_name} has neither "
                    f"'embedding_size' nor 'hidden_size'"
                ),
            ),
        ):
            LanguageEncoder(
                pretrained=False,
                frozen=False,
                pooling_method=PoolingMethod.NONE.value,
                model_name=model_name,
                use_embeddings_only=True,
            )


GATED_LANGUAGE_MODELS = {
    LanguageEncoderType.GEMMA_2B,
    LanguageEncoderType.QWEN_2_1_5B,
    LanguageEncoderType.LLAMA_3_2_1B,
}

NO_SDPA_LANGUAGE_MODELS = {
    LanguageEncoderType.DEBERTA_V3_BASE,
}

TIKTOKEN_LANGUAGE_MODELS = {
    LanguageEncoderType.DEBERTA_V3_BASE,
}


def _integration_marks(encoder_type: LanguageEncoderType) -> list:
    marks = []
    if encoder_type in GATED_LANGUAGE_MODELS:
        marks.append(
            pytest.mark.skipif(
                True,
                reason=f"{encoder_type.value} is a gated model requiring authentication",
            )
        )
    if encoder_type in TIKTOKEN_LANGUAGE_MODELS:
        marks.append(
            pytest.mark.xfail(
                reason=(
                    f"{encoder_type.value} tokenizer requires `tiktoken`, "
                    "which is not installed in the default environment"
                ),
                strict=False,
                raises=ValueError,
            )
        )
    return marks


class TestLanguageEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(
                encoder_type.value,
                marks=_integration_marks(encoder_type),
            )
            for encoder_type in LanguageEncoderType
        ],
    )
    def test_forward_pass_per_model(
        self,
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
        model_name: str,
    ):
        batch_size = 2
        no_sdpa_values = {m.value for m in NO_SDPA_LANGUAGE_MODELS}
        attention_type = (
            AttentionImplementation.EAGER.value
            if model_name in no_sdpa_values
            else AttentionImplementation.SDPA.value
        )
        encoder = LanguageEncoder(
            pretrained=False,
            frozen=False,
            pooling_method=PoolingMethod.DEFAULT.value,
            model_name=model_name,
            attention_type=attention_type,
        )
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=10,
        )
        output = encoder(inputs=inputs)
        features = output[EncoderOutputKeys.LANGUAGE.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    def test_pretrained_weights_differ_from_random_init(
        self,
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        model_name = LanguageEncoderType.ALBERT_BASE.value
        pretrained_encoder = LanguageEncoder(
            pretrained=True,
            frozen=False,
            pooling_method=PoolingMethod.DEFAULT.value,
            model_name=model_name,
        )
        random_encoder = LanguageEncoder(
            pretrained=False,
            frozen=False,
            pooling_method=PoolingMethod.DEFAULT.value,
            model_name=model_name,
        )
        pretrained_encoder.eval()
        random_encoder.eval()
        inputs = token_input_factory(batch_size=2, sequence_length=10)

        with torch.no_grad():
            pretrained_output = pretrained_encoder(inputs=inputs)
            random_output = random_encoder(inputs=inputs)

        pretrained_features = pretrained_output[EncoderOutputKeys.LANGUAGE.value]
        random_features = random_output[EncoderOutputKeys.LANGUAGE.value]
        assert not torch.allclose(pretrained_features, random_features, atol=1e-3)

    @pytest.mark.integration
    def test_frozen_pretrained_has_no_gradients(
        self,
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = LanguageEncoder(
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.DEFAULT.value,
            model_name=LanguageEncoderType.ALBERT_BASE.value,
        )
        for parameter in encoder.parameters():
            assert not parameter.requires_grad
