"""Tests for versatil.models.encoding.encoders.language.language module."""
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from versatil.data.constants import SampleKey
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    LanguageEncoderType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.language.language import LanguageEncoder
from versatil.models.encoding.encoders.unconditional import Encoder


HIDDEN_SIZE = 768
VOCAB_SIZE = 30522
MAX_TOKEN_LEN = 16


def _mock_build_encoder(self):
    """Side-effect to set self.encoder with expected attributes."""
    self.encoder = MagicMock()
    self.encoder.parameters.return_value = iter(
        [torch.nn.Parameter(torch.zeros(1))]
    )
    self.encoder.device = torch.device("cpu")
    self.encoder.embedding_dim = HIDDEN_SIZE
    self.config = MagicMock()
    self.config.hidden_size = HIDDEN_SIZE
    self.config.vocab_size = VOCAB_SIZE


def _mock_setup_pooling(self):
    """Side-effect to set pooling-related attributes."""
    self.pooling_head = None
    if self.pooling_method == PoolingMethod.NONE.value:
        self.output_dim = (self.max_token_len, self.feature_dim)
        self.padding_dim = self.max_token_len
    else:
        self.output_dim = self.feature_dim
        self.padding_dim = 1


@pytest.fixture
def language_encoder_factory() -> Callable[..., LanguageEncoder]:
    """Factory for LanguageEncoder with mocked backbone and pooling."""

    def factory(
        pretrained: bool = False,
        frozen: bool = False,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        model_name: str = LanguageEncoderType.BERT_BASE.value,
        max_token_len: int = MAX_TOKEN_LEN,
        use_embeddings_only: bool = False,
    ) -> LanguageEncoder:
        with (
            patch.object(LanguageEncoder, "_build_encoder", _mock_build_encoder),
            patch.object(LanguageEncoder, "_setup_pooling", _mock_setup_pooling),
        ):
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
        time_steps: int | None = None,
        include_padding_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        if time_steps is not None:
            shape = (batch_size, time_steps, sequence_length)
        else:
            shape = (batch_size, sequence_length)
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

    def test_inherits_from_encoder(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        assert isinstance(encoder, Encoder)

    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.DEFAULT.value,
        PoolingMethod.AVERAGE.value,
    ])
    @pytest.mark.parametrize("max_token_len", [16, 64])
    @pytest.mark.parametrize("model_name", [
        LanguageEncoderType.BERT_BASE.value,
        LanguageEncoderType.DISTILBERT_BASE.value,
    ])
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
                pytest.raises(ValueError, match="use_embeddings_only=True"),
            ),
            (
                True,
                PoolingMethod.AVERAGE.value,
                pytest.raises(ValueError, match="use_embeddings_only=True"),
            ),
        ],
    )
    def test_embeddings_only_pooling_validation(
        self,
        use_embeddings_only: bool,
        pooling_method: str,
        expectation,
    ):
        with expectation:
            with (
                patch.object(
                    LanguageEncoder, "_build_encoder", _mock_build_encoder
                ),
                patch.object(
                    LanguageEncoder, "_setup_pooling", _mock_setup_pooling
                ),
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
            f"{EncoderOutputKeys.LANGUAGE.value}"
            f"_{EncoderOutputKeys.PADDING_MASK.value}"
        )
        assert encoder.padding_mask_name == expected


class TestLanguageEncoderPoolFeatures:

    def test_default_pooling_returns_cls_token(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
        batch_size = 2
        sequence_length = 10
        hidden_state = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, HIDDEN_SIZE)).astype(
                np.float32
            )
        )
        outputs = BaseModelOutput(last_hidden_state=hidden_state)
        result = encoder._pool_features(outputs=outputs)
        assert result.shape == (batch_size, HIDDEN_SIZE)
        assert torch.allclose(result, hidden_state[:, 0])

    def test_average_pooling_excludes_cls_token(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.AVERAGE.value
        )
        batch_size = 2
        sequence_length = 10
        hidden_state = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, HIDDEN_SIZE)).astype(
                np.float32
            )
        )
        outputs = BaseModelOutput(last_hidden_state=hidden_state)
        result = encoder._pool_features(outputs=outputs)
        expected = hidden_state[:, 1:].mean(dim=1)
        assert result.shape == (batch_size, HIDDEN_SIZE)
        assert torch.allclose(result, expected)

    def test_learned_aggregation_pooling(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value
        )
        mock_pooling_head = MagicMock()
        batch_size = 2
        sequence_length = 10
        mock_pooling_head.return_value = torch.zeros(batch_size, HIDDEN_SIZE)
        encoder.pooling_head = mock_pooling_head
        hidden_state = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, HIDDEN_SIZE)).astype(
                np.float32
            )
        )
        outputs = BaseModelOutput(last_hidden_state=hidden_state)
        result = encoder._pool_features(outputs=outputs)
        mock_pooling_head.assert_called_once()
        # Verify CLS token is excluded
        call_args = mock_pooling_head.call_args[0][0]
        assert torch.allclose(call_args, hidden_state[:, 1:])
        assert result.shape == (batch_size, HIDDEN_SIZE)

    def test_none_pooling_returns_full_hidden_state(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.NONE.value
        )
        batch_size = 2
        sequence_length = 10
        hidden_state = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, HIDDEN_SIZE)).astype(
                np.float32
            )
        )
        outputs = BaseModelOutput(last_hidden_state=hidden_state)
        result = encoder._pool_features(outputs=outputs)
        assert result.shape == (batch_size, sequence_length, HIDDEN_SIZE)
        assert torch.allclose(result, hidden_state)

    def test_invalid_pooling_method_raises(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        encoder = language_encoder_factory()
        encoder.pooling_method = "invalid_method"
        hidden_state = torch.from_numpy(
            rng.standard_normal((2, 10, HIDDEN_SIZE)).astype(np.float32)
        )
        outputs = BaseModelOutput(last_hidden_state=hidden_state)
        with pytest.raises(ValueError, match="Unsupported pooling method"):
            encoder._pool_features(outputs=outputs)

    def test_none_hidden_state_raises_runtime_error(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
        outputs = BaseModelOutput(last_hidden_state=None)
        with pytest.raises(RuntimeError, match="last_hidden_state must be present"):
            encoder._pool_features(outputs=outputs)

    def test_learned_aggregation_with_none_head_raises(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        rng: np.random.Generator,
    ):
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value
        )
        encoder.pooling_head = None
        hidden_state = torch.from_numpy(
            rng.standard_normal((2, 10, HIDDEN_SIZE)).astype(np.float32)
        )
        outputs = BaseModelOutput(last_hidden_state=hidden_state)
        with pytest.raises(RuntimeError, match="pooling_head must be initialized"):
            encoder._pool_features(outputs=outputs)


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
            text_input_ids=text_ids, language_mask=mask
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
            text_input_ids=text_ids, language_mask=mask
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
            rng.integers(
                low=0, high=VOCAB_SIZE, size=(2, max_token_len)
            ).astype(np.int64)
        )
        mask = torch.zeros(2, max_token_len, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=text_ids, language_mask=mask
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
            text_input_ids=text_ids, language_mask=None
        )
        assert result_ids.shape[1] == max_token_len
        assert result_mask is None


class TestLanguageEncoderForward:

    @pytest.mark.parametrize(
        "time_steps, expected_ndim",
        [
            (None, 2),
            (3, 3),
        ],
    )
    def test_output_shape_with_and_without_time(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
        expected_ndim: int,
    ):
        batch_size = 2
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
        encoder.use_embeddings_only = False
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.zeros(
            batch_size * (time_steps or 1), MAX_TOKEN_LEN, HIDDEN_SIZE
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
        assert features.ndim == expected_ndim
        assert features.shape[0] == batch_size
        if time_steps is not None:
            assert features.shape[1] == time_steps

    def test_missing_language_key_raises(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        with pytest.raises(ValueError, match="Expected key"):
            encoder(inputs={"wrong_key": torch.zeros(2, 10)})

    def test_non_tensor_input_raises(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        with pytest.raises(ValueError, match="tokenized_observations must be a tensor"):
            encoder(
                inputs={SampleKey.TOKENIZED_OBSERVATIONS.value: "not a tensor"}
            )

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
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.DEFAULT.value
        )
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
        encoder = language_encoder_factory(
            pooling_method=PoolingMethod.NONE.value
        )
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
        assert features.shape == (batch_size, MAX_TOKEN_LEN, HIDDEN_SIZE)


class TestLanguageEncoderGetVocabSize:

    def test_returns_config_vocab_size(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        assert encoder.get_vocab_size() == VOCAB_SIZE


class TestLanguageEncoderGetOutputSpecification:

    @pytest.mark.parametrize(
        "pooling_method, expected_dim",
        [
            (PoolingMethod.DEFAULT.value, HIDDEN_SIZE),
            (PoolingMethod.AVERAGE.value, HIDDEN_SIZE),
            (PoolingMethod.NONE.value, (MAX_TOKEN_LEN, HIDDEN_SIZE)),
        ],
    )
    def test_output_dimension_matches_pooling_method(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
        pooling_method: str,
        expected_dim: int | tuple,
    ):
        encoder = language_encoder_factory(pooling_method=pooling_method)
        specification = encoder.get_output_specification()
        assert specification.dimensions[EncoderOutputKeys.LANGUAGE.value] == expected_dim

    def test_features_include_language_and_padding_mask(
        self,
        language_encoder_factory: Callable[..., LanguageEncoder],
    ):
        encoder = language_encoder_factory()
        specification = encoder.get_output_specification()
        assert EncoderOutputKeys.LANGUAGE.value in specification.features
        assert encoder.padding_mask_name in specification.features
        assert len(specification.features) == 2


class TestLanguageEncoderIntegration:

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name",
        [encoder_type.value for encoder_type in LanguageEncoderType],
    )
    def test_forward_pass_per_model(
        self,
        token_input_factory: Callable[..., dict[str, torch.Tensor]],
        model_name: str,
    ):
        batch_size = 2
        encoder = LanguageEncoder(
            pretrained=False,
            frozen=False,
            pooling_method=PoolingMethod.DEFAULT.value,
            model_name=model_name,
        )
        inputs = token_input_factory(
            batch_size=batch_size,
            sequence_length=10,
        )
        output = encoder(inputs=inputs)
        features = output[EncoderOutputKeys.LANGUAGE.value]
        assert features.ndim == 2
        assert features.shape[0] == batch_size
