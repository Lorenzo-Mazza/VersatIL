"""Tests for versatil.configs.data.tokenizer module."""

import pytest

from versatil.configs.data.tokenizer import (
    ActionDiscretizerConfig,
    ActionTokenIdMappingConfig,
    ActionTokenizationConfig,
    ObservationTokenizationConfig,
    TokenizationConfig,
)
from versatil.data.constants import (
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    BinningStrategy,
)
from versatil.models.encoding.encoders.constants import LanguageEncoderType


@pytest.mark.unit
class TestObservationTokenizationConfig:
    def test_tokenizer_model_default_is_language_encoder_string(self):
        config = ObservationTokenizationConfig()
        assert config.tokenizer_model == LanguageEncoderType.BERT_BASE.value

    def test_observation_keys_default_to_empty_list(self):
        config = ObservationTokenizationConfig()
        assert config.observation_keys == []

    @pytest.mark.parametrize("bin_continuous_data", [True, False])
    @pytest.mark.parametrize("num_bins", [128, 512])
    def test_stores_binning_configuration(self, bin_continuous_data, num_bins):
        config = ObservationTokenizationConfig(
            bin_continuous_data=bin_continuous_data, num_bins=num_bins
        )
        assert config.bin_continuous_data == bin_continuous_data
        assert config.num_bins == num_bins

    @pytest.mark.parametrize("max_token_len", [128, 512])
    def test_stores_max_token_len(self, max_token_len):
        config = ObservationTokenizationConfig(max_token_len=max_token_len)
        assert config.max_token_len == max_token_len

    @pytest.mark.parametrize("raw_text", [True, False])
    def test_stores_raw_text(self, raw_text):
        config = ObservationTokenizationConfig(raw_text=raw_text)
        assert config.raw_text == raw_text

    def test_raw_text_default_is_false(self):
        config = ObservationTokenizationConfig()
        assert config.raw_text is False


@pytest.mark.unit
class TestActionTokenizationConfig:
    def test_action_discretizer_default_is_fast(self):
        config = ActionTokenizationConfig()
        assert config.action_discretizer.type == ActionDiscretizerType.FAST.value

    @pytest.mark.parametrize("use_pretrained", [True, False])
    def test_stores_pretrained_fast_flag(self, use_pretrained):
        config = ActionTokenizationConfig()
        config.action_discretizer.use_pretrained = use_pretrained
        assert config.action_discretizer.use_pretrained == use_pretrained

    @pytest.mark.parametrize(
        "tokenizer_model", ["physical-intelligence/fast", "custom"]
    )
    def test_stores_fast_tokenizer_model(self, tokenizer_model):
        config = ActionTokenizationConfig(
            action_discretizer=ActionDiscretizerConfig(
                type=ActionDiscretizerType.FAST.value,
                tokenizer_model=tokenizer_model,
            )
        )
        assert config.action_discretizer.tokenizer_model == tokenizer_model

    @pytest.mark.parametrize("num_bins", [64, 256])
    def test_stores_binned_action_discretizer(self, num_bins):
        config = ActionTokenizationConfig(
            action_discretizer=ActionDiscretizerConfig(
                type=ActionDiscretizerType.BINNED.value,
                num_bins=num_bins,
            )
        )
        assert config.action_discretizer.type == ActionDiscretizerType.BINNED.value
        assert config.action_discretizer.num_bins == num_bins

    @pytest.mark.parametrize(
        ("num_bins", "min_value", "max_value"),
        [
            (64, -1.0, 1.0),
            (256, -2.0, 2.0),
        ],
    )
    def test_stores_uniform_binning_action_discretizer(
        self,
        num_bins,
        min_value,
        max_value,
    ):
        config = ActionTokenizationConfig(
            action_discretizer=ActionDiscretizerConfig(
                type=ActionDiscretizerType.BINNED.value,
                binning_strategy=BinningStrategy.UNIFORM.value,
                num_bins=num_bins,
                min_value=min_value,
                max_value=max_value,
            )
        )
        assert config.action_discretizer.type == ActionDiscretizerType.BINNED.value
        assert (
            config.action_discretizer.binning_strategy == BinningStrategy.UNIFORM.value
        )
        assert config.action_discretizer.num_bins == num_bins
        assert config.action_discretizer.min_value == min_value
        assert config.action_discretizer.max_value == max_value

    def test_token_id_mapping_default_is_identity(self):
        config = ActionTokenizationConfig()
        assert config.token_id_mapping.type == ActionTokenIdMappingType.IDENTITY.value

    def test_stores_language_token_id_mapping(self):
        config = ActionTokenizationConfig()
        config.token_id_mapping.type = (
            ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value
        )
        config.token_id_mapping.language_tokenizer_model = "google/gemma-2b"
        config.token_id_mapping.num_special_tokens_to_skip = 64
        assert (
            config.token_id_mapping.type
            == ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value
        )
        assert config.token_id_mapping.language_tokenizer_model == "google/gemma-2b"
        assert config.token_id_mapping.num_special_tokens_to_skip == 64

    @pytest.mark.parametrize("max_token_len", [64, 128])
    def test_stores_action_max_token_len(self, max_token_len):
        config = ActionTokenizationConfig(max_token_len=max_token_len)
        assert config.max_token_len == max_token_len

    def test_stores_language_mapping_config_directly(self):
        config = ActionTokenIdMappingConfig(
            type=ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value,
            language_tokenizer_model="google/gemma-2b",
            num_special_tokens_to_skip=32,
        )
        assert config.type == ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value
        assert config.language_tokenizer_model == "google/gemma-2b"
        assert config.num_special_tokens_to_skip == 32


@pytest.mark.unit
class TestTokenizationConfig:
    @pytest.mark.parametrize("tokenize_observations", [True, False])
    @pytest.mark.parametrize("tokenize_actions", [True, False])
    def test_stores_tokenization_flags(self, tokenize_observations, tokenize_actions):
        config = TokenizationConfig(
            tokenize_observations=tokenize_observations,
            tokenize_actions=tokenize_actions,
        )
        assert config.tokenize_observations == tokenize_observations
        assert config.tokenize_actions == tokenize_actions

    def test_tokenizer_configs_default_to_none(self):
        config = TokenizationConfig()
        assert config.observation_tokenizer is None
        assert config.action_tokenizer is None
