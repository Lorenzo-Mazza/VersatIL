"""Tests for versatil.configs.data.tokenizer module."""

import pytest

from versatil.configs.data.tokenizer import (
    ActionTokenizationConfig,
    ObservationTokenizationConfig,
    TokenizationConfig,
)
from versatil.data.constants import TokenizerType
from versatil.models.encoding.encoders.constants import LanguageEncoderType


@pytest.mark.unit
class TestObservationTokenizationConfig:
    def test_tokenizer_model_default_is_gemma_2b_string(self):
        config = ObservationTokenizationConfig()
        assert config.tokenizer_model == LanguageEncoderType.GEMMA_2B.value

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
    def test_tokenizer_chain_default_contains_fast(self):
        config = ActionTokenizationConfig()
        assert config.tokenizer_chain == [TokenizerType.FAST.value]

    @pytest.mark.parametrize("use_pretrained_fast", [True, False])
    def test_stores_pretrained_fast_flag(self, use_pretrained_fast):
        config = ActionTokenizationConfig(use_pretrained_fast=use_pretrained_fast)
        assert config.use_pretrained_fast == use_pretrained_fast

    def test_language_tokenizer_model_default_is_none(self):
        config = ActionTokenizationConfig()
        assert config.language_tokenizer_model is None

    def test_stores_chained_tokenizers(self):
        config = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="google/gemma-2b",
        )
        assert len(config.tokenizer_chain) == 2
        assert config.language_tokenizer_model == "google/gemma-2b"


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
