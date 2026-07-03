"""Shared fixtures for decoder factory integration tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.models.decoding.constants import AlgorithmContextKey
from versatil.models.encoding.encoders.constants import EncoderOutputKeys


@pytest.fixture
def mock_tokenizer_factory() -> Callable[..., MagicMock]:
    """Factory for mock Tokenizer with configurable vocab size."""

    def factory(vocab_size: int = 32, max_token_len: int = 256) -> MagicMock:
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = MagicMock(spec=ActionTokenizer)
        eos_token_id = vocab_size
        effective_vocab_size = vocab_size + 1
        tokenizer.action_tokenizer.vocab_size = effective_vocab_size
        tokenizer.action_tokenizer.eos_token_id = eos_token_id
        tokenizer.action_tokenizer.max_token_len = max_token_len
        return tokenizer

    return factory


@pytest.fixture
def tokenized_actions_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for tokenized action dictionaries."""

    def factory(
        batch_size: int = 2,
        action_token_length: int = 8,
        vocab_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        token_ids = torch.from_numpy(
            rng.integers(
                low=0, high=vocab_size, size=(batch_size, action_token_length)
            ).astype(np.int64)
        )
        return {SampleKey.TOKENIZED_ACTIONS.value: token_ids}

    return factory


@pytest.fixture
def tokenized_text_features_factory() -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for tokenized language observation feature dictionaries."""

    def factory(
        batch_size: int = 2,
        text_token_length: int = 4,
        vocab_size: int = 64,
        include_padding_mask: bool = True,
        padded_last_token: bool = False,
    ) -> dict[str, torch.Tensor]:
        token_ids = torch.arange(batch_size * text_token_length).reshape(
            batch_size,
            text_token_length,
        )
        features = {
            SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids.remainder(vocab_size).to(
                torch.long
            )
        }
        if include_padding_mask:
            padding_mask = torch.zeros(
                batch_size,
                text_token_length,
                dtype=torch.bool,
            )
            if padded_last_token:
                padding_mask[:, -1] = True
            features[SampleKey.IS_PAD_OBSERVATION.value] = padding_mask
        return features

    return factory


@pytest.fixture
def encoded_sequence_features_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for encoded sequential observation feature dictionaries."""

    def factory(
        key: str = "vision_features",
        batch_size: int = 2,
        feature_token_length: int = 5,
        feature_dimension: int = 12,
        include_padding_mask: bool = False,
        padded_last_token: bool = True,
    ) -> dict[str, torch.Tensor]:
        feature_values = rng.standard_normal(
            size=(batch_size, feature_token_length, feature_dimension),
        )
        features = {
            key: torch.as_tensor(feature_values, dtype=torch.float32),
        }
        if include_padding_mask:
            padding_mask = torch.zeros(
                batch_size,
                feature_token_length,
                dtype=torch.bool,
            )
            if padded_last_token:
                padding_mask[:, -1] = True
            features[f"{key}_{EncoderOutputKeys.PADDING_MASK.value}"] = padding_mask
        return features

    return factory


@pytest.fixture
def spatial_features_with_timestep_factory(
    spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for spatial features (B, C, H, W) with a timestep key."""

    def factory(
        batch_size: int = 2,
        channels: int = 256,
        height: int = 7,
        width: int = 7,
        feature_keys: list[str] | None = None,
        include_timestep: bool = True,
    ) -> dict[str, torch.Tensor]:
        features = spatial_feature_factory(
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            feature_keys=feature_keys,
        )
        if include_timestep:
            features[AlgorithmContextKey.TIMESTEP.value] = torch.from_numpy(
                rng.standard_normal((batch_size,)).astype(np.float32)
            )
        return features

    return factory


@pytest.fixture
def flat_features_with_timestep_factory(
    flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for flat features (B, D) with a timestep key."""

    def factory(
        batch_size: int = 2,
        feature_dim: int = 256,
        feature_keys: list[str] | None = None,
        include_timestep: bool = True,
        timestep_shape: tuple[int, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=feature_dim,
            feature_keys=feature_keys,
        )
        if include_timestep:
            if timestep_shape is None:
                timestep_shape = (batch_size,)
            features[AlgorithmContextKey.TIMESTEP.value] = torch.from_numpy(
                rng.integers(low=0, high=100, size=timestep_shape).astype(np.int64)
            )
        return features

    return factory


@pytest.fixture
def noisy_actions_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for noisy action dictionaries with shape (B, T, D)."""

    def factory(
        batch_size: int = 2,
        prediction_horizon: int = 4,
        action_keys_to_dims: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_keys_to_dims is None:
            action_keys_to_dims = {"position_action": 3}
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, prediction_horizon, dim)).astype(
                    np.float32
                )
            )
            for key, dim in action_keys_to_dims.items()
        }

    return factory
