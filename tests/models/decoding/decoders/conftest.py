"""Shared fixtures for decoder factory integration tests."""
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.constants import DecoderOutputKey


@pytest.fixture
def mock_action_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ActionSpace with configurable actions_metadata."""

    class _MockMeta:
        def __init__(
            self, requires_prediction_head: bool, prediction_dimension: int
        ):
            self.requires_prediction_head = requires_prediction_head
            self.prediction_dimension = prediction_dimension

    def factory(
        position_dim: int = 3,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
    ) -> MagicMock:
        metadata = {
            "position_action": _MockMeta(
                requires_prediction_head=True,
                prediction_dimension=position_dim,
            ),
        }
        total_dim = position_dim
        if has_orientation:
            metadata["orientation_action"] = _MockMeta(
                requires_prediction_head=True,
                prediction_dimension=orientation_dim,
            )
            total_dim += orientation_dim
        if has_gripper:
            metadata["gripper_action"] = _MockMeta(
                requires_prediction_head=True,
                prediction_dimension=gripper_dim,
            )
            total_dim += gripper_dim
        action_space = MagicMock(spec=ActionSpace)
        action_space.actions_metadata = metadata
        action_space.get_total_action_dim.return_value = total_dim
        action_space.has_gripper_actions = has_gripper
        action_space.gripper_dim = gripper_dim
        action_space.has_orientation_actions = has_orientation
        action_space.orientation_dim = orientation_dim
        action_space.has_position_actions = True
        action_space.position_dim = position_dim
        return action_space

    return factory


@pytest.fixture
def mock_observation_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ObservationSpace."""

    def factory() -> MagicMock:
        return MagicMock(spec=ObservationSpace)

    return factory


@pytest.fixture
def mock_tokenizer_factory() -> Callable[..., MagicMock]:
    """Factory for mock Tokenizer with configurable vocab size."""

    def factory(vocab_size: int = 32) -> MagicMock:
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = MagicMock()
        tokenizer.action_tokenizer.vocab_size = vocab_size
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
            features[DecoderOutputKey.TIMESTEP.value] = torch.from_numpy(
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
            features[DecoderOutputKey.TIMESTEP.value] = torch.from_numpy(
                rng.integers(
                    low=0, high=100, size=timestep_shape
                ).astype(np.int64)
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
                rng.standard_normal(
                    (batch_size, prediction_horizon, dim)
                ).astype(np.float32)
            )
            for key, dim in action_keys_to_dims.items()
        }

    return factory