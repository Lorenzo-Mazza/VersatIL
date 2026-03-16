"""Shared fixtures for decoding tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.metadata import ActionMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.action_heads.single_output import ActionHead


def _make_action_meta(
    requires_prediction_head: bool,
    prediction_dimension: int,
) -> MagicMock:
    """Create a mock ActionMetadata with the given attributes."""
    meta = MagicMock(spec=ActionMetadata)
    meta.requires_prediction_head = requires_prediction_head
    meta.prediction_dimension = prediction_dimension
    return meta


@pytest.fixture
def mock_action_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ActionSpace with configurable actions_metadata."""

    def factory(
        position_dim: int = 3,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
    ) -> MagicMock:
        metadata = {
            "position_action": _make_action_meta(
                requires_prediction_head=True,
                prediction_dimension=position_dim,
            ),
        }
        total_dim = position_dim
        if has_orientation:
            metadata["orientation_action"] = _make_action_meta(
                requires_prediction_head=True,
                prediction_dimension=orientation_dim,
            )
            total_dim += orientation_dim
        if has_gripper:
            metadata["gripper_action"] = _make_action_meta(
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
        has_position = position_dim > 0
        action_space.has_position_actions = has_position
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
def spatial_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for spatial feature dictionaries (B, C, H, W)."""

    def factory(
        batch_size: int = 2,
        channels: int = 256,
        height: int = 7,
        width: int = 7,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["rgb_features"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, channels, height, width)).astype(
                    np.float32
                )
            )
            for key in feature_keys
        }

    return factory


@pytest.fixture
def flat_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for flat feature dictionaries (B, D)."""

    def factory(
        batch_size: int = 2,
        feature_dim: int = 256,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["rgb_features"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, feature_dim)).astype(np.float32)
            )
            for key in feature_keys
        }

    return factory


@pytest.fixture
def temporal_spatial_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for temporal spatial feature dictionaries (B, T, C, H, W)."""

    def factory(
        batch_size: int = 2,
        observation_horizon: int = 2,
        channels: int = 256,
        height: int = 7,
        width: int = 7,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["rgb_features"]
        return {
            key: torch.from_numpy(
                rng.standard_normal(
                    (batch_size, observation_horizon, channels, height, width)
                ).astype(np.float32)
            )
            for key in feature_keys
        }

    return factory


@pytest.fixture
def sequential_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for 3D sequential feature dictionaries (B, Seq, D)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        feature_dimension: int = 64,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["seq_feature"]
        return {
            key: torch.from_numpy(
                rng.standard_normal(
                    (batch_size, sequence_length, feature_dimension)
                ).astype(np.float32)
            )
            for key in feature_keys
        }

    return factory


@pytest.fixture
def temporal_flat_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for 4D temporal-flat feature dictionaries (B, T, Seq, D)."""

    def factory(
        batch_size: int = 2,
        observation_horizon: int = 2,
        sequence_length: int = 4,
        feature_dimension: int = 64,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["temporal_seq_feature"]
        return {
            key: torch.from_numpy(
                rng.standard_normal(
                    (
                        batch_size,
                        observation_horizon,
                        sequence_length,
                        feature_dimension,
                    )
                ).astype(np.float32)
            )
            for key in feature_keys
        }

    return factory


@pytest.fixture
def action_head_factory() -> Callable[..., ActionHead]:
    """Factory for ActionHead instances."""

    def factory(
        input_dim: int = 64,
        blocks: list | None = None,
        output_dim: int | None = None,
    ) -> ActionHead:
        head = ActionHead(input_dim=input_dim, blocks=blocks)
        if output_dim is not None:
            head.set_output_dim(output_dim)
        return head

    return factory


@pytest.fixture
def gaussian_head_factory() -> Callable[..., GaussianHead]:
    """Factory for GaussianHead instances."""

    def factory(
        input_dim: int = 64,
        blocks: list | None = None,
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
        output_dim: int | None = None,
    ) -> GaussianHead:
        head = GaussianHead(
            input_dim=input_dim,
            blocks=blocks,
            min_logvar=min_logvar,
            max_logvar=max_logvar,
        )
        if output_dim is not None:
            head.set_output_dim(output_dim)
        return head

    return factory


@pytest.fixture
def action_heads_factory(
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., dict[str, ActionHead]]:
    """Factory for action heads dict matching a mock action space."""

    def factory(
        action_space: MagicMock,
        input_dim: int = 64,
    ) -> dict[str, ActionHead]:
        heads = {}
        for key, meta in action_space.actions_metadata.items():
            if meta.requires_prediction_head:
                heads[key] = action_head_factory(input_dim=input_dim)
        return heads

    return factory
