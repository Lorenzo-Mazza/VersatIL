"""Tests for versatil.models.decoding.decoders.timestep_conditioning module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.timestep_conditioning import (
    extract_timestep_conditioning,
    filter_timestep_feature,
    validate_noisy_action_tensors,
)

BATCH_SIZE = 2
PREDICTION_HORIZON = 4
ACTION_DIMENSION = 3
EMBEDDING_DIMENSION = 8
DECODER_NAME = "TestDecoder"


@pytest.fixture
def timestep_action_heads_factory(
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., dict[str, ActionHead]]:
    """Factory for action heads used by timestep conditioning tests."""

    def factory(output_dim: int = ACTION_DIMENSION) -> dict[str, ActionHead]:
        return {
            "position_action": action_head_factory(
                input_dim=EMBEDDING_DIMENSION,
                output_dim=output_dim,
            )
        }

    return factory


class TestValidateNoisyActionTensors:
    def test_accepts_valid_action_tensors(
        self,
        timestep_action_heads_factory: Callable[..., dict[str, ActionHead]],
    ):
        action_heads = timestep_action_heads_factory()
        actions = {
            "position_action": torch.zeros(
                BATCH_SIZE, PREDICTION_HORIZON, ACTION_DIMENSION
            )
        }
        batch_size, action_device = validate_noisy_action_tensors(
            actions=actions,
            action_heads=action_heads,
            prediction_horizon=PREDICTION_HORIZON,
            decoder_name=DECODER_NAME,
        )
        assert batch_size == BATCH_SIZE
        assert action_device.type == "cpu"

    def test_raises_for_action_key_mismatch(
        self,
        timestep_action_heads_factory: Callable[..., dict[str, ActionHead]],
    ):
        action_heads = timestep_action_heads_factory()
        actions = {
            "wrong_action": torch.zeros(
                BATCH_SIZE, PREDICTION_HORIZON, ACTION_DIMENSION
            )
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "TestDecoder expected action keys "
                "['position_action'], got ['wrong_action']."
            ),
        ):
            validate_noisy_action_tensors(
                actions=actions,
                action_heads=action_heads,
                prediction_horizon=PREDICTION_HORIZON,
                decoder_name=DECODER_NAME,
            )

    def test_raises_for_invalid_action_rank(
        self,
        timestep_action_heads_factory: Callable[..., dict[str, ActionHead]],
    ):
        action_heads = timestep_action_heads_factory()
        actions = {"position_action": torch.zeros(BATCH_SIZE, ACTION_DIMENSION)}
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action 'position_action' must have shape "
                "(B, prediction_horizon, action_dim), got "
                "torch.Size([2, 3])."
            ),
        ):
            validate_noisy_action_tensors(
                actions=actions,
                action_heads=action_heads,
                prediction_horizon=PREDICTION_HORIZON,
                decoder_name=DECODER_NAME,
            )


class TestExtractTimestepConditioning:
    def test_extracts_vector_timestep(
        self,
    ):
        timesteps = torch.arange(BATCH_SIZE)
        features = {DecoderOutputKey.TIMESTEP.value: timesteps}
        extracted = extract_timestep_conditioning(
            features=features,
            batch_size=BATCH_SIZE,
            action_device=torch.device("cpu"),
        )
        assert extracted is timesteps

    def test_squeezes_column_timestep(
        self,
    ):
        features = {
            DecoderOutputKey.TIMESTEP.value: torch.arange(BATCH_SIZE).unsqueeze(-1)
        }
        extracted = extract_timestep_conditioning(
            features=features,
            batch_size=BATCH_SIZE,
            action_device=torch.device("cpu"),
        )
        assert extracted.shape == (BATCH_SIZE,)

    def test_raises_for_missing_timestep(
        self,
    ):
        features = {"rgb_features": torch.zeros(BATCH_SIZE, EMBEDDING_DIMENSION)}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
                "The algorithm should inject timesteps into features."
            ),
        ):
            extract_timestep_conditioning(
                features=features,
                batch_size=BATCH_SIZE,
                action_device=torch.device("cpu"),
            )

    def test_raises_for_invalid_timestep_shape(
        self,
    ):
        features = {DecoderOutputKey.TIMESTEP.value: torch.zeros(BATCH_SIZE, 2)}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"'{DecoderOutputKey.TIMESTEP.value}' must have shape "
                "(B,) or (B, 1), got torch.Size([2, 2])."
            ),
        ):
            extract_timestep_conditioning(
                features=features,
                batch_size=BATCH_SIZE,
                action_device=torch.device("cpu"),
            )


class TestFilterTimestepFeature:
    def test_returns_new_dict_without_timestep(
        self,
    ):
        timestep = torch.arange(BATCH_SIZE)
        observation = torch.zeros(BATCH_SIZE, EMBEDDING_DIMENSION)
        features = {
            DecoderOutputKey.TIMESTEP.value: timestep,
            "rgb_features": observation,
        }
        filtered = filter_timestep_feature(features=features)
        assert filtered == {"rgb_features": observation}
        assert DecoderOutputKey.TIMESTEP.value in features
