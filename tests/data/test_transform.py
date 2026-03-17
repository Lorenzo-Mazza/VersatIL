"""Tests for versatil.data.transform module."""

import logging
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import (
    BinaryGripperRange,
    GripperType,
    SampleKey,
)
from versatil.data.metadata import (
    ActionMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.transform import (
    detokenize_actions,
    normalize_actions,
    normalize_observation,
    normalize_sample,
    tokenize_actions,
    tokenize_observation,
    tokenize_sample,
    unnormalize_actions,
)


@pytest.fixture
def mock_normalizer() -> Callable[..., MagicMock]:
    """Factory for mock normalizers. normalize: x*2, unnormalize: x/2."""

    def factory(keys: list[str]) -> MagicMock:
        normalizer = MagicMock()
        normalizer.params_dict = MagicMock()
        normalizer.params_dict.keys.return_value = keys

        def make_single_field(key):
            single = MagicMock()
            single.normalize.side_effect = lambda x: x * 2
            single.unnormalize.side_effect = lambda x: x / 2
            return single

        normalizer.__getitem__ = lambda self, key: make_single_field(key)
        return normalizer

    return factory


@pytest.fixture
def mock_observation_tokenizer() -> Callable[..., MagicMock]:
    """Factory for mock observation tokenizers."""

    def factory(
        observation_keys: list[str],
        tokenized_output: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ) -> MagicMock:
        tokenizer = MagicMock()
        tokenizer.observation_keys = observation_keys
        if tokenized_output is None:
            tokenized_output = torch.tensor([1, 2, 3])
        if padding_mask is None:
            padding_mask = torch.tensor([False, False, False])
        tokenizer.tokenize.return_value = {
            SampleKey.TOKENIZED_OBSERVATIONS.value: tokenized_output,
            SampleKey.IS_PAD_OBSERVATION.value: padding_mask,
        }
        return tokenizer

    return factory


@pytest.fixture
def mock_action_tokenizer() -> Callable[..., MagicMock]:
    """Factory for mock action tokenizers."""

    def factory(
        encoded_output: torch.Tensor = None,
        decoded_output: np.ndarray = None,
        padding_mask: torch.Tensor = None,
    ) -> MagicMock:
        tokenizer = MagicMock()
        if encoded_output is None:
            encoded_output = torch.tensor([10, 20, 30])
        if padding_mask is None:
            padding_mask = torch.tensor([False, False, False])
        tokenizer.encode.return_value = {
            SampleKey.TOKENIZED_ACTIONS.value: encoded_output,
            SampleKey.IS_PAD_ACTION.value: padding_mask,
        }
        if decoded_output is not None:
            tokenizer.decode.return_value = decoded_output
        return tokenizer

    return factory


@pytest.fixture
def mock_tokenizer(
    mock_observation_tokenizer: Callable[..., MagicMock],
    mock_action_tokenizer: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    """Factory for mock Tokenizer wrapping observation and action tokenizers."""

    def factory(
        observation_keys: list[str] = None,
        decoded_output: np.ndarray = None,
        has_observation_tokenizer: bool = True,
        has_action_tokenizer: bool = True,
    ) -> MagicMock:
        tokenizer = MagicMock()
        tokenizer.observation_tokenizer = (
            mock_observation_tokenizer(observation_keys=observation_keys or [])
            if has_observation_tokenizer
            else None
        )
        tokenizer.action_tokenizer = (
            mock_action_tokenizer(decoded_output=decoded_output)
            if has_action_tokenizer
            else None
        )
        return tokenizer

    return factory


class TestNormalizeObservation:
    def test_normalizes_keys_present_in_normalizer(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        mock_normalizer: Callable[..., MagicMock],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=["position"])
        observation = {"position": torch.tensor([1.0, 2.0, 3.0])}

        result = normalize_observation(
            observation=observation,
            normalizer=normalizer,
            observation_space=observation_space,
        )

        torch.testing.assert_close(result["position"], torch.tensor([2.0, 4.0, 6.0]))

    def test_passes_through_keys_not_in_normalizer(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        mock_normalizer: Callable[..., MagicMock],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=[])
        original_tensor = torch.tensor([1.0, 2.0, 3.0])

        result = normalize_observation(
            observation={"position": original_tensor},
            normalizer=normalizer,
            observation_space=observation_space,
        )

        torch.testing.assert_close(result["position"], original_tensor)

    def test_does_not_modify_original_dict(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        mock_normalizer: Callable[..., MagicMock],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=["position"])
        original_tensor = torch.tensor([1.0, 2.0, 3.0])
        observation = {"position": original_tensor}

        normalize_observation(
            observation=observation,
            normalizer=normalizer,
            observation_space=observation_space,
        )

        torch.testing.assert_close(observation["position"], original_tensor)


class TestNormalizeActions:
    def test_normalizes_action_keys_present_in_normalizer(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_normalizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=["position"])

        result = normalize_actions(
            actions={"position": torch.tensor([1.0, 2.0, 3.0])},
            normalizer=normalizer,
            action_space=action_space,
        )

        torch.testing.assert_close(result["position"], torch.tensor([2.0, 4.0, 6.0]))

    def test_passes_through_keys_not_in_normalizer(
        self,
        action_space_factory: Callable[..., ActionSpace],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        mock_normalizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "gripper": gripper_action_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=[])
        original_tensor = torch.tensor([0.0, 1.0])

        result = normalize_actions(
            actions={"gripper": original_tensor},
            normalizer=normalizer,
            action_space=action_space,
        )

        torch.testing.assert_close(result["gripper"], original_tensor)


class TestUnnormalizeActions:
    def test_unnormalizes_action_keys_present_in_normalizer(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_normalizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=["position"])

        result = unnormalize_actions(
            normalized_actions={"position": torch.tensor([2.0, 4.0, 6.0])},
            normalizer=normalizer,
            action_space=action_space,
        )

        torch.testing.assert_close(result["position"], torch.tensor([1.0, 2.0, 3.0]))


class TestNormalizeSample:
    def test_normalizes_both_observations_and_actions(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        action_space_factory: Callable[..., ActionSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_normalizer: Callable[..., MagicMock],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=["position"])

        sample = {
            SampleKey.OBSERVATION.value: {"position": torch.tensor([1.0, 2.0, 3.0])},
            SampleKey.ACTION.value: {"position": torch.tensor([0.5, 1.0, 1.5])},
        }

        result = normalize_sample(
            sample=sample,
            normalizer=normalizer,
            observation_space=observation_space,
            action_space=action_space,
        )

        torch.testing.assert_close(
            result[SampleKey.OBSERVATION.value]["position"],
            torch.tensor([2.0, 4.0, 6.0]),
        )
        torch.testing.assert_close(
            result[SampleKey.ACTION.value]["position"],
            torch.tensor([1.0, 2.0, 3.0]),
        )

    def test_does_not_modify_original_sample(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        action_space_factory: Callable[..., ActionSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_normalizer: Callable[..., MagicMock],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        normalizer = mock_normalizer(keys=["position"])
        original_observation = torch.tensor([1.0, 2.0, 3.0])

        sample = {
            SampleKey.OBSERVATION.value: {"position": original_observation},
            SampleKey.ACTION.value: {"position": torch.tensor([0.5])},
        }

        normalize_sample(
            sample=sample,
            normalizer=normalizer,
            observation_space=observation_space,
            action_space=action_space,
        )

        torch.testing.assert_close(
            sample[SampleKey.OBSERVATION.value]["position"],
            original_observation,
        )


class TestTokenizeSample:
    def test_tokenizes_observations_when_tokenizer_present(
        self,
        mock_tokenizer: Callable[..., MagicMock],
    ):
        tokenizer = mock_tokenizer(
            observation_keys=["position"],
            has_action_tokenizer=False,
        )
        sample = {
            SampleKey.OBSERVATION.value: {"position": torch.tensor([1.0, 2.0, 3.0])},
            SampleKey.ACTION.value: {},
        }

        result = tokenize_sample(
            sample=sample,
            tokenizer=tokenizer,
            action_space=MagicMock(),
        )

        tokenizer.observation_tokenizer.tokenize.assert_called_once()
        assert (
            SampleKey.TOKENIZED_OBSERVATIONS.value
            in result[SampleKey.OBSERVATION.value]
        )
        assert SampleKey.IS_PAD_OBSERVATION.value in result[SampleKey.OBSERVATION.value]

    def test_tokenizes_actions_when_tokenizer_present(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_tokenizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        tokenizer = mock_tokenizer(
            has_observation_tokenizer=False,
            has_action_tokenizer=True,
        )
        sample = {
            SampleKey.OBSERVATION.value: {},
            SampleKey.ACTION.value: {
                "position": torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                )
            },
        }

        result = tokenize_sample(
            sample=sample,
            tokenizer=tokenizer,
            action_space=action_space,
        )

        tokenizer.action_tokenizer.encode.assert_called_once()
        assert SampleKey.TOKENIZED_ACTIONS.value in result[SampleKey.ACTION.value]

    def test_skips_both_when_tokenizers_are_none(
        self,
        mock_tokenizer: Callable[..., MagicMock],
    ):
        tokenizer = mock_tokenizer(
            has_observation_tokenizer=False,
            has_action_tokenizer=False,
        )
        sample = {
            SampleKey.OBSERVATION.value: {"position": torch.tensor([1.0])},
            SampleKey.ACTION.value: {"position": torch.tensor([2.0])},
        }

        result = tokenize_sample(
            sample=sample,
            tokenizer=tokenizer,
            action_space=MagicMock(),
        )

        assert (
            SampleKey.TOKENIZED_OBSERVATIONS.value
            not in result[SampleKey.OBSERVATION.value]
        )
        assert SampleKey.TOKENIZED_ACTIONS.value not in result[SampleKey.ACTION.value]


class TestTokenizeObservation:
    def test_tokenizes_matching_keys(
        self,
        mock_observation_tokenizer: Callable[..., MagicMock],
    ):
        tokenizer = mock_observation_tokenizer(observation_keys=["position"])
        observation = {"position": torch.tensor([1.0, 2.0, 3.0])}

        result = tokenize_observation(
            observation=observation,
            obs_tokenizer=tokenizer,
        )

        tokenizer.tokenize.assert_called_once()
        assert SampleKey.TOKENIZED_OBSERVATIONS.value in result
        assert SampleKey.IS_PAD_OBSERVATION.value in result
        # Original keys preserved
        assert "position" in result

    def test_raises_when_observation_key_missing(
        self,
        mock_observation_tokenizer: Callable[..., MagicMock],
    ):
        tokenizer = mock_observation_tokenizer(observation_keys=["missing_key"])

        with pytest.raises(KeyError, match="missing_key"):
            tokenize_observation(
                observation={"other_key": torch.tensor([1.0])},
                obs_tokenizer=tokenizer,
            )


class TestTokenizeActions:
    def test_concatenates_numerical_actions_and_encodes(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        tokenizer = mock_action_tokenizer()

        result = tokenize_actions(
            actions={
                "position": torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                )
            },
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        tokenizer.encode.assert_called_once()
        assert SampleKey.TOKENIZED_ACTIONS.value in result
        assert SampleKey.IS_PAD_ACTION.value in result

    def test_passes_existing_padding_mask_to_encoder(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        tokenizer = mock_action_tokenizer()
        padding_mask = torch.tensor([False, True, False, True])

        tokenize_actions(
            actions={
                "position": torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                ),
                SampleKey.IS_PAD_ACTION.value: padding_mask,
            },
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        call_kwargs = tokenizer.encode.call_args
        assert call_kwargs[1]["is_pad_mask"] is padding_mask

    def test_skips_non_numerical_and_non_prediction_head_actions(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        non_numerical = ActionMetadata(
            prediction_dimension=1,
            is_numerical=False,
            needs_normalization=False,
            dtype="string",
            is_precomputed=True,
            requires_prediction_head=False,
        )
        action_space = action_space_factory(
            actions_metadata={
                "language": non_numerical,
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        tokenizer = mock_action_tokenizer()

        tokenize_actions(
            actions={
                "position": torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                ),
                "language": torch.zeros(4, 1),
            },
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        # Only position (dim=3) should be concatenated
        encoded_tensor = tokenizer.encode.call_args[0][0]
        assert encoded_tensor.shape[-1] == 3

    def test_unsqueezes_one_dimensional_actions(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            }
        )
        tokenizer = mock_action_tokenizer()

        # 1D action tensor: (pred_horizon,) instead of (pred_horizon, dim)
        tokenize_actions(
            actions={"position": torch.tensor([1.0, 2.0, 3.0, 4.0])},
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        encoded_tensor = tokenizer.encode.call_args[0][0]
        assert encoded_tensor.ndim == 2


class TestDetokenizeActions:
    @pytest.mark.parametrize(
        "dimension, pred_horizon",
        [(3, 2), (6, 4), (1, 3)],
    )
    def test_splits_decoded_actions_by_metadata_dimension(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        mock_action_tokenizer: Callable[..., MagicMock],
        dimension: int,
        pred_horizon: int,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(
                        dimension=dimension
                    ),
                ),
            }
        )
        decoded = np.zeros((pred_horizon, dimension), dtype=np.float32)
        tokenizer = mock_action_tokenizer(decoded_output=decoded)

        result = detokenize_actions(
            action_tokens=torch.zeros(1, pred_horizon, dtype=torch.long),
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        assert "position" in result
        assert result["position"].shape == (1, pred_horizon, dimension)

    @pytest.mark.parametrize(
        "binary_range, raw_values, expected_values",
        [
            (BinaryGripperRange.ZERO_ONE.value, [0.7, 0.3], [1, 0]),
            (BinaryGripperRange.MINUS_ONE_ONE.value, [0.3, -0.7], [1, -1]),
        ],
    )
    def test_binary_gripper_rounding_by_range(
        self,
        action_space_factory: Callable[..., ActionSpace],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
        binary_range: str,
        raw_values: list[float],
        expected_values: list[int],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "gripper": gripper_action_metadata_factory(
                    gripper_type=GripperType.BINARY.value,
                    binary_gripper_range=binary_range,
                ),
            }
        )
        tokenizer = mock_action_tokenizer(
            decoded_output=np.array([[v] for v in raw_values], dtype=np.float32),
        )

        result = detokenize_actions(
            action_tokens=torch.zeros(1, len(raw_values), dtype=torch.long),
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        assert result["gripper"].dtype == torch.long
        for i, expected in enumerate(expected_values):
            assert result["gripper"][0, i, 0] == expected

    def test_on_the_fly_gripper_uses_source_metadata_for_rounding(
        self,
        action_space_factory: Callable[..., ActionSpace],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        gripper_source = gripper_observation_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        action_space = action_space_factory(
            actions_metadata={
                "gripper": on_the_fly_action_metadata_factory(
                    source_metadata=gripper_source,
                ),
            }
        )
        tokenizer = mock_action_tokenizer(
            decoded_output=np.array([[0.8]], dtype=np.float32),
        )

        result = detokenize_actions(
            action_tokens=torch.tensor([[10]]),
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        assert result["gripper"].dtype == torch.long
        assert result["gripper"][0, 0, 0] == 1

    def test_binary_gripper_unknown_range_warns_and_rounds(
        self,
        action_space_factory: Callable[..., ActionSpace],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        gripper_metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        # Override the range to an unrecognized value after construction
        gripper_metadata.binary_gripper_range = "unknown_range"

        action_space = action_space_factory(
            actions_metadata={
                "gripper": gripper_metadata,
            }
        )
        tokenizer = mock_action_tokenizer(
            decoded_output=np.array([[0.8]], dtype=np.float32),
        )

        with caplog.at_level(logging.WARNING):
            result = detokenize_actions(
                action_tokens=torch.tensor([[10]]),
                action_tokenizer=tokenizer,
                action_space=action_space,
            )

        assert "Gripper type is binary but range is not set" in caplog.text
        # Falls back to rounding: 0.8 → 1
        assert result["gripper"].dtype == torch.long
        assert result["gripper"][0, 0, 0] == 1

    def test_squeezes_trailing_singleton_dimension(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(dimension=3),
                ),
            }
        )
        tokenizer = mock_action_tokenizer(
            decoded_output=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        )

        # (batch=1, pred_horizon=1, 1) — trailing singleton
        result = detokenize_actions(
            action_tokens=torch.tensor([[[10]]]),
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        assert "position" in result

    def test_multi_key_split_preserves_order(
        self,
        action_space_factory: Callable[..., ActionSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        mock_action_tokenizer: Callable[..., MagicMock],
    ):
        """Actions are split in sorted key order matching tokenize_actions concatenation."""
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(dimension=3),
                ),
                "gripper": gripper_action_metadata_factory(
                    gripper_type=GripperType.CONTINUOUS.value,
                    prediction_dimension=1,
                ),
            }
        )
        # Sorted keys: "gripper" (dim=1), "position" (dim=3) → total=4
        tokenizer = mock_action_tokenizer(
            decoded_output=np.array([[10.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        )

        result = detokenize_actions(
            action_tokens=torch.tensor([[42]]),
            action_tokenizer=tokenizer,
            action_space=action_space,
        )

        assert result["gripper"].shape == (1, 1, 1)
        assert result["position"].shape == (1, 1, 3)
        # "gripper" comes first alphabetically
        torch.testing.assert_close(
            result["gripper"][0, 0, 0],
            torch.tensor(10.0),
        )
        torch.testing.assert_close(
            result["position"][0, 0],
            torch.tensor([1.0, 2.0, 3.0]),
        )
