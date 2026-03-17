"""Tests for versatil.data.preprocessing.sampler module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from versatil.data.preprocessing.sampler import (
    SequenceSampler,
    create_indices,
    downsample_mask,
    get_val_mask,
)


@pytest.fixture
def mock_replay_buffer_factory() -> Callable[..., MagicMock]:
    """Factory for creating mock ReplayBuffer for SequenceSampler."""

    def factory(
        data: dict[str, np.ndarray],
        episode_ends: np.ndarray,
    ) -> MagicMock:
        buffer = MagicMock()
        buffer.episode_ends = MagicMock()
        buffer.episode_ends.__getitem__ = lambda mock_self, idx: episode_ends[idx]

        def getitem(mock_self, key):
            return data[key]

        buffer.__getitem__ = getitem
        buffer.keys.return_value = list(data.keys())
        return buffer

    return factory


class TestCreateIndices:
    def test_single_episode_no_padding(self):
        # 10 steps, seq_len=3 → starts at 0..7 → 8 sequences
        indices = create_indices(
            episode_ends=np.array([10]),
            sequence_length=3,
            episode_mask=np.array([True]),
            pad_before=0,
            pad_after=0,
        )

        assert indices.shape == (8, 4)
        np.testing.assert_array_equal(indices[0], [0, 3, 0, 3])
        np.testing.assert_array_equal(indices[7], [7, 10, 0, 3])

    def test_pad_before_allows_overhanging_start(self):
        indices = create_indices(
            episode_ends=np.array([5]),
            sequence_length=3,
            episode_mask=np.array([True]),
            pad_before=1,
            pad_after=0,
        )

        # First index: starts at -1 → buffer_start=0, sample_start=1
        assert indices[0, 0] == 0
        assert indices[0, 2] == 1

    def test_pad_after_allows_overhanging_end(self):
        indices = create_indices(
            episode_ends=np.array([5]),
            sequence_length=3,
            episode_mask=np.array([True]),
            pad_before=0,
            pad_after=1,
        )

        # Last index overhangs end → sample_end < sequence_length
        assert indices[-1, 3] < 3

    def test_pad_before_and_after_simultaneously(self):
        # 5 steps, seq_len=3, pad_before=1, pad_after=1
        # Starts from -1 to 3 → 5 sequences
        indices = create_indices(
            episode_ends=np.array([5]),
            sequence_length=3,
            episode_mask=np.array([True]),
            pad_before=1,
            pad_after=1,
        )

        assert indices.shape[0] == 5
        # First has pad at start
        assert indices[0, 2] == 1
        # Last has pad at end
        assert indices[-1, 3] < 3

    def test_multiple_episodes(self):
        # [0:5] and [5:10], each length 5, seq_len=3 → 3 each → 6 total
        indices = create_indices(
            episode_ends=np.array([5, 10]),
            sequence_length=3,
            episode_mask=np.array([True, True]),
            pad_before=0,
            pad_after=0,
        )

        assert indices.shape[0] == 6
        second_episode = indices[indices[:, 0] >= 5]
        assert len(second_episode) == 3
        np.testing.assert_array_equal(second_episode[0], [5, 8, 0, 3])

    def test_masked_episode_is_skipped(self):
        indices = create_indices(
            episode_ends=np.array([5, 10]),
            sequence_length=3,
            episode_mask=np.array([False, True]),
            pad_before=0,
            pad_after=0,
        )

        assert indices.shape[0] == 3
        assert np.all(indices[:, 0] >= 5)

    def test_skip_initial_excludes_first_n_steps(self):
        indices = create_indices(
            episode_ends=np.array([10]),
            sequence_length=3,
            episode_mask=np.array([True]),
            pad_before=0,
            pad_after=0,
            skip_initial=2,
        )

        # Effective [2:10], length 8, seq_len=3 → 6 starts
        assert indices.shape[0] == 6
        assert indices[0, 0] == 2

    def test_all_masked_returns_empty(self):
        indices = create_indices(
            episode_ends=np.array([5, 10]),
            sequence_length=3,
            episode_mask=np.array([False, False]),
        )

        assert indices.shape == (0, 4)

    def test_episode_shorter_than_sequence_returns_empty(self):
        indices = create_indices(
            episode_ends=np.array([2]),
            sequence_length=5,
            episode_mask=np.array([True]),
            pad_before=0,
            pad_after=0,
        )

        assert indices.shape[0] == 0

    def test_debug_false_skips_assertions(self):
        # Should not raise even though we can't easily trigger bad state
        indices = create_indices(
            episode_ends=np.array([10]),
            sequence_length=3,
            episode_mask=np.array([True]),
            debug=False,
        )

        assert indices.shape[0] > 0


class TestGetValMask:
    def test_returns_correct_count(self):
        mask = get_val_mask(n_episodes=10, val_ratio=0.3, seed=42)

        assert mask.dtype == bool
        assert mask.shape == (10,)
        assert np.sum(mask) == 3

    def test_zero_ratio_returns_all_false(self):
        mask = get_val_mask(n_episodes=10, val_ratio=0.0, seed=42)

        assert np.sum(mask) == 0

    def test_full_ratio_returns_all_true(self):
        mask = get_val_mask(n_episodes=5, val_ratio=1.0, seed=42)

        assert np.sum(mask) == 5

    def test_very_small_ratio_rounds_to_zero(self):
        # 5 episodes * 0.01 = 0.05 → rounds to 0
        mask = get_val_mask(n_episodes=5, val_ratio=0.01, seed=42)

        assert np.sum(mask) == 0

    def test_deterministic_with_same_seed(self):
        mask_a = get_val_mask(n_episodes=20, val_ratio=0.25, seed=123)
        mask_b = get_val_mask(n_episodes=20, val_ratio=0.25, seed=123)

        np.testing.assert_array_equal(mask_a, mask_b)

    def test_different_seeds_produce_different_masks(self):
        mask_a = get_val_mask(n_episodes=20, val_ratio=0.5, seed=0)
        mask_b = get_val_mask(n_episodes=20, val_ratio=0.5, seed=999)

        assert not np.array_equal(mask_a, mask_b)


class TestDownsampleMask:
    def test_reduces_true_count_to_max_n(self):
        mask = np.ones(100, dtype=bool)

        result = downsample_mask(mask=mask, max_n=10, seed=42)

        assert np.sum(result) == 10

    def test_no_change_when_count_below_max(self):
        mask = np.zeros(100, dtype=bool)
        mask[:5] = True

        result = downsample_mask(mask=mask, max_n=10, seed=42)

        np.testing.assert_array_equal(result, mask)

    def test_none_max_n_returns_unchanged(self):
        mask = np.ones(50, dtype=bool)

        result = downsample_mask(mask=mask, max_n=None, seed=42)

        np.testing.assert_array_equal(result, mask)

    def test_deterministic_with_same_seed(self):
        mask = np.ones(100, dtype=bool)

        result_a = downsample_mask(mask=mask, max_n=10, seed=42)
        result_b = downsample_mask(mask=mask, max_n=10, seed=42)

        np.testing.assert_array_equal(result_a, result_b)


class TestSequenceSamplerInitialization:
    def test_length_matches_number_of_possible_sequences(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={"position": np.arange(30, dtype=np.float32).reshape(10, 3)},
            episode_ends=np.array([10]),
        )

        sampler = SequenceSampler(replay_buffer=buffer, sequence_length=3)

        assert len(sampler) == 8

    def test_all_masked_returns_zero_length(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={"position": np.ones((5, 2), dtype=np.float32)},
            episode_ends=np.array([5]),
        )

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3,
            episode_mask=np.array([False]),
        )

        assert len(sampler) == 0

    def test_uses_all_keys_from_buffer_when_none_specified(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={
                "position": np.ones((10, 3), dtype=np.float32),
                "gripper": np.ones((10, 1), dtype=np.float32),
            },
            episode_ends=np.array([10]),
        )

        sampler = SequenceSampler(replay_buffer=buffer, sequence_length=3)

        assert set(sampler.keys) == {"position", "gripper"}


class TestSequenceSamplerSampleSequence:
    def test_returns_correct_values(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        # [[0,1,2], [3,4,5], ..., [27,28,29]]
        buffer = mock_replay_buffer_factory(
            data={"position": np.arange(30, dtype=np.float32).reshape(10, 3)},
            episode_ends=np.array([10]),
        )

        sampler = SequenceSampler(replay_buffer=buffer, sequence_length=3)
        # idx=2 → buffer [2:5]
        sample = sampler.sample_sequence(idx=2)

        np.testing.assert_array_equal(
            sample["position"],
            np.array([[6, 7, 8], [9, 10, 11], [12, 13, 14]], dtype=np.float32),
        )

    def test_returns_multiple_keys(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={
                "position": np.arange(30, dtype=np.float32).reshape(10, 3),
                "gripper": np.arange(10, dtype=np.float32).reshape(10, 1),
            },
            episode_ends=np.array([10]),
        )

        sampler = SequenceSampler(replay_buffer=buffer, sequence_length=3)
        sample = sampler.sample_sequence(idx=0)

        assert "position" in sample
        assert "gripper" in sample
        assert sample["position"].shape == (3, 3)
        assert sample["gripper"].shape == (3, 1)


class TestSequenceSamplerPadding:
    def test_pad_before_with_zeros(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={"position": np.ones((5, 2), dtype=np.float32) * 7.0},
            episode_ends=np.array([5]),
        )

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3,
            pad_before=1,
            pad_with_zeros=True,
        )
        sample = sampler.sample_sequence(idx=0)

        np.testing.assert_array_equal(sample["position"][0], [0.0, 0.0])
        np.testing.assert_array_equal(sample["position"][1], [7.0, 7.0])

    def test_pad_before_with_repeated_values(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={
                "position": np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
            },
            episode_ends=np.array([3]),
        )

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3,
            pad_before=1,
            pad_with_zeros=False,
        )
        sample = sampler.sample_sequence(idx=0)

        # Padded row repeats first value of actual data
        np.testing.assert_array_equal(sample["position"][0], [10, 20])

    def test_pad_after_with_repeated_values(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        buffer = mock_replay_buffer_factory(
            data={
                "position": np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
            },
            episode_ends=np.array([3]),
        )

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3,
            pad_after=1,
            pad_with_zeros=False,
        )
        # Last sequence overhangs end → last row padded with repeated last value
        last_index = len(sampler) - 1
        sample = sampler.sample_sequence(idx=last_index)

        np.testing.assert_array_equal(sample["position"][-1], [50, 60])


class TestSequenceSamplerKeyFirstK:
    @pytest.mark.parametrize(
        "dtype,fill_check",
        [
            (np.float32, lambda values: np.all(np.isnan(values))),
            (np.int32, lambda values: np.all(values == 0)),
        ],
    )
    def test_fills_remaining_with_correct_default(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
        dtype: np.dtype,
        fill_check: Callable,
    ):
        buffer = mock_replay_buffer_factory(
            data={"values": np.arange(10, dtype=dtype).reshape(10, 1)},
            episode_ends=np.array([10]),
        )

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=5,
            key_first_k={"values": 2},
        )
        sample = sampler.sample_sequence(idx=0)

        # First 2 rows have data, remaining filled with dtype default
        assert not fill_check(sample["values"][:2])
        assert fill_check(sample["values"][2:])

    def test_string_dtype_fills_with_empty_string(
        self,
        mock_replay_buffer_factory: Callable[..., MagicMock],
    ):
        string_data = np.array(
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], dtype="U10"
        ).reshape(10, 1)
        buffer = mock_replay_buffer_factory(
            data={"labels": string_data},
            episode_ends=np.array([10]),
        )

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=5,
            key_first_k={"labels": 2},
        )
        sample = sampler.sample_sequence(idx=0)

        # First 2 have data, rest are empty strings
        assert sample["labels"][0, 0] == "a"
        assert sample["labels"][1, 0] == "b"
        assert sample["labels"][2, 0] == ""
