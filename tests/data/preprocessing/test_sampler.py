"""Tests for SequenceSampler and related utilities.

Tests the sequence sampling functionality including index creation, padding,
episode masking, partial data loading, and various edge cases.
"""

import pytest
import numpy as np

from refactoring.data.preprocessing.replay_buffer import ReplayBuffer
from refactoring.data.preprocessing.sampler import (
    SequenceSampler,
    create_indices,
    get_val_mask,
    downsample_mask
)


@pytest.fixture
def simple_buffer():
    """Create a simple buffer with one episode of length 10."""
    buffer = ReplayBuffer.create_empty_numpy()
    data = {
        'observations': np.arange(10 * 4).reshape(10, 4).astype(np.float32),
        'actions': np.arange(10 * 2).reshape(10, 2).astype(np.float32),
        'rewards': np.arange(10).astype(np.float32),
    }
    buffer.add_episode(data)
    return buffer


@pytest.fixture
def multi_episode_buffer():
    """Create buffer with multiple episodes of varying lengths."""
    buffer = ReplayBuffer.create_empty_numpy()

    # Episode 1: length 10
    buffer.add_episode({
        'observations': np.arange(10 * 4).reshape(10, 4).astype(np.float32),
        'actions': np.arange(10 * 2).reshape(10, 2).astype(np.float32),
        'rewards': np.arange(10).astype(np.float32),
    })

    # Episode 2: length 7
    buffer.add_episode({
        'observations': np.arange(7 * 4).reshape(7, 4).astype(np.float32) + 100,
        'actions': np.arange(7 * 2).reshape(7, 2).astype(np.float32) + 100,
        'rewards': np.arange(7).astype(np.float32) + 100,
    })

    # Episode 3: length 5
    buffer.add_episode({
        'observations': np.arange(5 * 4).reshape(5, 4).astype(np.float32) + 200,
        'actions': np.arange(5 * 2).reshape(5, 2).astype(np.float32) + 200,
        'rewards': np.arange(5).astype(np.float32) + 200,
    })

    return buffer


class TestCreateIndices:
    """Test the create_indices function."""

    def test_basic_indices_no_padding(self):
        """Test basic index creation without padding."""
        episode_ends = np.array([10])
        episode_mask = np.array([True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=3,
            episode_mask=episode_mask
        )

        # Should have 8 sequences (0-2, 1-3, ..., 7-9)
        assert len(indices) == 8
        assert indices[0, 0] == 0  # buffer_start
        assert indices[0, 1] == 3  # buffer_end
        assert indices[-1, 0] == 7
        assert indices[-1, 1] == 10

    def test_indices_with_pad_before(self):
        """Test index creation with pad_before."""
        episode_ends = np.array([10])
        episode_mask = np.array([True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=3,
            episode_mask=episode_mask,
            pad_before=1
        )

        # Should have 9 sequences (one extra at start with padding)
        assert len(indices) == 9
        # First sequence should have padding at start
        assert indices[0, 2] == 1  # sample_start_idx (padding offset)

    def test_indices_with_pad_after(self):
        """Test index creation with pad_after."""
        episode_ends = np.array([10])
        episode_mask = np.array([True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=3,
            episode_mask=episode_mask,
            pad_after=1
        )

        # Should have 9 sequences (one extra at end with padding)
        assert len(indices) == 9
        # Last sequence should have padding at end
        assert indices[-1, 3] == 2  # sample_end_idx (less than sequence_length)

    def test_indices_skip_initial(self):
        """Test skipping initial steps."""
        episode_ends = np.array([10])
        episode_mask = np.array([True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=3,
            episode_mask=episode_mask,
            skip_initial=2
        )

        # Should start from index 2
        assert indices[0, 0] == 2  # buffer_start at skip position

    def test_indices_multiple_episodes(self):
        """Test index creation across multiple episodes."""
        episode_ends = np.array([10, 17, 22])
        episode_mask = np.array([True, True, True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=3,
            episode_mask=episode_mask
        )

        # Should have sequences from all episodes
        # Episode 1: 8 sequences, Episode 2: 5 sequences, Episode 3: 3 sequences
        assert len(indices) == 16

    def test_indices_episode_mask(self):
        """Test that episode_mask filters episodes."""
        episode_ends = np.array([10, 17, 22])
        episode_mask = np.array([True, False, True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=3,
            episode_mask=episode_mask
        )

        # Should only have sequences from episode 0 and 2
        # Episode 1 (length 10): 8 sequences, Episode 3 (length 5): 3 sequences
        assert len(indices) == 11

    def test_indices_short_episode(self):
        """Test with episode shorter than sequence_length."""
        episode_ends = np.array([2])
        episode_mask = np.array([True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=5,
            episode_mask=episode_mask,
            pad_before=2,
            pad_after=2
        )

        # Should still create valid indices with padding
        assert len(indices) > 0

    def test_indices_empty_result(self):
        """Test with conditions that produce no sequences."""
        # Very short episode with no padding allowance
        episode_ends = np.array([2])
        episode_mask = np.array([True])

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=10,
            episode_mask=episode_mask,
            pad_before=0,
            pad_after=0
        )

        # Should have no valid sequences
        assert len(indices) == 0


class TestGetValMask:
    """Test the get_val_mask function."""

    def test_val_mask_basic(self):
        """Test basic validation mask creation."""
        mask = get_val_mask(n_episodes=10, val_ratio=0.3, seed=42)

        assert len(mask) == 10
        assert mask.dtype == bool
        # Should have approximately 30% True
        assert 2 <= np.sum(mask) <= 4

    def test_val_mask_zero_ratio(self):
        """Test with zero validation ratio."""
        mask = get_val_mask(n_episodes=10, val_ratio=0.0, seed=42)

        assert len(mask) == 10
        assert np.sum(mask) == 0

    def test_val_mask_full_ratio(self):
        """Test with full validation ratio."""
        mask = get_val_mask(n_episodes=10, val_ratio=1.0, seed=42)

        assert len(mask) == 10
        assert np.sum(mask) == 10

    def test_val_mask_reproducible(self):
        """Test that same seed gives same results."""
        mask1 = get_val_mask(n_episodes=20, val_ratio=0.5, seed=123)
        mask2 = get_val_mask(n_episodes=20, val_ratio=0.5, seed=123)

        np.testing.assert_array_equal(mask1, mask2)

    def test_val_mask_different_seeds(self):
        """Test that different seeds give different results."""
        mask1 = get_val_mask(n_episodes=20, val_ratio=0.5, seed=123)
        mask2 = get_val_mask(n_episodes=20, val_ratio=0.5, seed=456)

        # Very unlikely to be identical
        assert not np.array_equal(mask1, mask2)


class TestDownsampleMask:
    """Test the downsample_mask function."""

    def test_downsample_basic(self):
        """Test basic downsampling."""
        mask = np.ones(100, dtype=bool)
        downsampled = downsample_mask(mask, max_n=10, seed=42)

        assert len(downsampled) == 100
        assert np.sum(downsampled) == 10

    def test_downsample_no_change(self):
        """Test when mask is already smaller than max_n."""
        mask = np.ones(5, dtype=bool)
        downsampled = downsample_mask(mask, max_n=10, seed=42)

        np.testing.assert_array_equal(mask, downsampled)

    def test_downsample_none(self):
        """Test with max_n=None (no downsampling)."""
        mask = np.ones(100, dtype=bool)
        downsampled = downsample_mask(mask, max_n=None, seed=42)

        np.testing.assert_array_equal(mask, downsampled)

    def test_downsample_reproducible(self):
        """Test reproducibility with same seed."""
        mask = np.ones(100, dtype=bool)
        down1 = downsample_mask(mask, max_n=10, seed=42)
        down2 = downsample_mask(mask, max_n=10, seed=42)

        np.testing.assert_array_equal(down1, down2)

    def test_downsample_preserves_false(self):
        """Test that False values stay False."""
        mask = np.array([True, False, True, False, True, True])
        downsampled = downsample_mask(mask, max_n=2, seed=42)

        # All originally False should still be False
        assert not downsampled[1]
        assert not downsampled[3]
        # Should have exactly 2 True values
        assert np.sum(downsampled) == 2


class TestSequenceSamplerCreation:
    """Test SequenceSampler initialization and basic properties."""

    def test_create_basic(self, simple_buffer):
        """Test basic sampler creation."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        assert len(sampler) > 0
        assert sampler.sequence_length == 3

    def test_create_with_padding(self, simple_buffer):
        """Test creation with padding parameters."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=1,
            pad_after=1
        )

        # Should have more sequences due to padding
        sampler_no_pad = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        assert len(sampler) > len(sampler_no_pad)

    def test_create_with_keys(self, simple_buffer):
        """Test creation with specific keys."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            keys=['observations', 'actions']
        )

        sample = sampler.sample_sequence(0)
        assert 'observations' in sample
        assert 'actions' in sample
        assert 'rewards' not in sample

    def test_create_with_episode_mask(self, multi_episode_buffer):
        """Test creation with episode mask."""
        # Mask out middle episode
        episode_mask = np.array([True, False, True])

        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            episode_mask=episode_mask
        )

        # Should have sequences from episodes 0 and 2 only
        sampler_all = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3
        )

        assert len(sampler) < len(sampler_all)

    def test_create_with_skip_initial(self, simple_buffer):
        """Test creation with skip_initial."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            skip_initial=2
        )

        # Should have fewer sequences
        sampler_no_skip = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        assert len(sampler) < len(sampler_no_skip)

    def test_create_empty_buffer(self):
        """Test creation with empty buffer."""
        buffer = ReplayBuffer.create_empty_numpy()

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3
        )

        assert len(sampler) == 0

    def test_length_calculation(self, simple_buffer):
        """Test __len__ returns correct count."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        # Episode length 10, sequence length 3 -> 8 sequences
        assert len(sampler) == 8


class TestSequenceSampling:
    """Test sampling sequences from the buffer."""

    def test_sample_basic(self, simple_buffer):
        """Test basic sequence sampling."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        sample = sampler.sample_sequence(0)

        assert 'observations' in sample
        assert 'actions' in sample
        assert 'rewards' in sample
        assert sample['observations'].shape == (3, 4)
        assert sample['actions'].shape == (3, 2)
        assert sample['rewards'].shape == (3,)

    def test_sample_values(self, simple_buffer):
        """Test that sampled values are correct."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        sample = sampler.sample_sequence(0)

        # First sequence should be steps 0-2
        expected_obs = np.arange(3 * 4).reshape(3, 4).astype(np.float32)
        np.testing.assert_array_equal(sample['observations'], expected_obs)

    def test_sample_different_indices(self, simple_buffer):
        """Test sampling different sequence indices."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        sample0 = sampler.sample_sequence(0)
        sample1 = sampler.sample_sequence(1)

        # Should be different (shifted by 1)
        assert not np.array_equal(sample0['observations'], sample1['observations'])
        # Second sample should start where first sample's second step is
        np.testing.assert_array_equal(
            sample0['observations'][1],
            sample1['observations'][0]
        )

    def test_sample_last_sequence(self, simple_buffer):
        """Test sampling the last available sequence."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        last_idx = len(sampler) - 1
        sample = sampler.sample_sequence(last_idx)

        # Last sequence should be steps 7-9
        expected_obs = np.arange(7 * 4, 10 * 4).reshape(3, 4).astype(np.float32)
        np.testing.assert_array_equal(sample['observations'], expected_obs)


class TestPaddingBehavior:
    """Test padding behavior with different configurations."""

    def test_pad_before_with_repeat(self, simple_buffer):
        """Test pad_before with repeated values."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=1,
            pad_with_zeros=False
        )

        # First sequence should have padding
        sample = sampler.sample_sequence(0)

        # First value should be repeated (padded)
        assert sample['observations'].shape == (3, 4)
        # The padding should repeat the first actual value
        np.testing.assert_array_equal(
            sample['observations'][0],
            sample['observations'][1]
        )

    def test_pad_before_with_zeros(self, simple_buffer):
        """Test pad_before with zeros."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=1,
            pad_with_zeros=True
        )

        sample = sampler.sample_sequence(0)

        # First value should be zero
        np.testing.assert_array_equal(
            sample['observations'][0],
            np.zeros(4, dtype=np.float32)
        )

    def test_pad_after_with_repeat(self, simple_buffer):
        """Test pad_after with repeated values."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_after=1,
            pad_with_zeros=False
        )

        # Last sequence should have padding
        last_idx = len(sampler) - 1
        sample = sampler.sample_sequence(last_idx)

        # Last value should be repeated
        np.testing.assert_array_equal(
            sample['observations'][-1],
            sample['observations'][-2]
        )

    def test_pad_after_with_zeros(self, simple_buffer):
        """Test pad_after with zeros."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_after=1,
            pad_with_zeros=True
        )

        last_idx = len(sampler) - 1
        sample = sampler.sample_sequence(last_idx)

        # Last value should be zero
        np.testing.assert_array_equal(
            sample['observations'][-1],
            np.zeros(4, dtype=np.float32)
        )

    def test_no_padding_needed(self, simple_buffer):
        """Test sequences that don't need padding."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=1,
            pad_after=1
        )

        # Middle sequences should have no padding
        middle_idx = len(sampler) // 2
        sample = sampler.sample_sequence(middle_idx)

        # Should be normal data, no repeated or zero values
        assert sample['observations'].shape == (3, 4)


class TestKeyFirstK:
    """Test key_first_k functionality for partial data loading."""

    def test_key_first_k_basic(self, simple_buffer):
        """Test loading only first k steps for a key."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=5,
            key_first_k={'observations': 2}
        )

        sample = sampler.sample_sequence(0)

        # First 2 observations should be real data
        expected_first_two = np.arange(2 * 4).reshape(2, 4).astype(np.float32)
        np.testing.assert_array_equal(
            sample['observations'][:2],
            expected_first_two
        )

        # Rest should be nan (for float dtype)
        assert np.all(np.isnan(sample['observations'][2:]))

    def test_key_first_k_integer_dtype(self, simple_buffer):
        """Test key_first_k with integer dtype uses 0 fill."""
        # Add integer data
        simple_buffer.data['int_data'] = np.arange(10, dtype=np.int32)

        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=5,
            keys=['int_data'],
            key_first_k={'int_data': 2}
        )

        sample = sampler.sample_sequence(0)

        # First 2 should be real data
        np.testing.assert_array_equal(sample['int_data'][:2], np.array([0, 1]))
        # Rest should be 0
        np.testing.assert_array_equal(sample['int_data'][2:], np.zeros(3, dtype=np.int32))

    def test_key_first_k_full_sequence(self, simple_buffer):
        """Test when first_k >= sequence length."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            key_first_k={'observations': 10}
        )

        sample = sampler.sample_sequence(0)

        # Should load all data (no partial loading)
        expected = np.arange(3 * 4).reshape(3, 4).astype(np.float32)
        np.testing.assert_array_equal(sample['observations'], expected)

    def test_key_first_k_multiple_keys(self, simple_buffer):
        """Test key_first_k with multiple keys."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=5,
            key_first_k={
                'observations': 2,
                'actions': 3
            }
        )

        sample = sampler.sample_sequence(0)

        # Observations should have 2 real + 3 nan
        assert not np.any(np.isnan(sample['observations'][:2]))
        assert np.all(np.isnan(sample['observations'][2:]))

        # Actions should have 3 real + 2 nan
        assert not np.any(np.isnan(sample['actions'][:3]))
        assert np.all(np.isnan(sample['actions'][3:]))

        # Rewards should be fully loaded (not in key_first_k)
        assert not np.any(np.isnan(sample['rewards']))


class TestMultiEpisode:
    """Test sampling across multiple episodes."""

    def test_episodes_separate(self, multi_episode_buffer):
        """Test that sequences don't cross episode boundaries."""
        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3
        )

        # Sample from different episodes
        # Episode 1 ends at 10, Episode 2 at 17, Episode 3 at 22
        # Find a sequence from each episode

        # Check episode boundaries
        episode_ends = multi_episode_buffer.episode_ends[:]
        assert episode_ends[0] == 10
        assert episode_ends[1] == 17
        assert episode_ends[2] == 22

    def test_episode_mask_filters(self, multi_episode_buffer):
        """Test that episode mask properly filters episodes."""
        # Only sample from first and last episode
        mask = np.array([True, False, True])

        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            episode_mask=mask
        )

        # Sample all sequences and verify they're from correct episodes
        for i in range(len(sampler)):
            sample = sampler.sample_sequence(i)
            # Values from episode 2 are in range [100, 199]
            # Make sure no samples contain those values
            assert not np.any((sample['observations'] >= 100) & (sample['observations'] < 200))

    def test_skip_initial_per_episode(self, multi_episode_buffer):
        """Test that skip_initial applies to each episode."""
        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            skip_initial=2
        )

        sampler_no_skip = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            skip_initial=0
        )

        # Should have fewer sequences
        assert len(sampler) < len(sampler_no_skip)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_sequence_length_one(self, simple_buffer):
        """Test with sequence length of 1."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=1
        )

        # Should have 10 sequences (one per step)
        assert len(sampler) == 10

        sample = sampler.sample_sequence(0)
        assert sample['observations'].shape == (1, 4)

    def test_sequence_length_equals_episode(self, simple_buffer):
        """Test when sequence length equals episode length."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=10
        )

        # Should have exactly 1 sequence
        assert len(sampler) == 1

        sample = sampler.sample_sequence(0)
        assert sample['observations'].shape == (10, 4)

    def test_sequence_longer_than_episode(self, simple_buffer):
        """Test when sequence length exceeds episode length."""
        # Simple buffer has episode of length 10
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=15,
            pad_before=0,
            pad_after=0
        )

        # Should have 0 sequences without padding (handled gracefully)
        assert len(sampler) == 0
        assert sampler.indices.shape == (0, 4)  # Correct empty shape

    def test_sequence_longer_with_padding(self, simple_buffer):
        """Test long sequence with enough padding."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=15,
            pad_before=5,
            pad_after=5
        )

        # Should be able to create sequences with padding
        assert len(sampler) > 0

    def test_very_short_episode(self):
        """Test with very short episode."""
        buffer = ReplayBuffer.create_empty_numpy()
        buffer.add_episode({
            'observations': np.array([[1, 2, 3]], dtype=np.float32),
            'actions': np.array([[1, 2]], dtype=np.float32),
        })

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=1
        )

        assert len(sampler) == 1
        sample = sampler.sample_sequence(0)
        assert sample['observations'].shape == (1, 3)

    def test_max_padding_clamped(self, simple_buffer):
        """Test that padding is clamped to sequence_length - 1."""
        # Request excessive padding
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=5,
            pad_before=10,  # Will be clamped to 4
            pad_after=10    # Will be clamped to 4
        )

        # Should still work and create valid sequences
        assert len(sampler) > 0
        sample = sampler.sample_sequence(0)
        assert sample['observations'].shape == (5, 4)

    def test_negative_padding_ignored(self, simple_buffer):
        """Test that negative padding is treated as 0."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=-5,
            pad_after=-3
        )

        # Should work with padding treated as 0
        sampler_no_pad = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=0,
            pad_after=0
        )

        assert len(sampler) == len(sampler_no_pad)

    def test_all_episodes_masked(self, multi_episode_buffer):
        """Test with all episodes masked out."""
        mask = np.array([False, False, False])

        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            episode_mask=mask
        )

        assert len(sampler) == 0
        # Should have correct empty shape
        assert sampler.indices.shape == (0, 4)

    def test_single_episode_masked(self, multi_episode_buffer):
        """Test with single episode selected."""
        mask = np.array([False, True, False])

        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            episode_mask=mask
        )

        # Should have sequences from episode 1 only (length 7)
        # 7 - 3 + 1 = 5 sequences
        assert len(sampler) == 5


class TestErrorHandling:
    """Test error conditions and invalid inputs."""

    def test_invalid_sequence_length_zero(self, simple_buffer):
        """Test that sequence_length=0 raises error."""
        with pytest.raises(AssertionError):
            SequenceSampler(
                replay_buffer=simple_buffer,
                sequence_length=0
            )

    def test_invalid_sequence_length_negative(self, simple_buffer):
        """Test that negative sequence_length raises error."""
        with pytest.raises(AssertionError):
            SequenceSampler(
                replay_buffer=simple_buffer,
                sequence_length=-5
            )

    def test_invalid_key(self, simple_buffer):
        """Test requesting non-existent key."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            keys=['nonexistent_key']
        )

        # Should fail when sampling
        with pytest.raises(KeyError):
            sampler.sample_sequence(0)

    def test_index_out_of_range(self, simple_buffer):
        """Test sampling with invalid index."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        with pytest.raises(IndexError):
            sampler.sample_sequence(len(sampler))

    def test_negative_index_out_of_range(self, simple_buffer):
        """Test sampling with negative index out of range."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        with pytest.raises(IndexError):
            sampler.sample_sequence(-len(sampler) - 1)


class TestDataIntegrity:
    """Test that sampled data maintains integrity."""

    def test_data_not_modified(self, simple_buffer):
        """Test that sampling doesn't modify buffer."""
        original_data = simple_buffer['observations'][:].copy()

        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        # Sample multiple times
        for i in range(min(5, len(sampler))):
            _ = sampler.sample_sequence(i)

        # Buffer data should be unchanged
        np.testing.assert_array_equal(
            simple_buffer['observations'][:],
            original_data
        )

    def test_sample_with_padding_independent(self, simple_buffer):
        """Test that samples with padding are independent copies."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=1  # Force padding to ensure copies
        )

        sample1 = sampler.sample_sequence(0)
        sample2 = sampler.sample_sequence(0)

        # With padding, new arrays are allocated, so should be independent
        sample1['observations'][0, 0] = 999.0

        # sample2 should be unchanged (they're different array objects)
        assert float(sample2['observations'][0, 0]) != 999.0

    def test_sample_no_padding_may_share_memory(self, simple_buffer):
        """Test that samples without padding may share memory (views).

        This is a performance optimization - when no padding is needed,
        the implementation may return views instead of copies.
        """
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            pad_before=0,
            pad_after=0
        )

        sample = sampler.sample_sequence(1)  # Middle sequence, no padding

        # Sample should have correct shape and values
        assert sample['observations'].shape == (3, 4)
        expected = np.arange(1*4, 4*4).reshape(3, 4).astype(np.float32)
        np.testing.assert_array_equal(sample['observations'], expected)

    def test_consistent_shapes(self, multi_episode_buffer):
        """Test that all samples have consistent shapes."""
        sampler = SequenceSampler(
            replay_buffer=multi_episode_buffer,
            sequence_length=3,
            pad_before=1,
            pad_after=1
        )

        shapes_obs = []
        shapes_actions = []

        for i in range(len(sampler)):
            sample = sampler.sample_sequence(i)
            shapes_obs.append(sample['observations'].shape)
            shapes_actions.append(sample['actions'].shape)

        # All should have same shape
        assert len(set(shapes_obs)) == 1
        assert len(set(shapes_actions)) == 1
        assert shapes_obs[0] == (3, 4)
        assert shapes_actions[0] == (3, 2)


class TestZarrBackend:
    """Test SequenceSampler with Zarr backend."""

    def test_zarr_buffer_basic(self):
        """Test basic sampling from Zarr buffer."""
        buffer = ReplayBuffer.create_empty_zarr()
        buffer.add_episode({
            'observations': np.arange(10 * 4).reshape(10, 4).astype(np.float32),
            'actions': np.arange(10 * 2).reshape(10, 2).astype(np.float32),
        })

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3
        )

        assert len(sampler) == 8
        sample = sampler.sample_sequence(0)
        assert sample['observations'].shape == (3, 4)

    def test_zarr_buffer_with_padding(self):
        """Test Zarr buffer with padding."""
        buffer = ReplayBuffer.create_empty_zarr()
        buffer.add_episode({
            'observations': np.arange(10 * 4).reshape(10, 4).astype(np.float32),
            'actions': np.arange(10 * 2).reshape(10, 2).astype(np.float32),
        })

        sampler = SequenceSampler(
            replay_buffer=buffer,
            sequence_length=3,
            pad_before=1,
            pad_after=1
        )

        assert len(sampler) == 10
        sample = sampler.sample_sequence(0)
        assert sample['observations'].shape == (3, 4)


class TestPerformanceOptimizations:
    """Test performance optimization features."""

    def test_indices_precomputed(self, simple_buffer):
        """Test that indices are precomputed during init."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3
        )

        # Indices should be available
        assert hasattr(sampler, 'indices')
        assert len(sampler.indices) > 0
        assert sampler.indices.shape[1] == 4  # 4 columns per index

    def test_key_first_k_reduces_load(self, simple_buffer):
        """Test that key_first_k actually uses partial data."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=10,
            key_first_k={'observations': 2}
        )

        sample = sampler.sample_sequence(0)

        # Should have loaded only 2 observations
        # Rest should be nan
        assert not np.any(np.isnan(sample['observations'][:2]))
        assert np.all(np.isnan(sample['observations'][2:]))

    def test_keys_stored_as_list(self, simple_buffer):
        """Test that keys are stored as list (not OmegaConf)."""
        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=3,
            keys=['observations', 'actions']
        )

        assert isinstance(sampler.keys, list)
        assert len(sampler.keys) == 2


@pytest.fixture
def buffer_with_language():
    """Buffer with language instructions."""
    buffer = ReplayBuffer.create_empty_numpy()
    data = {
        'observations': np.arange(10 * 4).reshape(10, 4).astype(np.float32),
        'actions': np.arange(10 * 2).reshape(10, 2).astype(np.float32),
        'language': np.array([f'instruction_{i}' for i in range(10)], dtype=object),
    }
    buffer.add_episode(data)
    return buffer


@pytest.fixture
def language_instructions_varied():
    """Language instructions with varying lengths."""
    return np.array([
        'short',
        'this is a much longer instruction with many words',
        'medium length instruction',
        'action_embedding',
        'another very very long instruction',
        'normal',
        'brief',
        'extended instruction here',
        'tiny',
        'final'
    ], dtype=object)


class TestLanguageSampling:
    """Test language instruction sampling."""

    def test_sample_language_as_strings(self, buffer_with_language):
        """Test that language instructions are sampled as strings."""
        sampler = SequenceSampler(
            replay_buffer=buffer_with_language,
            sequence_length=3,
            keys=['observations', 'actions', 'language']
        )

        sample = sampler.sample_sequence(0)

        assert 'language' in sample
        assert sample['language'].dtype == object
        assert len(sample['language']) == 3
        assert sample['language'][0] == 'instruction_0'
        assert sample['language'][1] == 'instruction_1'

    def test_sample_language_with_padding(self, buffer_with_language):
        """Test language sampling with padding at episode boundaries."""
        sampler = SequenceSampler(
            replay_buffer=buffer_with_language,
            sequence_length=3,
            pad_before=1,
            keys=['language'],
            pad_with_zeros=False
        )

        sample = sampler.sample_sequence(0)
        assert sample['language'][0] == sample['language'][1]

    def test_sample_language_variable_lengths(self, simple_buffer, language_instructions_varied):
        """Test sampling language instructions of varying lengths."""
        simple_buffer.data['language'] = language_instructions_varied

        sampler = SequenceSampler(
            replay_buffer=simple_buffer,
            sequence_length=4,
            keys=['language']
        )

        sample = sampler.sample_sequence(0)

        assert all(isinstance(s, str) for s in sample['language'])
        assert sample['language'][0] == 'short'
        assert 'much longer instruction' in sample['language'][1]

    def test_language_with_key_first_k(self, buffer_with_language):
        """Test that language respects key_first_k optimization."""
        sampler = SequenceSampler(
            replay_buffer=buffer_with_language,
            sequence_length=5,
            key_first_k={'language': 2},
            keys=['language']
        )

        sample = sampler.sample_sequence(0)

        assert sample['language'][0] == 'instruction_0'
        assert sample['language'][1] == 'instruction_1'
        assert sample['language'][2] == ''
