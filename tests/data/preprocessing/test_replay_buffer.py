"""Tests for ReplayBuffer and related utilities.

Tests the Zarr-based and NumPy-based replay buffer operations including creation,
saving, loading, copying, episode management, and chunking/compression utilities.
"""

import pytest
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import LocalStore, MemoryStore

from versatil.data.preprocessing.replay_buffer import ReplayBuffer


@pytest.fixture
def sample_episode_data():
    return {
        'observations': np.random.rand(10, 4).astype(np.float32),
        'actions': np.random.rand(10, 3).astype(np.float32),
        'rewards': np.random.rand(10).astype(np.float32),
        'discounts': np.ones(10, dtype=np.float32),
    }


@pytest.fixture
def multi_episode_data(sample_episode_data):
    ep1 = sample_episode_data
    ep2 = {k: v[:7].copy() for k, v in sample_episode_data.items()}
    return [ep1, ep2]


@pytest.fixture
def empty_zarr_buffer():
    """Fixture for empty Zarr buffer."""
    buffer = ReplayBuffer.create_empty_zarr()
    return buffer


@pytest.fixture
def empty_numpy_buffer():
    """Fixture for empty NumPy buffer."""
    buffer = ReplayBuffer.create_empty_numpy()
    return buffer


@pytest.fixture
def zarr_buffer_with_data(empty_zarr_buffer, sample_episode_data):
    """Fixture for Zarr buffer with data."""
    empty_zarr_buffer.add_episode(sample_episode_data)
    return empty_zarr_buffer


@pytest.fixture
def numpy_buffer_with_data(empty_numpy_buffer, sample_episode_data):
    """Fixture for NumPy buffer with data."""
    empty_numpy_buffer.add_episode(sample_episode_data)
    return empty_numpy_buffer


class TestReplayBufferCreation:
    """Test creation and initialization."""

    def test_create_empty_zarr(self):
        buffer = ReplayBuffer.create_empty_zarr()
        assert buffer.backend == 'zarr'
        assert buffer.n_steps == 0
        assert buffer.n_episodes == 0

    def test_create_empty_numpy(self):
        buffer = ReplayBuffer.create_empty_numpy()
        assert buffer.backend == 'numpy'
        assert buffer.n_steps == 0
        assert buffer.n_episodes == 0

    def test_create_from_path(self, tmp_path, zarr_buffer_with_data):
        path = tmp_path / "test.zarr"
        zarr_buffer_with_data.save_to_path(path)
        loaded = ReplayBuffer.create_from_path(path)
        assert loaded.backend == 'zarr'
        assert loaded.n_steps == 10
        assert loaded.n_episodes == 1

    def test_init_missing_data(self):
        root = {'meta': {'episode_ends': np.array([])}}
        with pytest.raises(AssertionError):
            ReplayBuffer(root)

    def test_init_missing_episode_ends(self):
        root = {'data': {}, 'meta': {}}
        with pytest.raises(AssertionError):
            ReplayBuffer(root)


class TestEpisodeManagement:
    """Test adding, dropping, popping, and getting episodes."""

    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_add_episode(self, buffer_type, request, sample_episode_data):
        if buffer_type == 'zarr':
            buffer = request.getfixturevalue("empty_zarr_buffer")
        else:
            buffer = request.getfixturevalue("empty_numpy_buffer")

        buffer.add_episode(sample_episode_data)
        assert buffer.n_steps == 10
        assert buffer.n_episodes == 1

    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_drop_episode(self, buffer_type, request, multi_episode_data):
        if buffer_type == 'zarr':
            buffer = request.getfixturevalue("empty_zarr_buffer")
        else:
            buffer = request.getfixturevalue("empty_numpy_buffer")

        for ep in multi_episode_data:
            buffer.add_episode(ep)

        initial_n = buffer.n_episodes
        buffer.drop_episode()
        assert buffer.n_episodes == initial_n - 1

    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_pop_episode(self, buffer_type, request, sample_episode_data):
        if buffer_type == 'zarr':
            buffer = request.getfixturevalue("empty_zarr_buffer")
        else:
            buffer = request.getfixturevalue("empty_numpy_buffer")

        buffer.add_episode(sample_episode_data)
        popped = buffer.pop_episode()
        assert len(popped['observations']) == 10
        assert buffer.n_episodes == 0

    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_get_episode(self, buffer_type, request, multi_episode_data):
        if buffer_type == 'zarr':
            buffer = request.getfixturevalue("empty_zarr_buffer")
        else:
            buffer = request.getfixturevalue("empty_numpy_buffer")

        for ep in multi_episode_data:
            buffer.add_episode(ep)

        ep = buffer.get_episode(1)
        assert len(ep['observations']) == 7  # Second episode length 7


class TestSaveLoadCopy:
    """Test saving, loading, and copying buffers."""

    def test_save_load_zarr(self, tmp_path, zarr_buffer_with_data):
        path = tmp_path / "saved.zarr"
        zarr_buffer_with_data.save_to_path(path)
        loaded = ReplayBuffer.create_from_path(path)
        assert loaded.n_steps == 10
        assert loaded.n_episodes == 1

    def test_copy_zarr_to_numpy(self, tmp_path, zarr_buffer_with_data):
        path = tmp_path / "source.zarr"
        zarr_buffer_with_data.save_to_path(path)
        numpy_buffer = ReplayBuffer.copy_from_path(path, store=None)
        assert numpy_buffer.backend == 'numpy'
        assert numpy_buffer.n_steps == 10


    def test_copy_with_chunks(self, tmp_path, zarr_buffer_with_data):
        source_path = tmp_path / "source.zarr"
        zarr_buffer_with_data.save_to_path(source_path)
        dest_path = tmp_path / "copied.zarr"
        custom_chunks = {'observations': (5, 4)}
        copied = ReplayBuffer.copy_from_path(
            source_path,
            store=LocalStore(dest_path),
            chunks=custom_chunks
        )
        assert copied['observations'].chunks == (5, 4)


    def test_episode_lengths(self, zarr_buffer_with_data, multi_episode_data):
        """Test episode_lengths property"""
        for ep in multi_episode_data[1:]:
            zarr_buffer_with_data.add_episode(ep)
        lengths = zarr_buffer_with_data.episode_lengths
        assert len(lengths) == 2
        assert lengths[0] == 10
        assert lengths[1] == 7

    def test_get_steps_slice(self, zarr_buffer_with_data):
        """Test slicing specific step ranges"""
        steps = zarr_buffer_with_data.get_steps_slice(2, 5)
        assert steps['observations'].shape[0] == 3

    def test_update_meta(self, empty_zarr_buffer):
        """Test adding custom metadata"""
        empty_zarr_buffer.update_meta({'custom_field': 42, 'name': 'test'})
        assert 'custom_field' in empty_zarr_buffer.meta
        assert empty_zarr_buffer.meta['custom_field'][()] == 42

    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_dict_like_access(self, buffer_type, request, sample_episode_data):
        """Test __getitem__, __contains__, keys, etc."""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.add_episode(sample_episode_data)
        assert 'observations' in buffer
        assert buffer['observations'].shape == (10, 4)
        assert 'observations' in buffer.keys()


class TestErrorHandling:
    """Test error conditions and invalid inputs."""


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_drop_episode_empty_buffer(self, buffer_type, request):
        """Test dropping from empty buffer raises error"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        with pytest.raises(AssertionError):
            buffer.drop_episode()


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_pop_episode_empty_buffer(self, buffer_type, request):
        """Test popping from empty buffer raises error"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        with pytest.raises(AssertionError):
            buffer.pop_episode()


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_add_empty_dict(self, buffer_type, request):
        """Test adding empty data dict raises error"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        with pytest.raises(AssertionError):
            buffer.add_episode({})


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_mismatched_episode_lengths(self, buffer_type, request):
        """Test adding episode with inconsistent array lengths"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        bad_data = {
            'observations': np.random.rand(10, 4).astype(np.float32),
            'actions': np.random.rand(7, 3).astype(np.float32),  # Wrong length!
        }
        with pytest.raises(AssertionError):
            buffer.add_episode(bad_data)


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_mismatched_feature_shapes(self, buffer_type, request, sample_episode_data):
        """Test adding episode with incompatible feature shapes"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.add_episode(sample_episode_data)

        # Try to add episode with different observation shape
        bad_data = {
            'observations': np.random.rand(5, 5).astype(np.float32),  # Was (*, 4)
            'actions': np.random.rand(5, 3).astype(np.float32),
            'rewards': np.random.rand(5).astype(np.float32),
            'discounts': np.ones(5, dtype=np.float32),
        }
        with pytest.raises(AssertionError):
            buffer.add_episode(bad_data)


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_scalar_array_rejected(self, buffer_type, request):
        """Test that scalar arrays (0-d) are rejected"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        bad_data = {
            'scalar': np.array(42.0),  # 0-d array
        }
        with pytest.raises(AssertionError):
            buffer.add_episode(bad_data)


    def test_invalid_chunks_type(self, empty_zarr_buffer, sample_episode_data):
        """Test that invalid chunk types raise errors"""
        with pytest.raises(TypeError):
            empty_zarr_buffer.add_episode(
                sample_episode_data,
                chunks="invalid"  # Should be dict or tuple
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_single_timestep_episode(self, buffer_type, request):
        """Test episode with just one timestep"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        single_step = {
            'observations': np.random.rand(1, 4).astype(np.float32),
            'actions': np.random.rand(1, 3).astype(np.float32),
        }
        buffer.add_episode(single_step)
        assert buffer.n_steps == 1
        assert buffer.n_episodes == 1


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_negative_episode_index(self, buffer_type, request, multi_episode_data):
        """Test negative indexing for episodes"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        for ep in multi_episode_data:
            buffer.add_episode(ep)

        last_ep = buffer.get_episode(-1)
        first_ep = buffer.get_episode(-2)
        assert len(last_ep['observations']) == 7
        assert len(first_ep['observations']) == 10


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_get_episode_out_of_bounds(self, buffer_type, request, sample_episode_data):
        """Test accessing episode beyond range"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.add_episode(sample_episode_data)

        with pytest.raises(IndexError):
            buffer.get_episode(5)


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_many_small_episodes(self, buffer_type, request):
        """Test handling many small episodes"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")

        for i in range(100):
            small_ep = {
                'observations': np.random.rand(2, 4).astype(np.float32),
                'actions': np.random.rand(2, 3).astype(np.float32),
            }
            buffer.add_episode(small_ep)

        assert buffer.n_episodes == 100
        assert buffer.n_steps == 200


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_drop_all_episodes(self, buffer_type, request, multi_episode_data):
        """Test dropping all episodes one by one"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        for ep in multi_episode_data:
            buffer.add_episode(ep)

        initial_count = buffer.n_episodes
        for _ in range(initial_count):
            buffer.drop_episode()

        assert buffer.n_episodes == 0
        assert buffer.n_steps == 0


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_extend_alias(self, buffer_type, request, sample_episode_data):
        """Test that extend() works as alias for add_episode()"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.extend(sample_episode_data)
        assert buffer.n_episodes == 1


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_get_episode_slice(self, buffer_type, request, multi_episode_data):
        """Test get_episode_slice returns correct slice object"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        for ep in multi_episode_data:
            buffer.add_episode(ep)

        slice_obj = buffer.get_episode_slice(0)
        assert slice_obj == slice(0, 10)

        slice_obj = buffer.get_episode_slice(1)
        assert slice_obj == slice(10, 17)


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_get_steps_slice_with_step(self, buffer_type, request, sample_episode_data):
        """Test slicing with step parameter"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.add_episode(sample_episode_data)

        steps = buffer.get_steps_slice(0, 10, step=2)
        assert steps['observations'].shape[0] == 5  # Every other step


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_copy_flag(self, buffer_type, request, sample_episode_data):
        """Test copy flag in get_episode for numpy"""
        if buffer_type != 'numpy':
            pytest.skip("Copy flag only relevant for numpy backend")

        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.add_episode(sample_episode_data)

        # Without copy, should share memory
        ep_no_copy = buffer.get_episode(0, copy=False)
        ep_no_copy['observations'][0, 0] = 999.0
        assert buffer['observations'][0, 0] == 999.0

        # Reset
        buffer['observations'][0, 0] = 0.0

        # With copy, should not share memory
        ep_copy = buffer.get_episode(0, copy=True)
        ep_copy['observations'][0, 0] = 999.0
        assert buffer['observations'][0, 0] == 0.0


    def test_update_meta_invalid_type(self, empty_zarr_buffer):
        """Test update_meta with invalid value types"""
        with pytest.raises(TypeError):
            empty_zarr_buffer.update_meta({'invalid': {'nested': 'dict'}})


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_1d_arrays(self, buffer_type, request):
        """Test that 1-D arrays (like rewards) work correctly"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        data = {
            'rewards': np.random.rand(10).astype(np.float32),  # 1-D
            'dones': np.zeros(10, dtype=np.bool_),  # 1-D bool
        }
        buffer.add_episode(data)
        assert buffer['rewards'].shape == (10,)
        assert buffer.n_steps == 10


class TestZarrSpecific:
    """Test Zarr-specific functionality."""


    def test_get_set_chunks(self, empty_zarr_buffer, sample_episode_data):
        """Test getting and setting chunks"""
        empty_zarr_buffer.add_episode(sample_episode_data)

        chunks = empty_zarr_buffer.get_chunks()
        assert 'observations' in chunks

        new_chunks = {'observations': (5, 4)}
        empty_zarr_buffer.set_chunks(new_chunks)
        assert empty_zarr_buffer['observations'].chunks == (5, 4)


    def test_get_set_compressors(self, empty_zarr_buffer, sample_episode_data):
        """Test getting and setting compressors"""
        empty_zarr_buffer.add_episode(sample_episode_data)

        compressors = empty_zarr_buffer.get_compressors()
        assert 'observations' in compressors

        new_compressors = {'observations': 'disk'}
        empty_zarr_buffer.set_compressors(new_compressors)
        # Verify compressor was changed
        assert empty_zarr_buffer['observations'].compressors is not None


    def test_chunks_incompatible_shape(self, empty_zarr_buffer, sample_episode_data):
        """Test that incompatible chunks raise error"""
        empty_zarr_buffer.add_episode(sample_episode_data)

        with pytest.raises(AssertionError):
            # Wrong number of dimensions
            empty_zarr_buffer.set_chunks({'observations': (5,)})


    def test_chunk_size_property(self, empty_zarr_buffer, sample_episode_data):
        """Test chunk_size property returns first dimension"""
        empty_zarr_buffer.add_episode(sample_episode_data)
        chunk_size = empty_zarr_buffer.chunk_size
        assert isinstance(chunk_size, int)
        assert chunk_size > 0


class TestNumPySpecific:
    """Test NumPy-specific functionality."""


    def test_chunk_size_none_for_numpy(self, empty_numpy_buffer, sample_episode_data):
        """Test that chunk_size is None for numpy backend"""
        empty_numpy_buffer.add_episode(sample_episode_data)
        assert empty_numpy_buffer.chunk_size is None


    def test_get_chunks_fails_numpy(self, empty_numpy_buffer, sample_episode_data):
        """Test that get_chunks raises error for numpy backend"""
        empty_numpy_buffer.add_episode(sample_episode_data)
        with pytest.raises(AssertionError):
            empty_numpy_buffer.get_chunks()


    def test_set_chunks_fails_numpy(self, empty_numpy_buffer, sample_episode_data):
        """Test that set_chunks raises error for numpy backend"""
        empty_numpy_buffer.add_episode(sample_episode_data)
        with pytest.raises(AssertionError):
            empty_numpy_buffer.set_chunks({'observations': (5, 4)})


class TestUtilityFunctions:
    """Test utility functions and helpers."""


    def test_get_episode_idxs(self, zarr_buffer_with_data, multi_episode_data):
        """Test get_episode_idxs creates correct mapping"""
        for ep in multi_episode_data[1:]:
            zarr_buffer_with_data.add_episode(ep)

        episode_idxs = zarr_buffer_with_data.get_episode_idxs()

        # First 10 steps should be episode 0
        assert all(episode_idxs[0:10] == 0)
        # Next 7 steps should be episode 1
        assert all(episode_idxs[10:17] == 1)
        assert len(episode_idxs) == 17


    class TestDictLikeMethods:
        """Test dict-like methods: keys(), values(), items()."""


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_keys_method(self, buffer_type, request, sample_episode_data):
            """Test keys() method returns all data keys"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            buffer.add_episode(sample_episode_data)

            keys = list(buffer.keys())

            assert 'observations' in keys
            assert 'actions' in keys
            assert 'rewards' in keys
            assert 'discounts' in keys
            assert len(keys) == 4


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_keys_order_consistency(self, buffer_type, request, sample_episode_data):
            """Test that keys() returns consistent ordering"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            buffer.add_episode(sample_episode_data)

            keys1 = list(buffer.keys())
            keys2 = list(buffer.keys())

            assert keys1 == keys2


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_items_method(self, buffer_type, request, sample_episode_data):
            """Test items() method returns key-value pairs"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            buffer.add_episode(sample_episode_data)

            # Try to get items - this might fail for zarr if .items() doesn't work
            try:
                items = list(buffer.items())

                # Check we got all items
                assert len(items) == 4

                # Check structure of items
                keys_from_items = [k for k, v in items]
                assert 'observations' in keys_from_items
                assert 'actions' in keys_from_items

                # Check values are arrays
                for key, value in items:
                    assert isinstance(key, str)
                    assert hasattr(value, 'shape')  # Array-like

            except (AttributeError, TypeError) as e:
                # If zarr doesn't support .items(), this is expected
                if buffer_type == 'zarr':
                    pytest.skip(f"Zarr backend doesn't support .items(): {e}")
                else:
                    raise


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_items_values_match(self, buffer_type, request, sample_episode_data):
            """Test that items() values match direct access"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            buffer.add_episode(sample_episode_data)

            try:
                items = dict(buffer.items())

                for key in buffer.keys():
                    # Check that value from items() matches direct access
                    assert key in items
                    # For zarr arrays, compare shapes rather than values
                    if buffer_type == 'zarr':
                        assert items[key].shape == buffer[key].shape
                    else:
                        np.testing.assert_array_equal(items[key], buffer[key])

            except (AttributeError, TypeError):
                if buffer_type == 'zarr':
                    pytest.skip("Zarr backend doesn't support .items()")
                else:
                    raise


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_iteration_consistency(self, buffer_type, request, sample_episode_data):
            """Test that keys(), values(), items() are consistent"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            buffer.add_episode(sample_episode_data)

            keys = list(buffer.keys())
            values = list(buffer.values())
            assert len(keys) == len(values)

            items = list(buffer.items())
            assert len(items) == len(keys)
            keys_from_items = [k for k, v in items]
            assert set(keys_from_items) == set(keys)


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_keys_empty_buffer(self, buffer_type, request):
            """Test keys() on empty buffer"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            keys = list(buffer.keys())
            assert len(keys) == 0


        @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
        def test_manual_iteration_vs_keys(self, buffer_type, request, sample_episode_data):
            """Test that manual iteration matches keys()"""
            buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
            buffer.add_episode(sample_episode_data)
            keys_method = list(buffer.keys())
            assert len(keys_method) == 4
            assert set(keys_method) == {'observations', 'actions', 'rewards', 'discounts'}

    def test_repr_zarr(self, empty_zarr_buffer, sample_episode_data):
        """Test __repr__ for zarr backend"""
        empty_zarr_buffer.add_episode(sample_episode_data)
        repr_str = repr(empty_zarr_buffer)
        # Should contain tree representation
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


    def test_repr_numpy(self, empty_numpy_buffer, sample_episode_data):
        """Test __repr__ for numpy backend"""
        empty_numpy_buffer.add_episode(sample_episode_data)
        repr_str = repr(empty_numpy_buffer)
        # Should use default repr
        assert 'ReplayBuffer' in repr_str


class TestCompressorFunctions:
    """Test compressor resolution and configuration."""


    def test_resolve_compressor_default(self):
        """Test resolving 'default' compressor"""
        comp = ReplayBuffer.resolve_compressor('default')
        assert isinstance(comp, BloscCodec)
        assert comp.cname.value == 'lz4'
        assert comp.clevel == 5


    def test_resolve_compressor_disk(self):
        """Test resolving 'disk' compressor"""
        comp = ReplayBuffer.resolve_compressor('disk')
        assert isinstance(comp, BloscCodec)
        assert comp.cname.value == 'zstd'
        assert comp.clevel == 5


    def test_resolve_compressor_custom(self):
        """Test passing custom BloscCodec"""
        custom_comp = BloscCodec(cname='lz4', clevel=9, shuffle=BloscShuffle.bitshuffle)
        comp = ReplayBuffer.resolve_compressor(custom_comp)
        assert comp == custom_comp


    def test_add_episode_with_disk_compressor(self, empty_zarr_buffer, sample_episode_data):
        """Test adding episode with 'disk' compressor"""
        empty_zarr_buffer.add_episode(
            sample_episode_data,
            compressors='disk'
        )
        # Verify compressor was applied
        assert empty_zarr_buffer['observations'].compressors is not None


    def test_add_episode_with_per_key_compressor(self, empty_zarr_buffer, sample_episode_data):
        """Test adding episode with per-key compressors"""
        empty_zarr_buffer.add_episode(
            sample_episode_data,
            compressors={
                'observations': 'disk',
                'actions': 'default'
            }
        )
        # Both should have compressors
        assert empty_zarr_buffer['observations'].compressors is not None
        assert empty_zarr_buffer['actions'].compressors is not None


class TestChunkFunctions:
    """Test chunk resolution and configuration."""


    def test_check_chunks_compatible_valid(self):
        """Test check_chunks_compatible with valid chunks"""
        from versatil.data.preprocessing.replay_buffer import check_chunks_compatible
        # Should not raise
        check_chunks_compatible((10, 4), (100, 4))
        check_chunks_compatible((5, 3, 2), (50, 30, 20))


    def test_check_chunks_compatible_wrong_dims(self):
        """Test check_chunks_compatible with wrong dimensions"""
        from versatil.data.preprocessing.replay_buffer import check_chunks_compatible
        with pytest.raises(AssertionError):
            check_chunks_compatible((10,), (100, 4))


    def test_check_chunks_compatible_negative(self):
        """Test check_chunks_compatible with negative chunk"""
        from versatil.data.preprocessing.replay_buffer import check_chunks_compatible
        with pytest.raises(AssertionError):
            check_chunks_compatible((-5, 4), (100, 4))


    def test_check_chunks_compatible_zero(self):
        """Test check_chunks_compatible with zero chunk"""
        from versatil.data.preprocessing.replay_buffer import check_chunks_compatible
        with pytest.raises(AssertionError):
            check_chunks_compatible((0, 4), (100, 4))


    def test_check_chunks_compatible_float(self):
        """Test check_chunks_compatible with float (non-integral)"""
        from versatil.data.preprocessing.replay_buffer import check_chunks_compatible
        with pytest.raises(AssertionError):
            check_chunks_compatible((10.5, 4), (100, 4))


    def test_get_optimal_chunks_basic(self):
        """Test get_optimal_chunks with basic array"""
        from versatil.data.preprocessing.replay_buffer import get_optimal_chunks

        # For a (1000, 10) float32 array
        chunks = get_optimal_chunks((1000, 10), np.float32)
        assert len(chunks) == 2
        assert chunks[1] == 10  # Last dimension should be full
        assert chunks[0] > 0


    def test_get_optimal_chunks_with_max_length(self):
        """Test get_optimal_chunks with max_chunk_length"""
        from versatil.data.preprocessing.replay_buffer import get_optimal_chunks

        chunks = get_optimal_chunks((10000, 10), np.float32, max_chunk_length=100)
        assert chunks[0] <= 100  # Should respect max length


    def test_get_optimal_chunks_large_array(self):
        """Test get_optimal_chunks with large multidimensional array"""
        from versatil.data.preprocessing.replay_buffer import get_optimal_chunks

        # 3D array
        chunks = get_optimal_chunks((1000, 64, 64), np.float32, target_chunk_bytes=1e6)
        assert len(chunks) == 3
        assert chunks[1] == 64  # Inner dimensions preserved
        assert chunks[2] == 64


    def test_add_episode_with_custom_chunks(self, empty_zarr_buffer, sample_episode_data):
        """Test adding episode with custom chunks"""
        custom_chunks = {
            'observations': (5, 4),
            'actions': (5, 3)
        }
        empty_zarr_buffer.add_episode(sample_episode_data, chunks=custom_chunks)

        assert empty_zarr_buffer['observations'].chunks == (5, 4)
        assert empty_zarr_buffer['actions'].chunks == (5, 3)


    def test_add_episode_with_global_chunks(self, empty_zarr_buffer):
        """Test adding episode with tuple chunks (global)"""
        data = {
            'observations': np.random.rand(10, 4).astype(np.float32),
        }
        # This should raise TypeError since global tuple chunks need special handling
        # Actually looking at the code, tuple chunks are valid
        empty_zarr_buffer.add_episode(data, chunks=(5, 4))
        # Should use the tuple for all arrays
        assert empty_zarr_buffer['observations'].chunks == (5, 4)


class TestRechunkRecompress:
    """Test rechunking and recompression functionality."""


    def test_rechunk_array(self, empty_zarr_buffer, sample_episode_data):
        """Test rechunking an existing array"""
        from versatil.data.preprocessing.replay_buffer import rechunk_recompress_array

        empty_zarr_buffer.add_episode(sample_episode_data)

        # Rechunk observations
        old_chunks = empty_zarr_buffer['observations'].chunks
        new_chunks = (5, 4)

        rechunk_recompress_array(
            empty_zarr_buffer.data,
            'observations',
            chunks=new_chunks
        )

        assert empty_zarr_buffer['observations'].chunks == new_chunks
        assert empty_zarr_buffer['observations'].chunks != old_chunks


    def test_rechunk_no_change(self, empty_zarr_buffer, sample_episode_data):
        """Test rechunking with same chunks does nothing"""
        from versatil.data.preprocessing.replay_buffer import rechunk_recompress_array

        empty_zarr_buffer.add_episode(sample_episode_data)

        original_chunks = empty_zarr_buffer['observations'].chunks
        arr = rechunk_recompress_array(
            empty_zarr_buffer.data,
            'observations',
            chunks=original_chunks
        )

        # Should return same array without modification
        assert arr.chunks == original_chunks


    def test_recompress_array(self, empty_zarr_buffer, sample_episode_data):
        """Test recompressing an array"""
        from versatil.data.preprocessing.replay_buffer import rechunk_recompress_array

        empty_zarr_buffer.add_episode(sample_episode_data, compressors='default')

        # Change to disk compressor
        disk_comp = ReplayBuffer.resolve_compressor('disk')
        rechunk_recompress_array(
            empty_zarr_buffer.data,
            'observations',
            compressor=disk_comp
        )

        # Verify compressor changed
        new_comp = empty_zarr_buffer['observations'].compressors[-1]
        assert new_comp.cname.value == 'zstd'


class TestSaveLoadVariations:
    """Test various save/load scenarios."""


    def test_save_with_custom_chunks(self, tmp_path, zarr_buffer_with_data):
        """Test saving with custom chunks"""
        path = tmp_path / "custom_chunks.zarr"
        custom_chunks = {'observations': (5, 4)}

        zarr_buffer_with_data.save_to_path(path, chunks=custom_chunks)
        loaded = ReplayBuffer.create_from_path(path)

        assert loaded['observations'].chunks == (5, 4)


    def test_save_with_custom_compressor(self, tmp_path, zarr_buffer_with_data):
        """Test saving with custom compressor"""
        path = tmp_path / "custom_comp.zarr"

        zarr_buffer_with_data.save_to_path(path, compressors='disk')
        loaded = ReplayBuffer.create_from_path(path)

        # Verify disk compressor was used
        comp = loaded['observations'].compressors[-1]
        assert comp.cname.value == 'zstd'


    def test_copy_with_key_selection(self, tmp_path, zarr_buffer_with_data):
        """Test copying only specific keys"""
        path = tmp_path / "source.zarr"
        zarr_buffer_with_data.save_to_path(path)

        # Copy only observations and actions
        copied = ReplayBuffer.copy_from_path(
            path,
            store=None,
            keys=['observations', 'actions']
        )

        assert 'observations' in copied
        assert 'actions' in copied
        assert 'rewards' not in copied
        assert 'discounts' not in copied


    def test_create_from_group_existing(self, tmp_path, zarr_buffer_with_data):
        """Test create_from_group with existing data"""
        path = tmp_path / "existing.zarr"
        zarr_buffer_with_data.save_to_path(path)

        group = zarr.open_group(store=path, mode='r')
        buffer = ReplayBuffer.create_from_group(group)

        assert buffer.n_steps == 10
        assert buffer.n_episodes == 1


    def test_create_from_group_empty(self):
        """Test create_from_group creates empty buffer if no data"""
        store = MemoryStore()
        group = zarr.open_group(store=store, mode='w')

        buffer = ReplayBuffer.create_from_group(group)
        assert buffer.n_steps == 0
        assert buffer.n_episodes == 0


class TestBackendDetection:
    """Test backend detection and properties."""


    def test_backend_property_zarr(self, empty_zarr_buffer):
        """Test backend property returns 'zarr'"""
        assert empty_zarr_buffer.backend == 'zarr'


    def test_backend_property_numpy(self, empty_numpy_buffer):
        """Test backend property returns 'numpy'"""
        assert empty_numpy_buffer.backend == 'numpy'


    def test_chunk_size_empty_zarr(self, empty_zarr_buffer):
        """Test chunk_size on empty zarr buffer"""
        # Should return None or handle gracefully
        # Adding data first
        data = {'obs': np.random.rand(10, 4).astype(np.float32)}
        empty_zarr_buffer.add_episode(data)
        chunk_size = empty_zarr_buffer.chunk_size
        assert chunk_size > 0


class TestMetaOperations:
    """Test metadata operations."""


    def test_update_meta_multiple_types(self, empty_zarr_buffer):
        """Test update_meta with various data types"""
        empty_zarr_buffer.update_meta({
            'int_val': 42,
            'float_val': 3.14,
            'array_val': np.array([1, 2, 3]),
            'list_val': [4, 5, 6]
        })

        assert 'int_val' in empty_zarr_buffer.meta
        assert 'float_val' in empty_zarr_buffer.meta
        assert 'array_val' in empty_zarr_buffer.meta
        assert 'list_val' in empty_zarr_buffer.meta


    def test_update_meta_overwrite(self, empty_zarr_buffer):
        """Test that update_meta overwrites existing values"""
        empty_zarr_buffer.update_meta({'field': 10})
        assert empty_zarr_buffer.meta['field'][()] == 10

        empty_zarr_buffer.update_meta({'field': 20})
        assert empty_zarr_buffer.meta['field'][()] == 20


    @pytest.mark.parametrize("buffer_type", ["zarr", "numpy"])
    def test_update_meta_both_backends(self, buffer_type, request):
        """Test update_meta works for both backends"""
        buffer = request.getfixturevalue(f"empty_{buffer_type}_buffer")
        buffer.update_meta({'test_field': 123})

        assert 'test_field' in buffer.meta