"""Tests for versatil.data.preprocessing.replay_buffer module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import MemoryStore

from versatil.data.preprocessing.codecs import WebPCodec
from versatil.data.preprocessing.replay_buffer import (
    ReplayBuffer,
    _create_zarr_data_array,
    _get_serializer_codec,
    _is_uint8_image_array,
    check_chunks_compatible,
    get_optimal_chunks,
    rechunk_recompress_array,
)


@pytest.fixture
def numpy_buffer_factory(rng: np.random.Generator) -> Callable[..., ReplayBuffer]:
    """Factory for creating numpy-backed ReplayBuffer with episodes."""

    def factory(
        num_episodes: int = 3,
        episode_length: int = 10,
        data_dim: int = 3,
    ) -> ReplayBuffer:
        buffer = ReplayBuffer.create_empty_numpy()
        for _ in range(num_episodes):
            episode = {
                "position": rng.standard_normal((episode_length, data_dim)).astype(np.float32),
                "gripper": rng.integers(0, 2, (episode_length, 1)).astype(np.float32),
            }
            buffer.add_episode(data=episode)
        return buffer

    return factory


@pytest.fixture
def zarr_buffer_factory(rng: np.random.Generator, tmp_path) -> Callable[..., ReplayBuffer]:
    """Factory for creating zarr-backed ReplayBuffer with episodes."""

    def factory(
        num_episodes: int = 3,
        episode_length: int = 10,
        data_dim: int = 3,
    ) -> ReplayBuffer:
        buffer = ReplayBuffer.create_empty_zarr()
        for _ in range(num_episodes):
            episode = {
                "position": rng.standard_normal((episode_length, data_dim)).astype(np.float32),
                "gripper": rng.integers(0, 2, (episode_length, 1)).astype(np.float32),
            }
            buffer.add_episode(data=episode)
        return buffer

    return factory


class TestCheckChunksCompatible:

    @pytest.mark.parametrize("chunks, shape, expectation", [
        ((10, 3), (100, 3), does_not_raise()),
        ((10,), (100, 3), pytest.raises(ValueError, match="Chunks dimensionality 1 does not match shape dimensionality 2")),
        ((0, 3), (100, 3), pytest.raises(ValueError, match="Chunk size must be positive, got 0")),
        ((-1, 3), (100, 3), pytest.raises(ValueError, match="Chunk size must be positive, got -1")),
    ])
    def test_chunk_shape_validation(self, chunks, shape, expectation):
        with expectation:
            check_chunks_compatible(chunks=chunks, shape=shape)

    def test_non_integral_chunk_raises_type_error(self):
        with pytest.raises(TypeError, match="Chunk size must be an integer"):
            check_chunks_compatible(chunks=(1.5, 3), shape=(100, 3))


class TestGetOptimalChunks:

    def test_returns_tuple_matching_shape_dimensions(self):
        chunks = get_optimal_chunks(shape=(1000, 100), dtype=np.float32)

        assert len(chunks) == 2

    def test_respects_max_chunk_length(self):
        chunks = get_optimal_chunks(
            shape=(10000, 3), dtype=np.float32, max_chunk_length=50,
        )

        assert chunks[0] <= 50

    def test_small_array_gets_single_chunk(self):
        chunks = get_optimal_chunks(
            shape=(10, 3), dtype=np.float32, target_chunk_bytes=2e6,
        )

        # 10 * 3 * 4 = 120 bytes << 2MB target → should chunk entire array
        assert chunks[0] == 10


class TestReplayBufferCreation:

    def test_create_empty_numpy(self):
        buffer = ReplayBuffer.create_empty_numpy()

        assert buffer.backend == "numpy"
        assert buffer.n_steps == 0
        assert buffer.n_episodes == 0

    def test_create_empty_zarr(self):
        buffer = ReplayBuffer.create_empty_zarr()

        assert buffer.backend == "zarr"
        assert buffer.n_steps == 0
        assert buffer.n_episodes == 0


class TestReplayBufferAddEpisode:

    @pytest.mark.parametrize("backend", ["numpy", "zarr"])
    def test_add_episode_increments_counts(
        self,
        backend: str,
        rng: np.random.Generator,
    ):
        buffer = (
            ReplayBuffer.create_empty_numpy()
            if backend == "numpy"
            else ReplayBuffer.create_empty_zarr()
        )
        episode = {"position": rng.standard_normal((5, 3)).astype(np.float32)}

        buffer.add_episode(data=episode)

        assert buffer.n_episodes == 1
        assert buffer.n_steps == 5

    @pytest.mark.parametrize("backend", ["numpy", "zarr"])
    def test_add_multiple_episodes_accumulates_steps(
        self,
        backend: str,
        rng: np.random.Generator,
    ):
        buffer = (
            ReplayBuffer.create_empty_numpy()
            if backend == "numpy"
            else ReplayBuffer.create_empty_zarr()
        )

        buffer.add_episode(data={"position": rng.standard_normal((5, 3)).astype(np.float32)})
        buffer.add_episode(data={"position": rng.standard_normal((7, 3)).astype(np.float32)})

        assert buffer.n_episodes == 2
        assert buffer.n_steps == 12


class TestReplayBufferDataAccess:

    @pytest.mark.parametrize("backend_fixture", ["numpy_buffer_factory", "zarr_buffer_factory"])
    def test_getitem_returns_data_array(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=2, episode_length=5, data_dim=3)

        position = buffer["position"]

        assert position.shape == (10, 3)

    @pytest.mark.parametrize("backend_fixture", ["numpy_buffer_factory", "zarr_buffer_factory"])
    def test_contains_returns_true_for_existing_key(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=1, episode_length=5)

        assert "position" in buffer
        assert "nonexistent" not in buffer

    @pytest.mark.parametrize("backend_fixture", ["numpy_buffer_factory", "zarr_buffer_factory"])
    def test_keys_returns_all_data_keys(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=1, episode_length=5)

        assert set(buffer.keys()) == {"position", "gripper"}


class TestReplayBufferEpisodeAccess:

    def test_get_episode_returns_correct_data(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=3, episode_length=5)

        episode = buffer.get_episode(idx=0)

        assert "position" in episode
        assert episode["position"].shape[0] == 5

    def test_get_episode_negative_index(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=3, episode_length=5)

        episode = buffer.get_episode(idx=-1)

        assert episode["position"].shape[0] == 5

    def test_episode_lengths_computed_correctly(
        self,
        rng: np.random.Generator,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        buffer.add_episode(data={"position": rng.standard_normal((5, 3)).astype(np.float32)})
        buffer.add_episode(data={"position": rng.standard_normal((8, 3)).astype(np.float32)})
        buffer.add_episode(data={"position": rng.standard_normal((3, 3)).astype(np.float32)})

        np.testing.assert_array_equal(buffer.episode_lengths, [5, 8, 3])

    def test_get_episode_slice_returns_correct_range(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=3, episode_length=5)

        episode_slice = buffer.get_episode_slice(idx=1)

        assert episode_slice == slice(5, 10)

    def test_get_episode_idxs_maps_steps_to_episodes(
        self,
        rng: np.random.Generator,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        buffer.add_episode(data={"position": rng.standard_normal((3, 2)).astype(np.float32)})
        buffer.add_episode(data={"position": rng.standard_normal((4, 2)).astype(np.float32)})

        episode_indices = buffer.get_episode_idxs()

        # Steps 0,1,2 → episode 0; steps 3,4,5,6 → episode 1
        np.testing.assert_array_equal(episode_indices, [0, 0, 0, 1, 1, 1, 1])

    def test_get_episode_with_copy_creates_independent_array(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)

        episode = buffer.get_episode(idx=0, copy=True)
        episode["position"][0, 0] = 999.0

        assert buffer["position"][0, 0] != 999.0


class TestReplayBufferDropAndPopEpisode:

    @pytest.mark.parametrize("backend_fixture", ["numpy_buffer_factory", "zarr_buffer_factory"])
    def test_drop_episode_removes_last(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=3, episode_length=5)

        buffer.drop_episode()

        assert buffer.n_episodes == 2
        assert buffer.n_steps == 10

    @pytest.mark.parametrize("backend_fixture", ["numpy_buffer_factory", "zarr_buffer_factory"])
    def test_pop_episode_returns_and_removes(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=3, episode_length=5)

        episode = buffer.pop_episode()

        assert buffer.n_episodes == 2
        assert "position" in episode
        assert episode["position"].shape[0] == 5

    @pytest.mark.parametrize("num_episodes, expectation", [
        (1, does_not_raise()),
        (0, pytest.raises(ValueError, match="empty buffer")),
    ])
    def test_drop_episode_validation(
        self,
        rng: np.random.Generator,
        num_episodes: int,
        expectation,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        for _ in range(num_episodes):
            buffer.add_episode(
                data={"position": rng.standard_normal((5, 3)).astype(np.float32)}
            )

        with expectation:
            buffer.drop_episode()

    @pytest.mark.parametrize("num_episodes, expectation", [
        (1, does_not_raise()),
        (0, pytest.raises(ValueError, match="empty buffer")),
    ])
    def test_pop_episode_validation(
        self,
        rng: np.random.Generator,
        num_episodes: int,
        expectation,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        for _ in range(num_episodes):
            buffer.add_episode(
                data={"position": rng.standard_normal((5, 3)).astype(np.float32)}
            )

        with expectation:
            buffer.pop_episode()


class TestReplayBufferSaveAndLoad:

    def test_save_and_load_roundtrip(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
        tmp_path,
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)
        save_path = str(tmp_path / "test_buffer.zarr")

        buffer.save_to_path(zarr_path=save_path)
        loaded = ReplayBuffer.create_from_path(zarr_path=save_path)

        assert loaded.n_episodes == 2
        assert loaded.n_steps == 10
        np.testing.assert_array_equal(
            loaded["position"][:],
            buffer["position"],
        )

    def test_copy_from_path_to_numpy(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
        tmp_path,
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5)
        save_path = str(tmp_path / "test_buffer.zarr")
        buffer.save_to_path(zarr_path=save_path)

        copied = ReplayBuffer.copy_from_path(zarr_path=save_path, store=None)

        assert copied.backend == "numpy"
        assert copied.n_episodes == 2
        assert copied.n_steps == 10


class TestReplayBufferRepr:

    def test_numpy_repr(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        representation = repr(buffer)

        assert isinstance(representation, str)

    def test_zarr_repr(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=1, episode_length=5)

        representation = repr(buffer)

        assert isinstance(representation, str)


class TestReplayBufferResolveCompressor:

    def test_default_returns_lz4(self):
        compressor = ReplayBuffer.resolve_compressor("default")

        assert compressor.cname.value == "lz4"

    def test_disk_returns_zstd(self):
        compressor = ReplayBuffer.resolve_compressor("disk")

        assert compressor.cname.value == "zstd"

    def test_passthrough_blosc_codec_instance(self):
        codec = BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.noshuffle)

        result = ReplayBuffer.resolve_compressor(codec)

        assert result is codec


class TestIsUint8ImageArray:

    def test_4d_uint8_returns_true(self, rng: np.random.Generator):
        array = rng.integers(0, 255, (10, 32, 32, 3), dtype=np.uint8)

        assert _is_uint8_image_array(array) is True

    def test_4d_float32_returns_false(self, rng: np.random.Generator):
        array = rng.standard_normal((10, 32, 32, 3)).astype(np.float32)

        assert _is_uint8_image_array(array) is False

    def test_2d_uint8_returns_false(self, rng: np.random.Generator):
        array = rng.integers(0, 255, (10, 3), dtype=np.uint8)

        assert _is_uint8_image_array(array) is False

    def test_5d_uint8_returns_true(self, rng: np.random.Generator):
        array = rng.integers(0, 255, (2, 5, 32, 32, 3), dtype=np.uint8)

        assert _is_uint8_image_array(array) is True


class TestGetSerializerCodec:

    def test_returns_none_for_blosc_compressed_array(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        data = rng.standard_normal((10, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(10, 3))

        result = _get_serializer_codec(group["test"])

        assert result is None

    def test_returns_webp_codec_for_webp_array(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        image_data = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        codec = WebPCodec(level=95)
        group.create_array(
            "test",
            data=image_data,
            chunks=(1, 32, 32, 3),
            serializer=codec,
            compressors=None,
        )

        result = _get_serializer_codec(group["test"])

        assert isinstance(result, WebPCodec)


class TestCreateZarrDataArray:

    def test_blosc_codec_preserves_chunks(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        data = rng.standard_normal((10, 3)).astype(np.float32)
        codec = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)

        array = _create_zarr_data_array(
            group=group, name="test", chunks=(5, 3), codec=codec, data=data,
        )

        assert array.shape == (10, 3)
        assert array.chunks == (5, 3)

    def test_webp_codec_overrides_chunks_to_single_image(
        self, rng: np.random.Generator,
    ):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        image_data = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        codec = WebPCodec(level=99)

        array = _create_zarr_data_array(
            group=group,
            name="test",
            chunks=(5, 32, 32, 3),
            codec=codec,
            data=image_data,
        )
        # WebPCodec forces chunks to (1, H, W, C) regardless of input
        assert array.chunks == (1, 32, 32, 3)

    def test_none_codec_creates_uncompressed_array(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        data = rng.standard_normal((10, 3)).astype(np.float32)

        array = _create_zarr_data_array(
            group=group, name="test", chunks=(5, 3), codec=None, data=data,
        )

        assert array.shape == (10, 3)

    def test_shape_dtype_without_data(self):
        group = zarr.open_group(store=MemoryStore(), mode="w")

        array = _create_zarr_data_array(
            group=group,
            name="test",
            chunks=(5, 3),
            codec=None,
            shape=(10, 3),
            dtype=np.float32,
            fill_value=0,
        )

        assert array.shape == (10, 3)
        assert array.dtype == np.float32


class TestRechunkRecompressArray:

    def test_returns_same_array_when_unchanged(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        codec = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)
        data = rng.standard_normal((20, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(10, 3), compressors=codec)

        result = rechunk_recompress_array(group=group, name="test")

        assert result.chunks == (10, 3)

    def test_rechunks_with_chunk_length(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        codec = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)
        data = rng.standard_normal((20, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(10, 3), compressors=codec)

        result = rechunk_recompress_array(
            group=group, name="test", chunk_length=5,
        )

        assert result.chunks == (5, 3)

    def test_recompresses_with_new_compressor(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        old_codec = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)
        new_codec = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle)
        data = rng.standard_normal((20, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(10, 3), compressors=old_codec)

        result = rechunk_recompress_array(
            group=group, name="test", compressor=new_codec,
        )

        assert result.compressors[-1].cname.value == "zstd"

    def test_preserves_data_after_rechunk(self, rng: np.random.Generator):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        codec = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)
        data = rng.standard_normal((20, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(10, 3), compressors=codec)

        result = rechunk_recompress_array(
            group=group, name="test", chunks=(5, 3),
        )

        np.testing.assert_array_almost_equal(result[:], data)


class TestReplayBufferCreateFromGroup:

    def test_creates_empty_buffer_when_data_missing(self):
        group = zarr.open_group(store=MemoryStore(), mode="w")

        buffer = ReplayBuffer.create_from_group(group=group)

        assert buffer.backend == "zarr"
        assert buffer.n_episodes == 0

    def test_loads_existing_buffer_when_data_present(
        self, rng: np.random.Generator,
    ):
        buffer = ReplayBuffer.create_empty_zarr()
        episode = {"position": rng.standard_normal((5, 3)).astype(np.float32)}
        buffer.add_episode(data=episode)

        loaded = ReplayBuffer.create_from_group(group=buffer.root)

        assert loaded.n_episodes == 1
        assert loaded.n_steps == 5


class TestReplayBufferUpdateMeta:

    def test_numpy_backend_updates_dict(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        buffer.update_meta(data={"custom_key": np.array([1, 2, 3])})

        np.testing.assert_array_equal(buffer.meta["custom_key"], [1, 2, 3])

    def test_zarr_backend_creates_array(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=1, episode_length=5)

        buffer.update_meta(data={"custom_key": np.array([10, 20])})

        np.testing.assert_array_equal(buffer.meta["custom_key"][:], [10, 20])

    def test_scalar_value_converted_to_array(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        buffer.update_meta(data={"scalar": 42})

        assert buffer.meta["scalar"] == 42

    def test_object_dtype_raises_type_error(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        bad_value = [{"nested": "dict"}]
        with pytest.raises(
            TypeError,
            match=re.escape(f"Invalid value type {type(bad_value)}"),
        ):
            buffer.update_meta(data={"bad": bad_value})


class TestReplayBufferValuesAndItems:

    @pytest.mark.parametrize(
        "backend_fixture",
        ["numpy_buffer_factory", "zarr_buffer_factory"],
    )
    def test_values_returns_all_data_arrays(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=1, episode_length=5)

        values = list(buffer.values())

        assert len(values) == 2

    @pytest.mark.parametrize(
        "backend_fixture",
        ["numpy_buffer_factory", "zarr_buffer_factory"],
    )
    def test_items_returns_key_array_pairs(
        self,
        backend_fixture: str,
        request: pytest.FixtureRequest,
    ):
        factory = request.getfixturevalue(backend_fixture)
        buffer = factory(num_episodes=1, episode_length=5)

        items = dict(buffer.items())

        assert "position" in items
        assert "gripper" in items


class TestReplayBufferChunkSize:

    def test_zarr_returns_chunk_size(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=1, episode_length=5)

        chunk_size = buffer.chunk_size

        assert isinstance(chunk_size, int)
        assert chunk_size > 0

    def test_numpy_returns_none(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        assert buffer.chunk_size is None


class TestReplayBufferExtend:

    def test_extend_is_alias_for_add_episode(
        self,
        rng: np.random.Generator,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        episode = {"position": rng.standard_normal((5, 3)).astype(np.float32)}

        buffer.extend(data=episode)

        assert buffer.n_episodes == 1
        assert buffer.n_steps == 5


class TestReplayBufferGetStepsSlice:

    def test_returns_sliced_data(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)

        result = buffer.get_steps_slice(start=2, stop=7)

        assert result["position"].shape == (5, 3)
        assert result["gripper"].shape == (5, 1)

    def test_step_parameter_applies_stride(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)

        result = buffer.get_steps_slice(start=0, stop=10, step=2)

        assert result["position"].shape == (5, 3)

    def test_copy_flag_copies_numpy_arrays(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5, data_dim=3)

        result = buffer.get_steps_slice(start=0, stop=5, copy=True)
        result["position"][0, 0] = 999.0

        assert buffer["position"][0, 0] != 999.0


class TestReplayBufferGetAndSetChunks:

    def test_get_chunks_returns_dict(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=1, episode_length=5, data_dim=3)

        chunks = buffer.get_chunks()

        assert "position" in chunks
        assert "gripper" in chunks
        assert isinstance(chunks["position"], tuple)

    def test_get_chunks_raises_on_numpy_backend(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        with pytest.raises(RuntimeError, match="Zarr backend"):
            buffer.get_chunks()

    def test_set_chunks_rechunks_array(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=2, episode_length=10, data_dim=3)

        buffer.set_chunks(chunks={"position": (5, 3)})

        assert buffer.get_chunks()["position"] == (5, 3)

    def test_set_chunks_raises_on_numpy_backend(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        with pytest.raises(RuntimeError, match="Zarr backend"):
            buffer.set_chunks(chunks={"position": (5, 3)})


class TestReplayBufferGetAndSetCompressors:

    def test_get_compressors_returns_dict(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=1, episode_length=5)

        compressors = buffer.get_compressors()

        assert "position" in compressors
        assert "gripper" in compressors

    def test_get_compressors_raises_on_numpy_backend(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        with pytest.raises(RuntimeError, match="Zarr backend"):
            buffer.get_compressors()

    def test_set_compressors_changes_codec(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=2, episode_length=10)

        buffer.set_compressors(compressors={"position": "disk"})

        compressors = buffer.get_compressors()
        assert compressors["position"].cname.value == "zstd"

    def test_set_compressors_raises_on_numpy_backend(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=1, episode_length=5)

        with pytest.raises(RuntimeError, match="Zarr backend"):
            buffer.set_compressors(compressors={"position": "disk"})


class TestResolveArrayChunks:

    def test_dict_with_matching_key(self, rng: np.random.Generator):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        result = ReplayBuffer._resolve_array_chunks(
            chunks={"position": (5, 3)}, key="position", array=array,
        )

        assert result == (5, 3)

    def test_dict_without_key_falls_back_to_optimal(
        self, rng: np.random.Generator,
    ):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        result = ReplayBuffer._resolve_array_chunks(
            chunks={}, key="position", array=array,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dict_without_key_uses_zarr_array_chunks(
        self, rng: np.random.Generator,
    ):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        data = rng.standard_normal((20, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(7, 3))

        result = ReplayBuffer._resolve_array_chunks(
            chunks={}, key="test", array=group["test"],
        )

        assert result == (7, 3)

    def test_tuple_chunks_used_directly(self, rng: np.random.Generator):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        result = ReplayBuffer._resolve_array_chunks(
            chunks=(10, 3), key="position", array=array,
        )

        assert result == (10, 3)

    def test_unsupported_type_raises_type_error(
        self, rng: np.random.Generator,
    ):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        with pytest.raises(
            TypeError,
            match=re.escape(f"Unsupported chunks type {type(5)}"),
        ):
            ReplayBuffer._resolve_array_chunks(
                chunks=5, key="position", array=array,
            )


class TestResolveArrayCompressor:

    def test_dict_with_matching_key(self, rng: np.random.Generator):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        result = ReplayBuffer._resolve_array_compressor(
            compressors={"position": "disk"}, key="position", array=array,
        )

        assert result.cname.value == "zstd"

    def test_dict_without_key_uses_zarr_array_compressor(
        self, rng: np.random.Generator,
    ):
        group = zarr.open_group(store=MemoryStore(), mode="w")
        codec = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle)
        data = rng.standard_normal((20, 3)).astype(np.float32)
        group.create_array("test", data=data, chunks=(10, 3), compressors=codec)

        result = ReplayBuffer._resolve_array_compressor(
            compressors={}, key="test", array=group["test"],
        )

        assert result.cname.value == "zstd"

    def test_string_compressor_resolved_globally(
        self, rng: np.random.Generator,
    ):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        result = ReplayBuffer._resolve_array_compressor(
            compressors="default", key="position", array=array,
        )

        assert result.cname.value == "lz4"

    def test_uint8_image_array_returns_webp_codec(
        self, rng: np.random.Generator,
    ):
        image_array = rng.integers(0, 255, (10, 32, 32, 3), dtype=np.uint8)

        result = ReplayBuffer._resolve_array_compressor(
            compressors={}, key="image", array=image_array,
        )

        assert isinstance(result, WebPCodec)

    def test_non_image_array_defaults_to_lz4(
        self, rng: np.random.Generator,
    ):
        array = rng.standard_normal((20, 3)).astype(np.float32)

        result = ReplayBuffer._resolve_array_compressor(
            compressors={}, key="position", array=array,
        )

        assert isinstance(result, BloscCodec)
        assert result.cname.value == "lz4"


class TestReplayBufferAddEpisodeErrors:

    def test_empty_data_raises(self):
        buffer = ReplayBuffer.create_empty_numpy()

        with pytest.raises(ValueError, match="must not be empty"):
            buffer.add_episode(data={})

    def test_inconsistent_episode_lengths_raises(
        self, rng: np.random.Generator,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        episode = {
            "position": rng.standard_normal((5, 3)).astype(np.float32),
            "gripper": rng.standard_normal((7, 1)).astype(np.float32),
        }

        with pytest.raises(ValueError, match="Inconsistent episode lengths"):
            buffer.add_episode(data=episode)

    def test_mismatched_shape_on_second_episode_raises(
        self, rng: np.random.Generator,
    ):
        buffer = ReplayBuffer.create_empty_numpy()
        buffer.add_episode(
            data={"position": rng.standard_normal((5, 3)).astype(np.float32)}
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            buffer.add_episode(
                data={"position": rng.standard_normal((5, 4)).astype(np.float32)}
            )


class TestReplayBufferCopyFromStoreToZarr:

    def test_copy_to_zarr_store_preserves_data(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
        tmp_path,
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)
        save_path = str(tmp_path / "source.zarr")
        buffer.save_to_path(zarr_path=save_path)

        destination_store = MemoryStore()
        copied = ReplayBuffer.copy_from_store(
            src_store=zarr.open_group(save_path, mode="r").store,
            store=destination_store,
        )

        assert copied.backend == "zarr"
        assert copied.n_episodes == 2
        assert copied.n_steps == 10

    def test_copy_with_selective_keys(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
        tmp_path,
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)
        save_path = str(tmp_path / "source.zarr")
        buffer.save_to_path(zarr_path=save_path)

        copied = ReplayBuffer.copy_from_store(
            src_store=zarr.open_group(save_path, mode="r").store,
            store=None,
            keys=["position"],
        )

        assert "position" in copied
        assert "gripper" not in copied


class TestReplayBufferSaveToStore:

    def test_save_zarr_buffer_to_store(
        self,
        zarr_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = zarr_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)
        destination_store = MemoryStore()

        buffer.save_to_store(store=destination_store)

        loaded = ReplayBuffer.create_from_group(
            zarr.open_group(store=destination_store, mode="r"),
        )
        assert loaded.n_episodes == 2
        assert loaded.n_steps == 10

    def test_save_with_custom_compressor(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)
        destination_store = MemoryStore()

        buffer.save_to_store(
            store=destination_store,
            compressors="disk",
        )

        loaded = ReplayBuffer.create_from_group(
            zarr.open_group(store=destination_store, mode="r"),
        )
        compressors = loaded.get_compressors()
        assert compressors["position"].cname.value == "zstd"

    def test_save_with_custom_chunks(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5, data_dim=3)
        destination_store = MemoryStore()

        buffer.save_to_store(
            store=destination_store,
            chunks={"position": (3, 3)},
        )

        loaded = ReplayBuffer.create_from_group(
            zarr.open_group(store=destination_store, mode="r"),
        )
        assert loaded.get_chunks()["position"] == (3, 3)


class TestReplayBufferCopyFromPathDeprecated:

    def test_backend_numpy_triggers_warning(
        self,
        numpy_buffer_factory: Callable[..., ReplayBuffer],
        tmp_path,
    ):
        buffer = numpy_buffer_factory(num_episodes=2, episode_length=5)
        save_path = str(tmp_path / "source.zarr")
        buffer.save_to_path(zarr_path=save_path)

        copied = ReplayBuffer.copy_from_path(
            zarr_path=save_path, backend="numpy",
        )

        assert copied.backend == "numpy"
        assert copied.n_episodes == 2