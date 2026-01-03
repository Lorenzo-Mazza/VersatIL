from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
from zarr.storage import LocalStore, MemoryStore
from zarr.codecs import BloscCodec, BloscShuffle
import numpy as np
from functools import cached_property


def check_chunks_compatible(chunks: tuple, shape: tuple):
    """Checks if given chunks are compatible with the array shape.

    Ensures that chunks and shape have the same dimensionality, each chunk size is a positive integer,
    and implicitly that chunks do not exceed shape dimensions (though not explicitly checked here).

    Args:
        chunks: Tuple of chunk sizes for each dimension.
        shape: Tuple of array shape dimensions.

    Raises:
        AssertionError: If lengths differ, chunks are not positive integers, or not integral.
    """
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(
    group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"
):
    """Rechunks and/or recompresses a Zarr array in-place within its group.

    If no changes are needed (chunks and compressor match existing), returns the original array.
    Otherwise, moves the array to a temporary key, copies it back with new chunks/compressor,
    and deletes the temporary. Uses zarr.copy for efficient transfer.

    Args:
        group: Zarr group containing the array.
        name: Name of the array to modify.
        chunks: Optional new chunks tuple; defaults to existing.
        chunk_length: Optional length for the first dimension chunk; overrides chunks if set.
        compressor: Optional new compressor; defaults to existing.
        tmp_key: Temporary key for moving the array (default '_temp').

    Returns:
        The modified Zarr array.
    """
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressors[-1] if old_arr.compressors else None

    if (chunks == old_arr.chunks) and (
        compressor == (old_arr.compressors[-1] if old_arr.compressors else None)
    ):
        # no change
        return old_arr
    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressors=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """Computes optimal chunk sizes for an array to target a specific chunk byte size.

    Reverses shape to prioritize inner dimensions, caps outer dimension if max_chunk_length given.
    Finds the split point where adding another dimension exceeds target bytes, then adjusts the
    last chunk to meet the target. Pads with 1s if needed for dimensionality.

    Args:
        shape: Tuple of array dimensions.
        dtype: NumPy dtype of the array.
        target_chunk_bytes: Desired approximate bytes per chunk (default 2MB).
        max_chunk_length: Optional cap on the outermost chunk size.

    Returns:
        Tuple of optimal chunk sizes.
    """
    itemsize = np.dtype(dtype).itemsize
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i
    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    return chunks


class ReplayBuffer:
    """Manages a replay buffer dataset in Zarr or NumPy format for storing episodes of data.

    The buffer organizes data into 'data' (arrays for observations, actions, etc.) and 'meta'
    (episode_ends array tracking cumulative step counts per episode). Supports creation from
    scratch, loading from paths/stores, copying with optional rechunking/recompression,
    saving, adding/dropping episodes, slicing, and backend-agnostic access (Zarr for disk,
    NumPy for in-memory). Ensures data consistency across keys. Uses cached properties for
    efficiency. Supports custom chunks and compressors per array key.

    Attributes:
        root: Union[zarr.Group, Dict] holding 'data' and 'meta' groups/dicts.

    Methods:
        create_empty_zarr: Classmethod to create an empty Zarr-based buffer.
        create_empty_numpy: Classmethod to create an empty NumPy-based buffer.
        create_from_group: Classmethod to create from existing Zarr group.
        create_from_path: Classmethod to load from Zarr file path.
        copy_from_store: Classmethod to copy from source store with optional modifications.
        copy_from_path: Classmethod to copy from Zarr path.
        save_to_store: Save buffer to a store with optional rechunk/recompress.
        save_to_path: Save to a file path.
        resolve_compressor: Staticmethod to get Blosc compressor by name.
        _resolve_array_compressor: Classmethod to resolve compressor for a key.
        _resolve_array_chunks: Classmethod to resolve chunks for a key.
        data: Cached property for data group/dict.
        meta: Cached property for meta group/dict.
        update_meta: Update meta with new key-value pairs as arrays.
        episode_ends: Property for episode_ends array.
        get_episode_idxs: Get array mapping steps to episode indices (Numba-optimized).
        backend: Property detecting 'zarr' or 'numpy'.
        __repr__: String representation (Zarr tree or default).
        keys/values/items/__getitem__/__contains__: Dict-like access to data.
        n_steps: Total number of steps.
        n_episodes: Number of episodes.
        chunk_size: First chunk size if Zarr.
        episode_lengths: Array of episode lengths.
        add_episode: Append a new episode (dict of arrays).
        drop_episode: Remove the last episode.
        pop_episode: Get and remove the last episode.
        extend: Alias for add_episode.
        get_episode: Get episode by index as dict of arrays.
        get_episode_slice: Get slice for an episode.
        get_steps_slice: Get sliced data dict for step range.
        get_chunks: Get current chunks dict (Zarr only).
        set_chunks: Set new chunks per key (Zarr only).
        get_compressors: Get current compressors dict (Zarr only).
        set_compressors: Set new compressors per key (Zarr only).
    """

    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        """Initializes the ReplayBuffer with a root group or dict.

        Validates presence of 'data', 'meta', 'episode_ends', and shape consistency across data arrays.

        Args:
            root: Zarr Group or dict with 'data' and 'meta' substructures.
        """
        assert "data" in root
        assert "meta" in root
        assert "episode_ends" in root["meta"]
        for key, value in root["data"].items():
            assert value.shape[0] == root["meta"]["episode_ends"][-1]
        self.root = root

    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        """Creates an empty Zarr-based ReplayBuffer.

        Initializes 'data' and 'meta' groups, with 'episode_ends' as an empty int64 array.

        Args:
            storage: Optional Zarr store (defaults to MemoryStore).
            root: Optional existing Zarr group to use.

        Returns:
            ReplayBuffer instance.
        """
        if root is None:
            if storage is None:
                storage = MemoryStore()
            root = zarr.open_group(store=storage, mode="w")
        data = root.create_group("data", overwrite=False)
        meta = root.create_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            episode_ends = meta.create_array(
                "episode_ends", shape=(0,), dtype=np.int64, compressors=None
            )
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls):
        """Creates an empty NumPy-based ReplayBuffer.

        Initializes root as dict with empty 'data' dict and 'meta' with zero-length episode_ends array.

        Returns:
            ReplayBuffer instance.
        """
        root = {
            "data": dict(),
            "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)},
        }
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        """Creates ReplayBuffer from an existing Zarr group.

        If 'data' missing, creates empty; else loads existing.

        Args:
            group: Zarr group.
            **kwargs: Passed to create_empty_zarr if needed.

        Returns:
            ReplayBuffer instance.
        """
        if "data" not in group:
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, **kwargs):
        """Loads ReplayBuffer from a Zarr file path in read mode.

        Args:
            zarr_path: Path to Zarr directory.
            **kwargs: Passed to constructor.

        Returns:
            ReplayBuffer instance.
        """
        group = zarr.open_group(path=os.path.expanduser(zarr_path), mode="r")
        return cls.create_from_group(group, **kwargs)

    @classmethod
    def copy_from_store(
        cls,
        src_store,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, BloscCodec] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        """Copies a ReplayBuffer from source store to new store or NumPy dict.

        If store None, copies to NumPy dict; else to Zarr store. Selectively copies keys,
        applying custom chunks/compressors if specified, otherwise preserves or defaults.

        Args:
            src_store: Source Zarr store.
            store: Optional destination store (None for NumPy).
            keys: Optional list of data keys to copy (all if None).
            chunks: Dict of key to chunks tuple, or fallback to source/optimal.
            compressors: Dict or single compressor, resolved per key.
            if_exists: Zarr copy behavior ('replace', etc.).
            **kwargs: Unused.

        Returns:
            ReplayBuffer instance from copied data.
        """
        src_root = zarr.open_group(store=src_store, mode="r")
        root = None
        if store is None:
            meta = dict()
            for key, value in src_root["meta"].members():
                if isinstance(value, zarr.Group):
                    continue
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]
            if keys is None:
                keys = src_root["data"].keys()
            data = dict()
            for key in keys:
                arr = src_root["data"][key]
                data[key] = arr[:]
            root = {"meta": meta, "data": data}
        else:
            root = zarr.open_group(store=store, mode="w")
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=src_store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            for key in keys:
                value = src_root["data"][key]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value
                )
                value_cpr = value.compressors[-1] if value.compressors else None
                if cks == value.chunks and cpr == value_cpr:
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressors=cpr,
                        if_exists=if_exists,
                    )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def copy_from_path(
        cls,
        zarr_path,
        backend=None,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, BloscCodec] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        """Copies ReplayBuffer from Zarr path, optionally to store or NumPy.

        Warns if backend specified (deprecated). Expands user path.

        Args:
            zarr_path: Source Zarr directory path.
            backend: Deprecated; use store=None for NumPy.
            store: Destination store (None for NumPy).
            keys/chunks/compressors/if_exists: As in copy_from_store.
            **kwargs: Passed to copy_from_store.

        Returns:
            ReplayBuffer instance.
        """
        if backend == "numpy":
            print("backend argument is deprecated!")
            store = None
        group = zarr.open_group(os.path.expanduser(zarr_path), mode="r")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            **kwargs,
        )

    def save_to_store(
        self,
        store,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, BloscCodec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):

        """Saves the buffer to a Zarr store, with optional rechunk/recompress.

        Copies meta directly; for data, copies efficiently if no change, else uses zarr.copy
        with new params. Handles NumPy backend by creating arrays anew.

        Args:
            store: Destination Zarr store.
            chunks: Per-key chunks overrides.
            compressors: Per-key or global compressor.
            if_exists: Zarr copy behavior.
            **kwargs: Unused.

        Returns:
            The store.
        """
        root = zarr.open_group(store, mode="w" if if_exists == "replace" else "a")
        if self.backend == "zarr":
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
        else:
            meta_group = root.create_group("meta", overwrite=True)
            for key, value in self.root["meta"].items():
                _ = meta_group.create_array(
                    name=key,
                    data=value,
                    shape=value.shape,
                    chunks=value.shape,
                    compressors=None,
                )

        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value
            )
            if isinstance(value, zarr.Array):
                value_cpr = value.compressors[-1] if value.compressors else None
                if cks == value.chunks and cpr == value_cpr:
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressors=cpr,
                        if_exists=if_exists,
                    )
            else:
                _ = data_group.create_array(
                    name=key, data=value, chunks=cks, compressors=cpr
                )
        return store

    def save_to_path(
        self,
        zarr_path,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, BloscCodec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        """Saves to a local Zarr path using LocalStore.

        Args:
            zarr_path: Destination path.
            chunks/compressors/if_exists/**kwargs: As in save_to_store.

        Returns:
            The store.
        """
        store = LocalStore(os.path.expanduser(zarr_path))
        return self.save_to_store(
            store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs
        )

    @staticmethod
    def resolve_compressor(compressor="default"):
        """Resolves compressor string to BloscCodec instance.

        'default': lz4 level 5, no shuffle.
        'disk': zstd level 5, bitshuffle.

        Args:
            compressor: 'default', 'disk', or BloscCodec instance.

        Returns:
            BloscCodec.
        """
        if compressor == "default":
            compressor = BloscCodec(
                cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle
            )
        elif compressor == "disk":
            compressor = BloscCodec(
                cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle
            )
        return compressor

    @classmethod
    def _resolve_array_compressor(
        cls, compressors: Union[dict, str, BloscCodec], key, array
    ):
        """Resolves compressor for a specific array key.

        From dict (key-specific), else global, fallback to array's or default.

        Args:
            compressors: Dict, str, or BloscCodec.
            key: Array key.
            array: Array (Zarr or NumPy) for fallback.

        Returns:
            BloscCodec or None.
        """
        cpr = None
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressors[-1] if array.compressors else None
        else:
            cpr = cls.resolve_compressor(compressors)
        if cpr is None:
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: Union[dict, tuple], key, array):
        """Resolves chunks for a specific array key.

        From dict (key-specific), tuple (global), fallback to array's or optimal.

        Args:
            chunks: Dict or tuple.
            key: Array key.
            array: Array for shape/dtype and fallback.

        Returns:
            Tuple of chunks.

        Raises:
            TypeError: If chunks type unsupported.
        """
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    @cached_property
    def data(self):
        """Cached access to the 'data' group or dict."""
        return self.root["data"]

    @cached_property
    def meta(self):
        """Cached access to the 'meta' group or dict."""
        return self.root["meta"]

    def update_meta(self, data):
        """Updates meta with new key-value pairs as NumPy arrays.

        Converts values to arrays if needed, overwrites existing keys.
        For Zarr, creates arrays with no compression; for NumPy, dict update.

        Args:
            data: Dict of key to value (scalar/list/array).

        Returns:
            Updated meta group/dict.

        Raises:
            TypeError: If value can't be converted to non-object array.
        """
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr
        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                _ = meta_group.create_array(
                    name=key,
                    data=value,
                    shape=value.shape,
                    chunks=value.shape,
                    overwrite=True,
                    compressors=None,
                )
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_ends(self):
        """Property for the episode_ends array."""
        return self.meta["episode_ends"]

    def get_episode_idxs(self):
        """Computes array mapping each step to its episode index.

        Uses Numba-jitted function for efficiency.

        Returns:
            NumPy array of episode indices per step.
        """
        import numba

        numba.jit(nopython=True)

        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    @property
    def backend(self):
        """Detects backend as 'zarr' or 'numpy' based on root type."""
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    def __repr__(self) -> str:
        """String representation: Zarr tree or default repr."""
        if self.backend == "zarr":
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        """Keys of the data dict/group."""
        return self.data.keys()

    def values(self):
        """Values of the data dict/group."""
        return self.data.values()

    def items(self):
        """Items of the data dict/group."""
        return self.data.items()

    def __getitem__(self, key):
        """Getitem for data[key]."""
        return self.data[key]

    def __contains__(self, key):
        """Contains check for data."""
        return key in self.data

    @property
    def n_steps(self):
        """Total steps: last episode_end or 0."""
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        """Number of episodes: length of episode_ends."""
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        """First dimension chunk size of first data array (Zarr only)."""
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        """Array of episode lengths from diffs of episode_ends."""
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(
        self,
        data: Dict[str, np.ndarray],
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, BloscCodec, dict] = dict(),
    ):
        """Adds an episode as dict of arrays, resizing all data arrays.

        Creates new keys if needed with resolved chunks/compressors.
        Appends to episode_ends, rechunks if grown significantly (Zarr).

        Args:
            data: Dict of key to NumPy array (consistent lengths).
            chunks: Per-key chunks for new arrays.
            compressors: Per-key or global for new arrays.

        Raises:
            AssertionError: If empty data or inconsistent shapes/lengths.
        """
        assert len(data) > 0
        is_zarr = self.backend == "zarr"
        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert len(value.shape) >= 1
            if episode_length is None:
                episode_length = len(value)
            else:
                assert episode_length == len(value)
        new_len = curr_len + episode_length
        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value
                    )
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value
                    )
                    arr = self.data.create_array(
                        name=key,
                        shape=new_shape,
                        chunks=cks,
                        dtype=value.dtype,
                        compressors=cpr,
                        fill_value=0,
                    )
                else:
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            arr[-value.shape[0] :] = value

        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(
                    self.meta,
                    "episode_ends",
                    chunk_length=int(episode_ends.shape[0] * 1.5),
                )

    def drop_episode(self):
        """Drops the last episode by resizing arrays backward.

        Raises:
            AssertionError: If no episodes.
        """
        is_zarr = self.backend == "zarr"
        episode_ends = self.episode_ends[:].copy()
        assert len(episode_ends) > 0
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def pop_episode(self):
        """Gets the last episode and drops it.

        Returns:
            Dict of arrays for the episode.

        Raises:
            AssertionError: If no episodes.
        """
        assert self.n_episodes > 0
        episode = self.get_episode(self.n_episodes - 1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        """Alias for add_episode.

        Args:
            data: As in add_episode.
        """
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        """Gets an episode by index as dict of sliced arrays.

        Args:
            idx: Episode index (handles negative via list).
            copy: If True, copy NumPy arrays.

        Returns:
            Dict of key to array slice.
        """
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result

    def get_episode_slice(self, idx):
        """Gets the slice object for an episode's steps.

        Args:
            idx: Episode index.

        Returns:
            slice(start, end)
        """
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """Gets dict of sliced arrays for a step range.

        Args:
            start/stop/step: Slice parameters.
            copy: Copy if NumPy backend.

        Returns:
            Dict of key to sliced array.
        """
        _slice = slice(start, stop, step)
        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_chunks(self) -> dict:
        """Gets current chunks per data key (Zarr only).

        Returns:
            Dict of key to chunks tuple.

        Raises:
            AssertionError: If not Zarr backend.
        """
        assert self.backend == "zarr"
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict):
        """Sets new chunks per data key if changed (Zarr only).

        Uses rechunk_recompress_array.

        Args:
            chunks: Dict of key to new chunks tuple.

        Raises:
            AssertionError: If not Zarr or invalid chunks.
        """
        assert self.backend == "zarr"
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        """Gets current compressor per data key (Zarr only).

        Returns:
            Dict of key to compressor or None.
        """
        assert self.backend == "zarr"
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressors[-1] if value.compressors else None
        return compressors

    def set_compressors(self, compressors: dict):
        """Sets new compressor per data key if changed (Zarr only).

        Resolves strings, uses rechunk_recompress_array.

        Args:
            compressors: Dict of key to compressor/str.

        Raises:
            AssertionError: If not Zarr.
        """
        assert self.backend == "zarr"
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                arr_cpr = arr.compressors[-1] if arr.compressors else None
                if compressor != arr_cpr:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
