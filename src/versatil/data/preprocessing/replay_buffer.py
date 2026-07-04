import logging
import math
import numbers
import os
from functools import cached_property
from typing import Any

import numba
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import LocalStore, MemoryStore

from versatil.data.preprocessing.codecs import WebPCodec

WEBP_QUALITY = 99


def check_chunks_compatible(chunks: tuple[int, ...], shape: tuple[int, ...]) -> None:
    """Checks if given chunks are compatible with the array shape.

    Ensures that chunks and shape have the same dimensionality, each chunk size is a positive integer,
    and implicitly that chunks do not exceed shape dimensions (though not explicitly checked here).

    Args:
        chunks: Tuple of chunk sizes for each dimension.
        shape: Tuple of array shape dimensions.

    Raises:
        ValueError: If lengths differ or chunks are not positive.
        TypeError: If chunks are not integral.
    """
    if len(shape) != len(chunks):
        raise ValueError(
            f"Chunks dimensionality {len(chunks)} does not match shape dimensionality {len(shape)}."
        )
    for c in chunks:
        if not isinstance(c, numbers.Integral):
            raise TypeError(f"Chunk size must be an integer, got {type(c)}.")
        if c <= 0:
            raise ValueError(f"Chunk size must be positive, got {c}.")


def rechunk_recompress_array(
    group: zarr.Group,
    name: str,
    chunks: tuple | None = None,
    chunk_length: int | None = None,
    compressor: BloscCodec | WebPCodec | None = None,
    tmp_key: str = "_temp",
) -> zarr.Array:
    """Rechunk and/or recompress an existing zarr array.

    Preserves the original codec (BloscCodec or WebPCodec) if no explicit
    compressor is provided. For WebPCodec arrays the codec is read from the
    array's metadata codec pipeline since it is a serializer, not a compressor.

    Args:
        group: Zarr group containing the array.
        name: Name of the array to rechunk/recompress.
        chunks: New chunk sizes. Defaults to existing chunks.
        chunk_length: Shorthand to override only the first chunk dimension.
        compressor: New codec. Defaults to the array's existing codec.
        tmp_key: Temporary key used during rechunking (unused, kept for API).
    """
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks=chunks, shape=old_arr.shape)

    if compressor is None:
        if old_arr.compressors:
            compressor = old_arr.compressors[-1]
        else:
            compressor = _get_serializer_codec(old_arr)

    old_codec = (
        old_arr.compressors[-1]
        if old_arr.compressors
        else _get_serializer_codec(old_arr)
    )
    if (chunks == old_arr.chunks) and (compressor == old_codec):
        return old_arr

    # Manually rechunk/recompress instead of using zarr.copy
    # TODO: update when issue is closed on https://github.com/zarr-developers/zarr-python/issues/2407
    data = old_arr[:]
    del group[name]
    arr = _create_zarr_data_array(
        group=group, name=name, chunks=chunks, codec=compressor, data=data
    )
    return arr


def get_optimal_chunks(
    shape: tuple[int, ...],
    dtype: np.dtype | type,
    target_chunk_bytes: float = 2e6,
    max_chunk_length: int | None = None,
) -> tuple[int, ...]:
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


def _is_uint8_image_array(array: np.ndarray | zarr.Array) -> bool:
    """Check if an array is a uint8 image (4D with shape T, H, W, C)."""
    return len(array.shape) >= 4 and array.dtype == np.uint8


def _get_serializer_codec(array: zarr.Array) -> WebPCodec | None:
    """Extract the WebPCodec serializer from a zarr array's metadata codecs.

    WebPCodec is an ArrayBytesCodec (serializer), so it doesn't appear in
    ``arr.compressors``. This helper inspects the metadata codec pipeline.

    Args:
        array: Zarr array to inspect.

    Returns:
        WebPCodec instance if found, None otherwise.
    """
    for codec in array.metadata.codecs:
        if isinstance(codec, WebPCodec):
            return codec
    return None


def _create_zarr_data_array(
    group: zarr.Group,
    name: str,
    chunks: tuple,
    codec: BloscCodec | WebPCodec | None,
    data: np.ndarray | None = None,
    shape: tuple | None = None,
    dtype: np.dtype | type | None = None,
    fill_value: int | None = None,
) -> zarr.Array:
    """Create a zarr array using the appropriate codec parameter.

    WebPCodec is an ArrayBytesCodec and uses the ``serializer`` parameter.
    BloscCodec is a BytesBytesCodec and uses the ``compressors`` parameter.

    Args:
        group: Zarr group to create the array in.
        name: Array name.
        chunks: Chunk sizes.
        codec: WebPCodec, BloscCodec, or None.
        data: Data to store. Mutually exclusive with shape/dtype.
        shape: Shape for empty arrays.
        dtype: Dtype for empty arrays.
        fill_value: Optional fill value for empty arrays.
    """
    if isinstance(codec, WebPCodec):
        array_shape = data.shape if data is not None else shape
        webp_chunks = (1, *array_shape[1:])
        return group.create_array(
            name=name,
            data=data,
            shape=shape,
            chunks=webp_chunks,
            dtype=dtype,
            fill_value=fill_value,
            serializer=codec,
            compressors=None,
        )

    return group.create_array(
        name=name,
        data=data,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        compressors=codec,
    )


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

    def __init__(self, root: zarr.Group | dict[str, dict]):
        """Initializes the ReplayBuffer with a root group or dict.

        Validates presence of 'data', 'meta', 'episode_ends', and shape consistency across data arrays.

        Args:
            root: Zarr Group or dict with 'data' and 'meta' substructures.

        Raises:
            ValueError: If root is missing required keys or data shapes are inconsistent.
        """
        for required_key in ("data", "meta"):
            if required_key not in root:
                raise ValueError(f"Root must contain a '{required_key}' group.")
        if "episode_ends" not in root["meta"]:
            raise ValueError("Meta group must contain 'episode_ends'.")
        for key in root["data"]:
            value = root["data"][key]
            expected_length = root["meta"]["episode_ends"][-1]
            if value.shape[0] != expected_length:
                raise ValueError(
                    f"Data array '{key}' has length {value.shape[0]} but episode_ends indicates {expected_length}."
                )
        self.root = root

    @classmethod
    def create_empty_zarr(
        cls,
        storage: LocalStore | MemoryStore | None = None,
        root: zarr.Group | None = None,
    ) -> "ReplayBuffer":
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
        root.create_group("data", overwrite=False)
        meta = root.create_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            meta.create_array(
                "episode_ends", shape=(0,), dtype=np.int64, compressors=None
            )
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls) -> "ReplayBuffer":
        """Creates an empty NumPy-based ReplayBuffer.

        Initializes root as dict with empty 'data' dict and 'meta' with zero-length episode_ends array.

        Returns:
            ReplayBuffer instance.
        """
        root = {"data": {}, "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)}}
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group: zarr.Group) -> "ReplayBuffer":
        """Creates ReplayBuffer from an existing Zarr group.

        If 'data' missing, creates empty; else loads existing.

        Args:
            group: Zarr group.

        Returns:
            ReplayBuffer instance.
        """
        if "data" not in group:
            buffer = cls.create_empty_zarr(root=group)
        else:
            buffer = cls(root=group)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path: str) -> "ReplayBuffer":
        """Loads ReplayBuffer from a Zarr file path in read mode.

        Args:
            zarr_path: Path to Zarr directory.

        Returns:
            ReplayBuffer instance.
        """
        group = zarr.open_group(store=os.path.expanduser(zarr_path), mode="r")
        return cls.create_from_group(group)

    @classmethod
    def copy_from_store(
        cls,
        src_store: LocalStore | MemoryStore,
        store: LocalStore | MemoryStore | None = None,
        keys: list[str] | None = None,
        chunks: dict[str, tuple] | None = None,
        compressors: dict | str | BloscCodec | None = None,
        if_exists: str = "replace",
        **kwargs: Any,
    ) -> "ReplayBuffer":
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
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        src_root = zarr.open_group(store=src_store, mode="r")
        root = None
        if store is None:
            meta = {}
            for key in src_root["meta"]:
                value = src_root["meta"][key]
                if isinstance(value, zarr.Group):
                    continue
                if value.shape == ():
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]
            if keys is None:
                keys = src_root["data"].keys()
            data = {}
            for key in keys:
                arr = src_root["data"][key]
                data[key] = arr[:]
            root = {"meta": meta, "data": data}
        else:
            root = zarr.open_group(store=store, mode="w")
            meta_group = root.create_group("meta", overwrite=True)
            for key in src_root["meta"]:
                value = src_root["meta"][key]
                if isinstance(value, zarr.Group):
                    continue
                _ = meta_group.create_array(
                    name=key,
                    data=value[:] if value.shape != () else np.array(value),
                    chunks=value.shape,
                    compressors=None,
                )
            # Manually copy data
            # TODO: update when issue is closed on https://github.com/zarr-developers/zarr-python/issues/2407
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            for key in keys:
                value = src_root["data"][key]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value
                )
                _ = _create_zarr_data_array(
                    group=data_group,
                    name=key,
                    data=value[:],
                    chunks=cks,
                    codec=cpr,
                )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def copy_from_path(
        cls,
        zarr_path: str,
        backend: str | None = None,
        store: LocalStore | MemoryStore | None = None,
        keys: list[str] | None = None,
        chunks: dict[str, tuple] | None = None,
        compressors: dict | str | BloscCodec | None = None,
        if_exists: str = "replace",
        **kwargs: Any,
    ) -> "ReplayBuffer":
        """Copies ReplayBuffer from Zarr path, optionally to store or NumPy.

        Warns if backend specified (deprecated). Expands user path.

        Args:
            zarr_path: Source Zarr directory path.
            backend: Deprecated; use store=None for NumPy.
            store: Destination store (None for NumPy).
            keys: Optional list of data keys to copy (all if None).
            chunks: Dict of key to chunks tuple, or fallback to source/optimal.
            compressors: Dict or single compressor, resolved per key.
            if_exists: Zarr copy behavior ('replace', etc.).
            **kwargs: Passed to copy_from_store.

        Returns:
            ReplayBuffer instance.
        """
        if backend == "numpy":
            logging.warning(msg="backend argument is deprecated!")
            store = None
        group = zarr.open_group(store=os.path.expanduser(zarr_path), mode="r")
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
        store: LocalStore | MemoryStore,
        chunks: dict[str, tuple] | None = None,
        compressors: str | BloscCodec | dict | None = None,
        if_exists: str = "replace",
        **kwargs,
    ) -> LocalStore | MemoryStore:
        """Write the replay buffer into a Zarr store with optional chunking and codecs."""
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        root = zarr.open_group(store, mode="w" if if_exists == "replace" else "a")

        # Manually copy meta
        # TODO: update when issue is closed on https://github.com/zarr-developers/zarr-python/issues/2407
        meta_group = root.create_group("meta", overwrite=True)
        for key in self.meta:
            value = self.meta[key]
            data_to_save = value[:] if isinstance(value, zarr.Array) else value
            _ = meta_group.create_array(
                name=key,
                data=data_to_save,
                chunks=value.shape,
                compressors=None,
            )
        # Manually copy data
        # TODO: update when issue is closed on https://github.com/zarr-developers/zarr-python/issues/2407
        data_group = root.create_group("data", overwrite=True)
        for key in self.root["data"]:
            value = self.root["data"][key]
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value
            )
            data_to_save = value[:] if isinstance(value, zarr.Array) else value
            _ = _create_zarr_data_array(
                group=data_group,
                name=key,
                data=data_to_save,
                chunks=cks,
                codec=cpr,
            )
        return store

    def save_to_path(
        self,
        zarr_path: str,
        chunks: dict[str, tuple] | None = None,
        compressors: str | BloscCodec | dict | None = None,
        if_exists: str = "replace",
        **kwargs: Any,
    ) -> LocalStore:
        """Saves to a local Zarr path using LocalStore.

        Args:
            zarr_path: Destination path.
            chunks: Per-key chunks for arrays.
            compressors: Per-key or global compressor.
            if_exists: Zarr write mode ('replace', etc.).
            **kwargs: Passed to save_to_store.

        Returns:
            The store.
        """
        store = LocalStore(os.path.expanduser(zarr_path))
        return self.save_to_store(
            store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs
        )

    @staticmethod
    def resolve_compressor(
        compressor: str | BloscCodec = "default",
    ) -> BloscCodec:
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
        cls,
        compressors: dict | str | BloscCodec | WebPCodec,
        key: str,
        array: zarr.Array | np.ndarray,
    ) -> BloscCodec | WebPCodec:
        """Resolves compressor for a specific array key.

        From dict (key-specific), else global, fallback to array's or default.
        For uint8 image arrays (4D), returns WebPCodec by default.

        Args:
            compressors: Dict, str, BloscCodec, or WebPCodec.
            key: Array key.
            array: Array (Zarr or NumPy) for fallback.

        Returns:
            BloscCodec or WebPCodec.
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
            if _is_uint8_image_array(array):
                return WebPCodec(level=WEBP_QUALITY)
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(
        cls,
        chunks: dict[str, tuple] | tuple,
        key: str,
        array: zarr.Array | np.ndarray,
    ) -> tuple[int, ...]:
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
    def data(self) -> zarr.Group | dict[str, np.ndarray]:
        """Cached access to the 'data' group or dict."""
        return self.root["data"]

    @cached_property
    def meta(self) -> zarr.Group | dict[str, np.ndarray]:
        """Cached access to the 'meta' group or dict."""
        return self.root["meta"]

    def update_meta(
        self, data: dict[str, np.ndarray | list | int | float]
    ) -> zarr.Group | dict[str, np.ndarray]:
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
        np_data = {}
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
                    chunks=value.shape,
                    overwrite=True,
                    compressors=None,
                )
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_ends(self) -> zarr.Array | np.ndarray:
        """Property for the episode_ends array."""
        return self.meta["episode_ends"]

    def get_episode_idxs(self) -> np.ndarray:
        """Computes array mapping each step to its episode index.

        Uses Numba-jitted function for efficiency.

        Returns:
            NumPy array of episode indices per step.
        """

        @numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):  # This will work with numpy array
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        # Convert zarr array to numpy first
        episode_ends_np = self.episode_ends[:]
        return _get_episode_idxs(episode_ends_np)

    @property
    def backend(self) -> str:
        """Detects backend as 'zarr' or 'numpy' based on root type."""
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    def __repr__(self) -> str:
        """String representation: Zarr tree or default repr."""
        if self.backend == "zarr":
            try:
                return str(self.root.tree())
            except ImportError:
                # 'rich' library not installed, fall back to simple repr
                return f"<ReplayBuffer zarr backend: {self.n_episodes} episodes, {self.n_steps} steps>"
        else:
            return super().__repr__()

    def keys(self) -> list[str]:
        """Keys of the data dict/group."""
        return self.data.keys()

    def values(self) -> list[zarr.Array | np.ndarray]:
        """Values of the data dict/group."""
        if self.backend == "zarr":
            return self.data.array_values()
        else:
            return self.data.values()

    def items(self) -> list[tuple[str, zarr.Array | np.ndarray]]:
        """Items of the data dict/group."""
        if self.backend == "zarr":
            return self.data.members()
        else:
            return self.data.items()

    def __getitem__(self, key: str) -> zarr.Array | np.ndarray:
        """Getitem for data[key]."""
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """Contains check for data."""
        return key in self.data

    @property
    def n_steps(self) -> int:
        """Total steps: last episode_end or 0."""
        if self.episode_ends.shape[0] == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self) -> int:
        """Number of episodes: length of episode_ends."""
        return self.episode_ends.shape[0]

    @property
    def chunk_size(self) -> int | None:
        """First dimension chunk size of first data array (Zarr only)."""
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self) -> np.ndarray:
        """Array of episode lengths from diffs of episode_ends."""
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(
        self,
        data: dict[str, np.ndarray],
        chunks: dict[str, tuple] | None = None,
        compressors: str | BloscCodec | dict | None = None,
    ) -> None:
        """Adds an episode as dict of arrays, resizing all data arrays.

        Creates new keys if needed with resolved chunks/compressors.
        Appends to episode_ends, rechunks if grown significantly (Zarr).

        Args:
            data: Dict of key to NumPy array (consistent lengths).
            chunks: Per-key chunks for new arrays.
            compressors: Per-key or global for new arrays.

        Raises:
            ValueError: If empty data, inconsistent lengths, or mismatched shapes.
        """
        if chunks is None:
            chunks = {}
        if compressors is None:
            compressors = {}
        if len(data) == 0:
            raise ValueError("Episode data must not be empty.")
        is_zarr = self.backend == "zarr"
        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            if len(value.shape) < 1:
                raise ValueError(f"Array '{key}' must be at least 1-dimensional.")
            if episode_length is None:
                episode_length = len(value)
            elif episode_length != len(value):
                raise ValueError(
                    f"Inconsistent episode lengths: expected {episode_length}, got {len(value)} for key '{key}'."
                )
        new_len = curr_len + episode_length
        for key in data:
            value = data[key]
            new_shape = (new_len,) + value.shape[1:]
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value
                    )
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value
                    )
                    zarr_dtype = str if value.dtype == np.dtype("O") else value.dtype
                    arr = _create_zarr_data_array(
                        group=self.data,
                        name=key,
                        chunks=cks,
                        codec=cpr,
                        shape=new_shape,
                        dtype=zarr_dtype,
                        fill_value=0,
                    )
                else:
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                if value.shape[1:] != arr.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for '{key}': existing {arr.shape[1:]}, got {value.shape[1:]}."
                    )
                if is_zarr:
                    # Convert to plain Python ints for zarr v3
                    arr.resize(tuple(int(x) for x in new_shape))
                else:
                    arr.resize(new_shape, refcheck=False)
            arr[-value.shape[0] :] = value

        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len
        if is_zarr and episode_ends.chunks[0] < episode_ends.shape[0]:
            rechunk_recompress_array(
                self.meta, "episode_ends", chunk_length=int(episode_ends.shape[0] * 1.5)
            )

    def drop_episode(self) -> None:
        """Drops the last episode by resizing arrays backward.

        Raises:
            ValueError: If no episodes to drop.
        """
        is_zarr = self.backend == "zarr"
        episode_ends = self.episode_ends[:].copy()
        if len(episode_ends) == 0:
            raise ValueError("Cannot drop episode from an empty buffer.")
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key in self.data:
            value = self.data[key]
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                # Convert to plain Python ints for zarr v3
                value.resize(tuple(int(x) for x in new_shape))
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def pop_episode(self) -> dict[str, np.ndarray]:
        """Gets the last episode and drops it.

        Returns:
            Dict of arrays for the episode.

        Raises:
            ValueError: If no episodes to pop.
        """
        if self.n_episodes == 0:
            raise ValueError("Cannot pop episode from an empty buffer.")
        episode = self.get_episode(self.n_episodes - 1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data: dict[str, np.ndarray]) -> None:
        """Alias for add_episode.

        Args:
            data: As in add_episode.
        """
        self.add_episode(data)

    def get_episode(self, idx: int, copy: bool = False) -> dict[str, np.ndarray]:
        """Gets an episode by index as dict of sliced arrays.

        Args:
            idx: Episode index (handles negative via list).
            copy: If True, copy NumPy arrays.

        Returns:
            Dict of key to array slice.
        """
        idx = list(range(self.episode_ends.shape[0]))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result

    def get_episode_slice(self, idx: int) -> slice:
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

    def get_steps_slice(
        self,
        start: int,
        stop: int,
        step: int | None = None,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        """Gets dict of sliced arrays for a step range.

        Args:
            start: Start step index.
            stop: Stop step index.
            step: Step size for slicing.
            copy: Copy if NumPy backend.

        Returns:
            Dict of key to sliced array.
        """
        start = int(start) if isinstance(start, np.integer) else start
        stop = int(stop) if isinstance(stop, np.integer) else stop
        _slice = slice(start, stop, step)
        result = {}
        for key in self.data:
            value = self.data[key]
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_chunks(self) -> dict[str, tuple[int, ...]]:
        """Gets current chunks per data key (Zarr only).

        Returns:
            Dict of key to chunks tuple.

        Raises:
            RuntimeError: If not Zarr backend.
        """
        if self.backend != "zarr":
            raise RuntimeError("get_chunks is only supported on Zarr backend.")
        chunks = {}
        for key in self.data:
            value = self.data[key]
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict[str, tuple[int, ...]]) -> None:
        """Sets new chunks per data key if changed (Zarr only).

        Uses rechunk_recompress_array.

        Args:
            chunks: Dict of key to new chunks tuple.

        Raises:
            RuntimeError: If not Zarr backend.
        """
        if self.backend != "zarr":
            raise RuntimeError("set_chunks is only supported on Zarr backend.")
        for key in chunks:
            value = chunks[key]
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict[str, BloscCodec | WebPCodec | None]:
        """Gets current codec per data key (Zarr only).

        Returns BloscCodec for numerically compressed arrays, WebPCodec for
        image arrays using WebP serialization, or None if uncompressed.

        Returns:
            Dict mapping key to BloscCodec, WebPCodec, or None.

        Raises:
            RuntimeError: If not Zarr backend.
        """
        if self.backend != "zarr":
            raise RuntimeError("get_compressors is only supported on Zarr backend.")
        compressors = {}
        for key in self.data:
            array = self.data[key]
            if array.compressors:
                compressors[key] = array.compressors[-1]
            else:
                compressors[key] = _get_serializer_codec(array)
        return compressors

    def set_compressors(
        self, compressors: dict[str, str | BloscCodec | WebPCodec]
    ) -> None:
        """Sets new compressor per data key if changed (Zarr only).

        Resolves strings, uses rechunk_recompress_array.

        Args:
            compressors: Dict of key to compressor/str.

        Raises:
            RuntimeError: If not Zarr backend.
        """
        if self.backend != "zarr":
            raise RuntimeError("set_compressors is only supported on Zarr backend.")
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                arr_cpr = arr.compressors[-1] if arr.compressors else None
                if compressor != arr_cpr:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
