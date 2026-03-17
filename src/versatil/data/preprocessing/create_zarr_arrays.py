"""Shared zarr replay buffer creation utilities.

Provides a unified pipeline for creating zarr stores from any dataset schema,
with WebP compression for uint8 images and lz4 for numerical arrays.
"""

import logging
from collections.abc import Iterable

import numpy as np
import zarr
import zarr.storage
from threadpoolctl import threadpool_limits
from zarr.codecs import BloscCodec, BloscShuffle

from versatil.data.preprocessing.codecs import WebPCodec
from versatil.data.raw.schemas.base import DatasetSchema

WEBP_QUALITY = 99


def is_uint8_image_spec(spec: dict) -> bool:
    """Check if a zarr array spec corresponds to a uint8 image array."""
    return len(spec["shape"]) == 4 and spec["dtype"] == "uint8"


def create_zarr_arrays(
    data_group: zarr.Group,
    schema: DatasetSchema,
    image_codec: WebPCodec,
    numeric_compressor: BloscCodec,
) -> None:
    """Create zarr arrays with codec selection based on data type.

    Uses WebP compression for uint8 image arrays and Blosc/lz4 for
    numerical arrays.

    Args:
        data_group: Zarr group to create arrays in.
        schema: Dataset schema providing array specifications.
        image_codec: WebP codec for uint8 image arrays.
        numeric_compressor: Blosc codec for numerical arrays.
    """
    specs = schema.get_zarr_array_specs()
    for key, spec in specs.items():
        dtype = str if spec["dtype"] == "str" else getattr(np, spec["dtype"])
        if is_uint8_image_spec(spec):
            # One image per chunk for WebP per-frame compression
            chunks = (1, *spec["shape"][1:])
            data_group.create_array(
                name=key,
                shape=spec["shape"],
                chunks=chunks,
                dtype=dtype,
                serializer=image_codec,
                compressors=None,
            )
        else:
            data_group.create_array(
                name=key,
                shape=spec["shape"],
                chunks=spec["chunks"],
                dtype=dtype,
                compressors=[numeric_compressor] if spec["needs_compressor"] else None,
            )


def create_zarr_replay_buffer(
    schema: DatasetSchema,
    episodes: Iterable[dict[str, np.ndarray]],
    total_episodes: int | None = None,
) -> None:
    """Create a zarr replay buffer from an iterable of episodes.

    Handles store creation, array setup with proper codecs, episode insertion,
    and metadata. Uses WebP for uint8 images and lz4 for numerical data.

    Args:
        schema: Dataset schema with zarr_path and array specs.
        episodes: Iterable yielding episode dicts mapping keys to arrays.
        total_episodes: Total number of episodes for progress reporting.
    """
    logging.info(
        msg=f"Creating Zarr dataset at {schema.zarr_path} "
        f"from schema {schema.__class__.__name__}"
    )

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    image_codec = WebPCodec(level=WEBP_QUALITY)
    numeric_compressor = BloscCodec(
        cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle
    )
    create_zarr_arrays(
        data_group=data_group,
        schema=schema,
        image_codec=image_codec,
        numeric_compressor=numeric_compressor,
    )

    episode_ends: list[int] = []
    cumulative_len = 0

    with threadpool_limits(limits=1):
        for i, episode_data in enumerate(episodes):
            if total_episodes is not None and i % 50 == 0:
                logging.info(msg=f"  Processing episode {i + 1}/{total_episodes}...")
            for key, array in episode_data.items():
                data_group[key].append(array)
            cumulative_len += len(next(iter(episode_data.values())))
            episode_ends.append(cumulative_len)

    meta_group.create_array(
        "episode_ends",
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )

    logging.info(
        msg=f"Created Zarr dataset with {len(episode_ends)} episodes, "
        f"{cumulative_len} total steps."
    )
