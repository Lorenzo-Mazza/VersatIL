"""Creates a Zarr-based replay buffer dataset from robot demonstration CSV files and associated images."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import zarr
import zarr.storage
from threadpoolctl import threadpool_limits
from zarr.codecs import BloscCodec, BloscShuffle

from refactoring.data.raw.schemas import CsvDatasetSchema


def create_replay_buffer(
        schema: CsvDatasetSchema,
        datasets_paths: list[str]
) -> None:
    """Creates a Zarr-based replay buffer using a Hydra-instantiated dataset schema.

    Args:
        schema: CsvDatasetSchema instance (instantiated by Hydra)
        datasets_paths: List of paths to episode CSV files
    """
    print(f"Creating Zarr dataset at {schema.zarr_path} with {len(datasets_paths)} episodes...")
    print(f"Using dataset schema: {schema.__class__.__name__}")

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    episode_ends = []
    cumulative_len = 0
    compressor = BloscCodec(cname='lz4', clevel=5, shuffle=BloscShuffle.noshuffle)

    if schema.metadata.image_width is None or schema.metadata.image_height is None:
        # Don't resize , use albumentations no-op
        resizer = A.NoOp()
        depth_resizer = A.NoOp()
    else:
        resizer = A.Resize(height=schema.metadata.image_height, width=schema.metadata.image_width)
        depth_resizer = A.Resize(
            height=schema.metadata.image_height,
            width=schema.metadata.image_width,
            interpolation=cv2.INTER_NEAREST
        )

    # Create empty zarr arrays based on schema
    _create_zarr_arrays(data_group=data_group, schema=schema, compressor=compressor)

    # Insert each episode into the zarr dataset
    with threadpool_limits(1):
        for path in sorted(datasets_paths, key=lambda x: int(Path(x).parent.name)):
            episode_df = pd.read_csv(path)
            episode_data = schema.extract_episode(episode_df, resizer, depth_resizer)

            for key, array in episode_data.items():
                data_group[key].append(array)

            cumulative_len += len(episode_df)
            episode_ends.append(cumulative_len)

    # Save metadata
    meta_group.create_array(
        'episode_ends',
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )

    print(f"Created Zarr dataset with {len(episode_ends)} episodes.")


def _create_zarr_arrays(
        data_group: zarr.Group,
        schema: CsvDatasetSchema,
        compressor: BloscCodec,
) -> None:
    """Create zarr arrays based on schema configuration."""
    specs = schema.get_zarr_array_specs()
    for key, spec in specs.items():
        dtype = str if spec['dtype'] == 'str' else getattr(np, spec['dtype'])
        data_group.create_array(
            key,
            shape=spec['shape'],
            chunks=spec['chunks'],
            dtype=dtype,
            compressors=[compressor] if spec['needs_compressor'] else None,
        )
