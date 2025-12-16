"""Creates a Zarr-based replay buffer dataset from HDF5 files (e.g., LIBERO)."""

import albumentations as A
import cv2
import h5py
import numpy as np
import zarr
import zarr.storage
from threadpoolctl import threadpool_limits
from zarr.codecs import BloscCodec, BloscShuffle

from refactoring.data.schemas.hdf5 import Hdf5DatasetSchema


def create_replay_buffer_from_hdf5(schema: Hdf5DatasetSchema) -> None:
    """Creates a Zarr-based replay buffer from an HDF5 file.

    Args:
        schema: Hdf5DatasetSchema instance with HDF5 and zarr paths configured
    """
    print(f"Creating Zarr dataset at {schema.zarr_path} from {schema.hdf5_path}")
    print(f"Using dataset schema: {schema.__class__.__name__}")

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    episode_ends = []
    cumulative_len = 0
    compressor = BloscCodec(cname='lz4', clevel=5, shuffle=BloscShuffle.noshuffle)

    obs = schema.raw_observations
    if obs.image_width is None or obs.image_height is None:
        resizer = A.NoOp()
        depth_resizer = A.NoOp()
    else:
        resizer = A.Resize(height=obs.image_height, width=obs.image_width)
        depth_resizer = A.Resize(
            height=obs.image_height,
            width=obs.image_width,
            interpolation=cv2.INTER_NEAREST
        )

    _create_zarr_arrays(data_group=data_group, schema=schema, compressor=compressor)

    # Insert each episode into the zarr dataset
    with threadpool_limits(1):
        with h5py.File(schema.hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

            for demo_name in demo_names:
                demo_group = f[f"data/{demo_name}"]
                episode_data = schema.extract_episode(demo_group, resizer, depth_resizer)

                for key, array in episode_data.items():
                    data_group[key].append(array)

                cumulative_len += len(next(iter(episode_data.values())))
                episode_ends.append(cumulative_len)

    meta_group.create_array(
        'episode_ends',
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )

    print(f"Created Zarr dataset with {len(episode_ends)} episodes, {cumulative_len} total steps.")


def _create_zarr_arrays(
        data_group: zarr.Group,
        schema: Hdf5DatasetSchema,
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