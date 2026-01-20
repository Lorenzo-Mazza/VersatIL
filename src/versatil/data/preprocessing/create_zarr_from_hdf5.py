"""Creates a Zarr-based replay buffer dataset from HDF5 files (e.g., LIBERO)."""

import albumentations as A
import cv2
import h5py
import numpy as np
import zarr
import zarr.storage
from threadpoolctl import threadpool_limits
from zarr.codecs import BloscCodec, BloscShuffle

from versatil.data.raw.schemas import Hdf5DatasetSchema


def create_replay_buffer_from_hdf5(schema: Hdf5DatasetSchema) -> None:
    """Creates a Zarr-based replay buffer from multiple HDF5 files.

    Args:
        schema: Hdf5DatasetSchema instance with HDF5 paths and zarr path configured
    """
    print(
        f"Creating Zarr dataset at {schema.zarr_path} from {len(schema.hdf5_paths)} HDF5 files"
    )
    print(f"Using dataset schema: {schema.__class__.__name__}")

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    episode_ends = []
    cumulative_len = 0
    compressor = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)

    cameras = schema.metadata.cameras
    if cameras:
        first_cam = next(iter(cameras.values()))
        image_width = first_cam.image_width
        image_height = first_cam.image_height
        resizer = A.Resize(height=image_height, width=image_width)
        depth_resizer = A.Resize(
            height=image_height, width=image_width, interpolation=cv2.INTER_NEAREST
        )
    else:
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

    _create_zarr_arrays(data_group=data_group, schema=schema, compressor=compressor)

    # Insert episodes from each HDF5 file into the zarr dataset
    with threadpool_limits(1):
        for hdf5_path in schema.hdf5_paths:
            print(f"  Processing: {hdf5_path}")
            with h5py.File(hdf5_path, "r") as f:
                demo_names = schema.get_demo_names(hdf5_path)
                demo_names_sorted = sorted(
                    demo_names, key=lambda x: int(x.split("_")[1])
                )

                for demo_name in demo_names_sorted:
                    demo_group = f[f"data/{demo_name}"]
                    episode_data = schema.extract_episode(
                        demo_group, resizer, depth_resizer
                    )

                    for key, array in episode_data.items():
                        data_group[key].append(array)

                    cumulative_len += len(next(iter(episode_data.values())))
                    episode_ends.append(cumulative_len)
                    # break
            # break

    meta_group.create_array(
        "episode_ends",
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )

    print(
        f"Created Zarr dataset with {len(episode_ends)} episodes, {cumulative_len} total steps."
    )


def _create_zarr_arrays(
    data_group: zarr.Group,
    schema: Hdf5DatasetSchema,
    compressor: BloscCodec,
) -> None:
    """Create zarr arrays based on schema configuration."""
    specs = schema.get_zarr_array_specs()
    for key, spec in specs.items():
        dtype = str if spec["dtype"] == "str" else getattr(np, spec["dtype"])
        data_group.create_array(
            key,
            shape=spec["shape"],
            chunks=spec["chunks"],
            dtype=dtype,
            compressors=[compressor] if spec["needs_compressor"] else None,
        )
