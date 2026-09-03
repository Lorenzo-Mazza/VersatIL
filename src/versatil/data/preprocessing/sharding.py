"""Storage-sharding policy for Zarr image arrays."""

import numbers

DEFAULT_IMAGE_FRAMES_PER_SHARD = 256


def is_uint8_image_spec(spec: dict) -> bool:
    """Check whether an array specification describes uint8 images.

    Args:
        spec: Zarr array specification containing shape and dtype entries.

    Returns:
        Whether the specification has a four-dimensional uint8 image layout.
    """
    return len(spec["shape"]) == 4 and spec["dtype"] == "uint8"


def validate_image_frames_per_shard(image_frames_per_shard: int | None) -> None:
    """Validate the requested number of image frames per shard.

    Args:
        image_frames_per_shard: Number of frames stored in each shard, or None to
            disable sharding.

    Raises:
        TypeError: If image_frames_per_shard is not an integer.
        ValueError: If image_frames_per_shard is not positive.
    """
    if image_frames_per_shard is None:
        return
    if isinstance(image_frames_per_shard, bool) or not isinstance(
        image_frames_per_shard, numbers.Integral
    ):
        raise TypeError(
            "image_frames_per_shard must be an integer or None, "
            f"got {type(image_frames_per_shard)}."
        )
    if image_frames_per_shard <= 0:
        raise ValueError(
            f"image_frames_per_shard must be positive, got {image_frames_per_shard}."
        )


def get_image_shard_shape(
    image_chunks: tuple[int, ...],
    image_frames_per_shard: int | None,
) -> tuple[int, ...] | None:
    """Resolve an image shard shape from its independently readable chunks.

    Args:
        image_chunks: Image chunk shape with time as its first dimension.
        image_frames_per_shard: Number of frames stored in each shard, or None to
            store every chunk separately.

    Returns:
        Shard shape, or None when sharding is disabled.

    Raises:
        TypeError: If image_frames_per_shard is not an integer.
        ValueError: If image_frames_per_shard is not a positive multiple of the
            image chunk length.
    """
    validate_image_frames_per_shard(
        image_frames_per_shard=image_frames_per_shard,
    )
    if image_frames_per_shard is None:
        return None
    image_chunk_length = image_chunks[0]
    if image_frames_per_shard % image_chunk_length != 0:
        raise ValueError(
            "image_frames_per_shard must be a multiple of image chunk "
            f"length {image_chunk_length}, got {image_frames_per_shard}."
        )
    return (int(image_frames_per_shard), *image_chunks[1:])
