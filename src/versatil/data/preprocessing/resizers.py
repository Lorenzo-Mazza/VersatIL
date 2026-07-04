"""Per-camera image resizers for raw dataset preprocessing."""

import albumentations as A
import cv2

from versatil.data.metadata import CameraMetadata


def build_camera_resizer(camera_metadata: CameraMetadata) -> A.Resize | A.NoOp:
    """Build the resizer matching one camera's configured resolution.

    Depth images resize with nearest-neighbor interpolation so values stay
    valid distances; cameras without a configured resolution pass through.

    Args:
        camera_metadata: Metadata of the camera whose images are resized.
    """
    if camera_metadata.image_height is None or camera_metadata.image_width is None:
        return A.NoOp()
    if camera_metadata.is_depth:
        return A.Resize(
            height=camera_metadata.image_height,
            width=camera_metadata.image_width,
            interpolation=cv2.INTER_NEAREST,
        )
    return A.Resize(
        height=camera_metadata.image_height,
        width=camera_metadata.image_width,
    )
